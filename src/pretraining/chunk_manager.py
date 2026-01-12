"""
Chunk-based storage for chess position data.

Stores positions in multiple chunk_XXXX.h5 files (~10000 examples each)
for memory-efficient streaming during training.

Uses HDF5 format with LZF compression for:
- ~30% smaller file size
- Memory-mapped reading (no RAM needed)
- Partial chunk loading
"""

import json
import os
from typing import Optional, Iterator, List
import numpy as np
import h5py


# Default chunk size (positions per chunk file)
DEFAULT_CHUNK_SIZE = 10000


class ChunkManager:
    """
    Manages chunked storage of chess positions using HDF5.

    Stores data in multiple chunk_XXXX.h5 files for:
    - Memory efficiency: only load what's needed
    - Compression: ~30% smaller than pickle
    - Robustness: corruption only affects one chunk
    - Parallel processing: chunks can be processed independently
    """

    def __init__(
        self,
        chunks_dir: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        verbose: bool = True,
        resume: bool = False,
    ):
        """
        Initialize chunk manager.

        Args:
            chunks_dir: Directory to store chunk files.
            chunk_size: Number of examples per chunk (~10000).
            verbose: Print progress information.
            resume: Resume from existing chunks (for incremental creation).
        """
        self.chunks_dir = chunks_dir
        self.chunk_size = chunk_size
        self.verbose = verbose

        # Buffer for accumulating examples before writing
        self._state_buffer: list[np.ndarray] = []
        self._policy_buffer: list[int] = []  # Store as indices
        self._value_buffer: list[float] = []

        self._chunk_count = 0
        self._total_examples = 0
        self._games_processed = 0
        self._processed_files: List[str] = []
        self._current_file: Optional[str] = None
        self._current_file_games: int = 0

        # Resume from existing chunks
        if resume:
            metadata = self.load_metadata(chunks_dir)
            if metadata:
                self._chunk_count = metadata.get("num_chunks", 0)
                self._total_examples = metadata.get("total_examples", 0)
                self._games_processed = metadata.get("games_processed", 0)
                self._processed_files = metadata.get("processed_files", [])
                self._current_file = metadata.get("current_file")
                self._current_file_games = metadata.get("current_file_games", 0)
                if verbose:
                    print(
                        f"Resuming from chunk {self._chunk_count}, {self._games_processed} games already processed"
                    )
                    if self._processed_files:
                        print(
                            f"  Already processed files: {len(self._processed_files)}"
                        )
                    if self._current_file:
                        print(
                            f"  Partial file: {self._current_file} ({self._current_file_games} games)"
                        )

    def get_chunk_path(self, chunk_idx: int) -> str:
        """Get path for a specific chunk file."""
        return os.path.join(self.chunks_dir, f"chunk_{chunk_idx:04d}.h5")

    def add_example(
        self,
        state: np.ndarray,
        policy_idx: int,
        value: float,
    ) -> None:
        """
        Add a single training example.

        Args:
            state: Board state encoding (C, H, W).
            policy_idx: Index of the move in policy vector.
            value: Game outcome from perspective of current player.
        """
        self._state_buffer.append(state)
        self._policy_buffer.append(policy_idx)
        self._value_buffer.append(value)

        # Flush when buffer is full
        if len(self._state_buffer) >= self.chunk_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Write buffer to an HDF5 chunk file."""
        if not self._state_buffer:
            return

        os.makedirs(self.chunks_dir, exist_ok=True)

        states = np.array(self._state_buffer, dtype=np.float32)
        policy_indices = np.array(self._policy_buffer, dtype=np.uint16)
        values = np.array(self._value_buffer, dtype=np.float32)

        # Validate data before writing to prevent corrupted chunks
        if np.isnan(states).any() or np.isinf(states).any():
            nan_count = np.isnan(states).sum() + np.isinf(states).sum()
            print(
                f"  WARNING: Skipping chunk {self._chunk_count} - {nan_count} NaN/Inf in states"
            )
            self._state_buffer.clear()
            self._policy_buffer.clear()
            self._value_buffer.clear()
            return

        if np.isnan(values).any() or np.isinf(values).any():
            nan_count = np.isnan(values).sum() + np.isinf(values).sum()
            print(
                f"  WARNING: Skipping chunk {self._chunk_count} - {nan_count} NaN/Inf in values"
            )
            self._state_buffer.clear()
            self._policy_buffer.clear()
            self._value_buffer.clear()
            return

        chunk_path = self.get_chunk_path(self._chunk_count)
        with h5py.File(chunk_path, "w") as f:
            # States with LZF compression (fast, good compression for float data)
            f.create_dataset("states", data=states, compression="lzf")
            # Policy indices - small, no compression needed
            f.create_dataset("policy_indices", data=policy_indices)
            # Values - small, no compression needed
            f.create_dataset("values", data=values)

        n_examples = len(self._state_buffer)
        self._total_examples += n_examples

        if self.verbose:
            print(f"  Wrote chunk_{self._chunk_count:04d}.h5 ({n_examples} examples)")

        # Clear buffers
        self._state_buffer.clear()
        self._policy_buffer.clear()
        self._value_buffer.clear()
        self._chunk_count += 1

        # Save metadata after each chunk for crash recovery
        self._save_metadata_incremental()

    def _save_metadata_incremental(self) -> None:
        """Save metadata incrementally for crash recovery."""
        metadata = {
            "num_chunks": self._chunk_count,
            "total_examples": self._total_examples,
            "chunk_size": self.chunk_size,
            "games_processed": self._games_processed,
            "processed_files": self._processed_files,
            "current_file": self._current_file,
            "current_file_games": self._current_file_games,
        }
        metadata_path = os.path.join(self.chunks_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def set_games_processed(self, count: int) -> None:
        """Set the number of games processed (for metadata)."""
        self._games_processed = count

    def set_processed_files(self, files: List[str]) -> None:
        """Set the list of processed PGN files (for resume tracking)."""
        self._processed_files = files

    def get_processed_files(self) -> List[str]:
        """Get the list of already processed PGN files."""
        return self._processed_files.copy()

    def set_current_file(self, file_path: Optional[str], games_processed: int = 0) -> None:
        """Set the current file being processed (for resume tracking)."""
        self._current_file = file_path
        self._current_file_games = games_processed

    def get_current_file(self) -> tuple[Optional[str], int]:
        """Get the current file and games processed in it."""
        return self._current_file, self._current_file_games

    def clear_current_file(self) -> None:
        """Clear current file tracking (when file is fully processed)."""
        self._current_file = None
        self._current_file_games = 0

    def finalize(self) -> int:
        """
        Finalize writing and flush remaining data.

        Returns:
            Total number of examples written.
        """
        # Flush remaining buffer
        if self._state_buffer:
            self._flush_buffer()

        # Write metadata as JSON
        metadata = {
            "num_chunks": self._chunk_count,
            "total_examples": self._total_examples,
            "chunk_size": self.chunk_size,
            "games_processed": self._games_processed,
            "processed_files": self._processed_files,
            "current_file": self._current_file,
            "current_file_games": self._current_file_games,
        }
        metadata_path = os.path.join(self.chunks_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if self.verbose:
            print(
                f"Finalized: {self._chunk_count} chunks, {self._total_examples} total examples, {self._games_processed} games"
            )

        return self._total_examples

    @classmethod
    def load_metadata(cls, chunks_dir: str) -> Optional[dict]:
        """
        Load metadata from chunks directory.

        Returns:
            Metadata dict or None if no chunks found.
        """
        json_path = os.path.join(chunks_dir, "metadata.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                return json.load(f)

        # No metadata file - check if chunks exist and regenerate
        if not os.path.exists(chunks_dir):
            return None

        chunk_paths = list(cls.iter_chunk_paths(chunks_dir))
        if not chunk_paths:
            return None

        # Regenerate metadata from existing chunks
        print(f"Regenerating metadata from {len(chunk_paths)} chunks...")
        total_examples = 0
        for chunk_path in chunk_paths:
            chunk = cls.load_chunk(chunk_path)
            total_examples += len(chunk["states"])

        metadata = {
            "num_chunks": len(chunk_paths),
            "total_examples": total_examples,
            "chunk_size": 10000,
        }

        # Save regenerated metadata
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"Regenerated metadata: {total_examples:,} examples in {len(chunk_paths)} chunks"
        )
        return metadata

    @classmethod
    def load_chunk(cls, chunk_path: str) -> dict:
        """Load a single chunk file."""
        with h5py.File(chunk_path, "r") as f:
            # Validate required fields
            required_fields = ["states", "policy_indices", "values"]
            missing = [field for field in required_fields if field not in f]
            if missing:
                raise ValueError(
                    f"Chunk {chunk_path} is missing required fields: {missing}. "
                    "Please regenerate chunks with the latest preprocessing."
                )

            data = {
                "states": f["states"][:],
                "policy_indices": f["policy_indices"][:],
                "values": f["values"][:],
            }

        # Validate data for NaN/Inf values
        if np.isnan(data["states"]).any() or np.isinf(data["states"]).any():
            raise ValueError(f"Chunk {chunk_path} contains NaN/Inf in states!")
        if np.isnan(data["values"]).any() or np.isinf(data["values"]).any():
            raise ValueError(f"Chunk {chunk_path} contains NaN/Inf in values!")

        return data

    @classmethod
    def iter_chunk_paths(cls, chunks_dir: str) -> Iterator[str]:
        """Iterate over chunk file paths in order."""
        if not os.path.exists(chunks_dir):
            return

        chunk_idx = 0
        while True:
            chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_idx:04d}.h5")
            if not os.path.exists(chunk_path):
                break
            yield chunk_path
            chunk_idx += 1

    @classmethod
    def count_chunks(cls, chunks_dir: str) -> int:
        """Count number of chunk files."""
        return sum(1 for _ in cls.iter_chunk_paths(chunks_dir))

    @classmethod
    def count_examples(cls, chunks_dir: str) -> int:
        """Count total examples across all chunks."""
        metadata = cls.load_metadata(chunks_dir)
        if metadata:
            return metadata["total_examples"]

        # Fallback: count manually
        total = 0
        for chunk_path in cls.iter_chunk_paths(chunks_dir):
            chunk = cls.load_chunk(chunk_path)
            total += len(chunk["states"])
        return total
