"""
PyTorch dataset for chess position pretraining.

Supports chunked storage for memory-efficient training on large datasets.
"""

import os
import random
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .pgn_processor import PGNProcessor
from .chunk_manager import ChunkManager, DEFAULT_CHUNK_SIZE
from .tactical_weighting import calculate_tactical_weight

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphazero.spatial_encoding import PositionHistory
from alphazero.move_encoding import encode_move_from_perspective, MOVE_ENCODING_SIZE


class ChessPositionDataset(Dataset):
    """
    PyTorch Dataset for chess positions with chunked storage.

    Data is stored in multiple chunk_XXXX.pkl files for memory efficiency.
    During training, chunks are loaded into RAM for fast access.
    """

    def __init__(
        self,
        pgn_path: str,
        chunks_dir: str | None = None,
        min_elo: int = 2200,
        max_games: int | None = None,
        max_positions: int | None = None,
        skip_first_n_moves: int = 8,
        history_length: int = 3,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        verbose: bool = True,
    ) -> None:
        """
        Initialize dataset.

        Args:
            pgn_path: Path to PGN file.
            chunks_dir: Directory for chunk storage (default: data/chunks).
            min_elo: Minimum player ELO.
            max_games: Maximum games to process (None = unlimited).
            max_positions: Maximum positions to load (None = unlimited).
            skip_first_n_moves: Skip opening moves (too theoretical).
            history_length: Number of history positions.
            chunk_size: Examples per chunk file.
            verbose: Print progress information.
        """
        self.pgn_path: str = pgn_path
        self.chunks_dir: str = chunks_dir or "data/chunks"
        self.min_elo: int = min_elo
        self.max_games: int | None = max_games
        self.max_positions: int | None = max_positions
        self.skip_first_n_moves: int = skip_first_n_moves
        self.history_length: int = history_length
        self.chunk_size: int = chunk_size
        self.verbose: bool = verbose

        # Data storage (loaded chunks)
        self._states: np.ndarray | None = None
        self._policy_indices: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._length: int = 0

        # Load or create data
        if self._chunks_exist():
            self._load_from_chunks()
        else:
            self._create_chunks_from_pgn()
            # Don't auto-load after creation - let caller decide (streaming vs RAM)

    def _chunks_exist(self) -> bool:
        """Check if chunks directory exists with valid data."""
        metadata: dict[str, Any] | None = ChunkManager.load_metadata(self.chunks_dir)
        return metadata is not None and metadata.get("num_chunks", 0) > 0

    def _create_chunks_from_pgn(self) -> None:
        """Process PGN file and create chunk files."""
        processor: PGNProcessor = PGNProcessor(
            self.pgn_path,
            min_elo=self.min_elo,
            max_games=self.max_games,
            skip_first_n_moves=self.skip_first_n_moves,
        )

        history: PositionHistory = PositionHistory(self.history_length)
        prev_game_idx: int = -1

        if self.verbose:
            print(f"Processing {self.pgn_path}...")
            estimated_total: int = processor.estimate_games()
            if self.max_games:
                print(
                    f"Processing up to {self.max_games} games (file contains ~{estimated_total})"
                )
            else:
                print(f"Estimated games in file: {estimated_total}")

        chunk_manager: ChunkManager = ChunkManager(
            self.chunks_dir,
            chunk_size=self.chunk_size,
            verbose=self.verbose,
        )

        position_count: int = 0
        for board, move, outcome in processor.process_all():
            if self.max_positions and position_count >= self.max_positions:
                break

            game_idx: int = processor.games_processed
            if game_idx != prev_game_idx:
                history.clear()
                prev_game_idx = game_idx

            history.push(board)
            state = history.encode(from_perspective=True)

            flip: bool = board.turn == False
            move_idx: int | None = encode_move_from_perspective(move, flip)
            if move_idx is None:
                continue

            value: float = outcome if board.turn else -outcome

            # Calculate tactical weight for this position
            weight: float = calculate_tactical_weight(board, move)

            chunk_manager.add_example(state, move_idx, value, weight=weight)
            position_count += 1

            if self.verbose and position_count % 50000 == 0:
                print(
                    f"  Processed {position_count:,} positions, "
                    f"{processor.games_processed} games, "
                    f"{processor.progress:.1%} complete"
                )

        total: int = chunk_manager.finalize()

        if self.verbose:
            print(
                f"Created {ChunkManager.count_chunks(self.chunks_dir)} chunks with {total:,} total positions"
            )

    def _load_from_chunks(self) -> None:
        """Load all chunks into RAM for fast training."""
        metadata: dict[str, Any] | None = ChunkManager.load_metadata(self.chunks_dir)
        if metadata is None:
            raise RuntimeError(f"No chunks found in {self.chunks_dir}")

        total_examples: int = metadata["total_examples"]
        if self.max_positions:
            total_examples = min(total_examples, self.max_positions)

        if self.verbose:
            # Estimate memory usage
            state_shape: int = (self.history_length + 1) * 12 + 6
            state_size: int = state_shape * 8 * 8 * 4  # float32
            # uint16 index
            policy_size: int = 2
            value_size: int = 4  # float32
            total_mem: float = (
                total_examples * (state_size + policy_size + value_size) / (1024**3)
            )
            print(f"Loading {total_examples:,} positions (~{total_mem:.1f} GB)...")

        # Pre-allocate arrays
        state_shape: int = (self.history_length + 1) * 12 + 6
        self._states = np.empty((total_examples, state_shape, 8, 8), dtype=np.float32)
        self._policy_indices = np.empty(total_examples, dtype=np.uint16)
        self._values = np.empty(total_examples, dtype=np.float32)

        # Load chunks
        loaded: int = 0
        for chunk_path in ChunkManager.iter_chunk_paths(self.chunks_dir):
            if loaded >= total_examples:
                break

            chunk: dict[str, np.ndarray] = ChunkManager.load_chunk(chunk_path)
            n: int = len(chunk["states"])

            # Limit to remaining needed
            remaining: int = total_examples - loaded
            if n > remaining:
                n = remaining

            self._states[loaded : loaded + n] = chunk["states"][:n]
            self._policy_indices[loaded : loaded + n] = chunk["policy_indices"][:n]
            self._values[loaded : loaded + n] = chunk["values"][:n]

            loaded += n

            if self.verbose:
                pct: int = loaded * 100 // total_examples
                print(f"\r  Loading chunks: {pct}%          ", end="", flush=True)

        if self.verbose:
            # Newline after progress
            print()
            print(f"Loaded {loaded:,} positions into RAM")

        self._length = loaded

    def __len__(self) -> int:
        """Return number of positions."""
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training example.

        Args:
            idx: Position index.

        Returns:
            Tuple of (state, policy, value) tensors.
        """
        assert self._states is not None
        assert self._policy_indices is not None
        assert self._values is not None
        state: torch.Tensor = torch.from_numpy(self._states[idx].copy()).float()

        # Reconstruct one-hot from index
        policy_idx = self._policy_indices[idx]
        policy: torch.Tensor = torch.zeros(MOVE_ENCODING_SIZE, dtype=torch.float32)
        policy[policy_idx] = 1.0

        value: torch.Tensor = torch.tensor(self._values[idx]).float()

        return state, policy, value

    def get_statistics(self) -> dict[str, int | float]:
        """Get dataset statistics."""
        assert self._values is not None
        return {
            "num_positions": self._length,
            "value_mean": float(np.mean(self._values[: self._length])),
            "value_std": float(np.std(self._values[: self._length])),
            "white_wins": int(np.sum(self._values[: self._length] > 0.5)),
            "black_wins": int(np.sum(self._values[: self._length] < -0.5)),
            "draws": int(np.sum(np.abs(self._values[: self._length]) <= 0.5)),
        }


class ChunkedIterableDataset(IterableDataset):
    """
    Iterable dataset that streams chunks for very large datasets.

    Use this when the full dataset doesn't fit in RAM.
    Loads one chunk at a time and shuffles within each chunk.
    """

    def __init__(
        self,
        chunks_dir: str,
        shuffle_chunks: bool = True,
        shuffle_within_chunk: bool = True,
    ) -> None:
        """
        Initialize streaming dataset.

        Args:
            chunks_dir: Directory containing chunk files.
            shuffle_chunks: Shuffle chunk order each epoch.
            shuffle_within_chunk: Shuffle examples within each chunk.
        """
        self.chunks_dir: str = chunks_dir
        self.shuffle_chunks: bool = shuffle_chunks
        self.shuffle_within_chunk: bool = shuffle_within_chunk

        # Get chunk paths
        self._chunk_paths: list[str] = list(ChunkManager.iter_chunk_paths(chunks_dir))
        self._metadata: dict[str, Any] | None = ChunkManager.load_metadata(chunks_dir)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate over all examples, streaming chunks."""
        chunk_paths: list[str] = self._chunk_paths.copy()

        if self.shuffle_chunks:
            random.shuffle(chunk_paths)

        for chunk_path in chunk_paths:
            chunk: dict[str, np.ndarray] = ChunkManager.load_chunk(chunk_path)
            n: int = len(chunk["states"])

            indices: list[int] = list(range(n))
            if self.shuffle_within_chunk:
                random.shuffle(indices)

            for idx in indices:
                state: torch.Tensor = torch.from_numpy(
                    chunk["states"][idx].copy()
                ).float()

                policy_idx = chunk["policy_indices"][idx]
                policy: torch.Tensor = torch.zeros(
                    MOVE_ENCODING_SIZE, dtype=torch.float32
                )
                policy[policy_idx] = 1.0

                value: torch.Tensor = torch.tensor(chunk["values"][idx]).float()

                yield state, policy, value

    def __len__(self) -> int:
        """Approximate length (may not be exact with streaming)."""
        if self._metadata:
            return self._metadata["total_examples"]
        return ChunkManager.count_examples(self.chunks_dir)
