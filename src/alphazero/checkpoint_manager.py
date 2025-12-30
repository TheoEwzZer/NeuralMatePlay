"""
Smart checkpoint management for training.

Handles efficient storage with milestone preservation and automatic cleanup.
"""

import os
import re
import pickle
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict

import torch


class CheckpointManager:
    """
    Manages training checkpoints with smart retention policies.

    - Pretraining: Keeps best model + milestones (1, 5, 10, 15...) + last 5
    - Training: Keeps milestones (every N) + last 5 checkpoints
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 5,
        milestone_interval: int = 5,
        verbose: bool = True,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints.
            keep_last_n: Number of recent checkpoints to keep.
            milestone_interval: Interval for milestone checkpoints.
            verbose: Print save/cleanup info.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self.milestone_interval = milestone_interval
        self.verbose = verbose

        # Thread safety
        self._lock = threading.Lock()
        self._writing_files: set = set()

        # Create directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _is_milestone(self, iteration: int) -> bool:
        """Check if iteration is a milestone."""
        return iteration == 1 or iteration % self.milestone_interval == 0

    def _log(self, message: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"  [Checkpoint] {message}")

    # =========================================================================
    # Pretraining Mode (best checkpoints)
    # =========================================================================

    def save_best_checkpoint(
        self,
        name: str,
        best_count: int,
        network: Any,
        state: Optional[Dict] = None,
    ) -> str:
        """
        Save a 'best' checkpoint for pretraining.

        Args:
            name: Base name (e.g., "pretrained").
            best_count: Number of times we've improved.
            network: Network to save (must have .save() or state_dict()).
            state: Optional training state dict.

        Returns:
            Path to saved checkpoint.
        """
        with self._lock:
            # Always save current best (overwritten each time)
            best_path = self.checkpoint_dir / f"{name}_best_network.pt"
            self._save_network(network, best_path)

            if state:
                state_path = self.checkpoint_dir / f"{name}_best_state.pkl"
                self._save_pickle(state, state_path)

            self._log(f"Saved {best_path.name} (improvement #{best_count})")

            # Always save numbered checkpoint for history
            numbered_path = (
                self.checkpoint_dir / f"{name}_best_{best_count}_network.pt"
            )
            self._save_network(network, numbered_path)

            if state:
                numbered_state = (
                    self.checkpoint_dir / f"{name}_best_{best_count}_state.pkl"
                )
                self._save_pickle(state, numbered_state)

            is_milestone = self._is_milestone(best_count)
            if is_milestone:
                self._log(f"Saved milestone: {numbered_path.name}")
            else:
                self._log(f"Saved checkpoint: {numbered_path.name}")

            # Cleanup old best checkpoints (keep milestones + last 5)
            self._cleanup_best_checkpoints(name)

            return str(best_path)

    def _cleanup_best_checkpoints(self, name: str) -> None:
        """Remove old best checkpoints, keeping milestones + last 5."""
        # Find all numbered best checkpoints
        pattern = re.compile(rf"{re.escape(name)}_best_(\d+)_network\.pt")
        best_counts = []

        for filepath in self.checkpoint_dir.glob(f"{name}_best_*_network.pt"):
            match = pattern.match(filepath.name)
            if match:
                best_counts.append(int(match.group(1)))

        if not best_counts:
            return

        # Determine which to keep
        keep = set()

        # Keep milestones
        for count in best_counts:
            if self._is_milestone(count):
                keep.add(count)

        # Keep last 5
        sorted_counts = sorted(best_counts)
        keep.update(sorted_counts[-self.keep_last_n:])

        # Delete the rest
        to_delete = set(best_counts) - keep
        deleted_count = 0

        for count in sorted(to_delete):
            for suffix in ["_network.pt", "_state.pkl"]:
                filepath = self.checkpoint_dir / f"{name}_best_{count}{suffix}"
                if filepath.exists():
                    try:
                        filepath.unlink()
                        deleted_count += 1
                    except OSError as e:
                        self._log(f"Warning: Could not delete {filepath.name}: {e}")

        if deleted_count > 0 and to_delete:
            self._log(f"Cleaned up best checkpoints: {sorted(to_delete)}")
            self._log(f"Keeping: milestones + last {self.keep_last_n} = {sorted(keep)}")

    # =========================================================================
    # Training Mode (iteration checkpoints)
    # =========================================================================

    def save_training_checkpoint(
        self,
        iteration: int,
        network: Any,
        state: Dict,
        buffer: Any = None,
    ) -> str:
        """
        Save a training checkpoint and cleanup old ones.

        Args:
            iteration: Current iteration number.
            network: Network to save.
            state: Training state (optimizer, scheduler, history, etc.).
            buffer: Optional replay buffer.

        Returns:
            Path to saved checkpoint.
        """
        with self._lock:
            prefix = f"iteration_{iteration}"

            # Mark files as being written
            files_to_write = [
                f"{prefix}_network.pt",
                f"{prefix}_state.pkl",
            ]
            if buffer is not None:
                files_to_write.append(f"{prefix}_buffer.pkl")

            for f in files_to_write:
                self._writing_files.add(f)

            try:
                # Save network
                network_path = self.checkpoint_dir / f"{prefix}_network.pt"
                self._save_network(network, network_path)

                # Save state
                state_path = self.checkpoint_dir / f"{prefix}_state.pkl"
                self._save_pickle(state, state_path)

                # Save buffer if provided
                if buffer is not None:
                    buffer_path = self.checkpoint_dir / f"{prefix}_buffer.pkl"
                    self._save_pickle(buffer, buffer_path)

                is_milestone = self._is_milestone(iteration)
                milestone_str = " (milestone)" if is_milestone else ""
                self._log(f"Saved iteration {iteration}{milestone_str}")

            finally:
                # Unmark files
                for f in files_to_write:
                    self._writing_files.discard(f)

            # Cleanup old checkpoints
            self._cleanup_training_checkpoints()

            return str(network_path)

    def _cleanup_training_checkpoints(self) -> None:
        """Remove old training checkpoints according to retention policy."""
        # Find all iteration checkpoints
        iterations = self._list_training_iterations()

        if not iterations:
            return

        # Determine which to keep
        keep = set()

        # Keep milestones
        for it in iterations:
            if self._is_milestone(it):
                keep.add(it)

        # Keep last N
        sorted_iterations = sorted(iterations)
        keep.update(sorted_iterations[-self.keep_last_n :])

        # Delete the rest
        to_delete = set(iterations) - keep
        deleted_count = 0

        for it in sorted(to_delete):
            prefix = f"iteration_{it}"
            for suffix in ["_network.pt", "_state.pkl", "_buffer.pkl"]:
                filepath = self.checkpoint_dir / f"{prefix}{suffix}"

                # Skip if file is being written
                if filepath.name in self._writing_files:
                    continue

                if filepath.exists():
                    try:
                        filepath.unlink()
                        deleted_count += 1
                    except OSError as e:
                        self._log(f"Warning: Could not delete {filepath.name}: {e}")

        if deleted_count > 0 and to_delete:
            self._log(f"Cleaned up iterations: {sorted(to_delete)}")
            self._log(f"Keeping: milestones + last {self.keep_last_n} = {sorted(keep)}")

    def _list_training_iterations(self) -> List[int]:
        """List all training iteration numbers."""
        iterations = set()
        pattern = re.compile(r"iteration_(\d+)_network\.pt")

        for filepath in self.checkpoint_dir.glob("iteration_*_network.pt"):
            match = pattern.match(filepath.name)
            if match:
                iterations.add(int(match.group(1)))

        return list(iterations)

    # =========================================================================
    # Loading & Utilities
    # =========================================================================

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the most recent training checkpoint.

        Returns:
            Prefix of latest checkpoint (e.g., "iteration_17") or None.
        """
        iterations = self._list_training_iterations()
        if not iterations:
            return None

        latest = max(iterations)
        return f"iteration_{latest}"

    def get_latest_iteration(self) -> int:
        """Get the latest iteration number, or 0 if none."""
        iterations = self._list_training_iterations()
        return max(iterations) if iterations else 0

    def list_checkpoints(self) -> List[Tuple[str, int]]:
        """
        List all checkpoints with their iteration/count.

        Returns:
            List of (prefix, number) tuples.
        """
        checkpoints = []

        # Training checkpoints
        for it in self._list_training_iterations():
            checkpoints.append((f"iteration_{it}", it))

        # Best checkpoints
        for filepath in self.checkpoint_dir.glob("*_best_*_network.pt"):
            match = re.search(r"_best_(\d+)_network\.pt", filepath.name)
            if match:
                count = int(match.group(1))
                name = filepath.name.replace(f"_best_{count}_network.pt", "")
                checkpoints.append((f"{name}_best_{count}", count))

        return sorted(checkpoints, key=lambda x: x[1])

    def load_training_checkpoint(
        self,
        iteration: Optional[int] = None,
        load_buffer: bool = True,
    ) -> Optional[Dict]:
        """
        Load a training checkpoint.

        Args:
            iteration: Specific iteration to load, or None for latest.
            load_buffer: Whether to load replay buffer.

        Returns:
            Dict with 'network_path', 'state', 'buffer' (if loaded), 'iteration'.
        """
        if iteration is None:
            iteration = self.get_latest_iteration()
            if iteration == 0:
                return None

        prefix = f"iteration_{iteration}"
        network_path = self.checkpoint_dir / f"{prefix}_network.pt"
        state_path = self.checkpoint_dir / f"{prefix}_state.pkl"
        buffer_path = self.checkpoint_dir / f"{prefix}_buffer.pkl"

        if not network_path.exists():
            self._log(f"Checkpoint not found: {prefix}")
            return None

        result = {
            "network_path": str(network_path),
            "iteration": iteration,
            "state": None,
            "buffer": None,
        }

        # Load state
        if state_path.exists():
            result["state"] = self._load_pickle_safe(state_path)

        # Load buffer
        if load_buffer and buffer_path.exists():
            result["buffer"] = self._load_pickle_safe(buffer_path)

        self._log(f"Loaded checkpoint: {prefix}")
        return result

    def load_best_checkpoint(self, name: str) -> Optional[Dict]:
        """
        Load the best pretraining checkpoint.

        Args:
            name: Base name (e.g., "pretrained").

        Returns:
            Dict with 'network_path', 'state'.
        """
        network_path = self.checkpoint_dir / f"{name}_best_network.pt"
        state_path = self.checkpoint_dir / f"{name}_best_state.pkl"

        if not network_path.exists():
            return None

        result = {
            "network_path": str(network_path),
            "state": None,
        }

        if state_path.exists():
            result["state"] = self._load_pickle_safe(state_path)

        return result

    def list_best_checkpoints(self, name: str) -> List[int]:
        """
        List all numbered best checkpoint counts.

        Args:
            name: Base name (e.g., "pretrained").

        Returns:
            List of best_count numbers, sorted ascending.
        """
        pattern = re.compile(rf"{re.escape(name)}_best_(\d+)_network\.pt")
        counts = []

        for filepath in self.checkpoint_dir.glob(f"{name}_best_*_network.pt"):
            match = pattern.match(filepath.name)
            if match:
                counts.append(int(match.group(1)))

        return sorted(counts)

    def load_best_numbered_checkpoint(
        self, name: str, best_count: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Load a specific numbered best checkpoint.

        Args:
            name: Base name (e.g., "pretrained").
            best_count: Specific checkpoint number, or None for latest.

        Returns:
            Dict with 'network_path', 'state', 'best_count'.
        """
        if best_count is None:
            # Get latest
            counts = self.list_best_checkpoints(name)
            if not counts:
                return None
            best_count = max(counts)

        network_path = self.checkpoint_dir / f"{name}_best_{best_count}_network.pt"
        state_path = self.checkpoint_dir / f"{name}_best_{best_count}_state.pkl"

        if not network_path.exists():
            self._log(f"Checkpoint not found: {name}_best_{best_count}")
            return None

        result = {
            "network_path": str(network_path),
            "state": None,
            "best_count": best_count,
        }

        if state_path.exists():
            result["state"] = self._load_pickle_safe(state_path)

        self._log(f"Loaded checkpoint: {name}_best_{best_count}")
        return result

    # =========================================================================
    # File I/O with integrity checks
    # =========================================================================

    def _save_network(self, network: Any, path: Path) -> None:
        """Save network to file."""
        # Support both .save() method and raw state_dict
        if hasattr(network, "save"):
            network.save(str(path))
        else:
            torch.save(network.state_dict(), path)

    def _save_pickle(self, obj: Any, path: Path) -> None:
        """Save object with pickle."""
        # Write to temp file first, then rename (atomic)
        temp_path = path.with_suffix(".tmp")
        try:
            with open(temp_path, "wb") as f:
                pickle.dump(obj, f)
            temp_path.replace(path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e

    def _load_pickle_safe(self, path: Path) -> Optional[Any]:
        """Load pickle with integrity check."""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as e:
            self._log(f"Warning: Corrupted file {path.name}: {e}")
            return None

    def verify_checkpoint_integrity(self, prefix: str) -> bool:
        """
        Verify a checkpoint is not corrupted.

        Args:
            prefix: Checkpoint prefix (e.g., "iteration_10").

        Returns:
            True if all files are valid.
        """
        network_path = self.checkpoint_dir / f"{prefix}_network.pt"
        state_path = self.checkpoint_dir / f"{prefix}_state.pkl"

        # Check network
        if network_path.exists():
            try:
                torch.load(network_path, map_location="cpu")
            except Exception as e:
                self._log(f"Corrupted network: {network_path.name}: {e}")
                return False

        # Check state
        if state_path.exists():
            if self._load_pickle_safe(state_path) is None:
                return False

        return True

    def find_valid_checkpoint(self) -> Optional[str]:
        """
        Find the most recent valid checkpoint.

        Falls back to earlier checkpoints if latest is corrupted.

        Returns:
            Valid checkpoint prefix or None.
        """
        iterations = sorted(self._list_training_iterations(), reverse=True)

        for it in iterations:
            prefix = f"iteration_{it}"
            if self.verify_checkpoint_integrity(prefix):
                return prefix
            else:
                self._log(f"Skipping corrupted checkpoint: {prefix}")

        return None

    def get_disk_usage(self) -> Dict[str, int]:
        """Get disk usage statistics."""
        total = 0
        by_type = {"network": 0, "state": 0, "buffer": 0}

        for filepath in self.checkpoint_dir.iterdir():
            if filepath.is_file():
                size = filepath.stat().st_size
                total += size

                if "_network.pt" in filepath.name:
                    by_type["network"] += size
                elif "_state.pkl" in filepath.name:
                    by_type["state"] += size
                elif "_buffer.pkl" in filepath.name:
                    by_type["buffer"] += size

        return {
            "total_bytes": total,
            "total_mb": total / (1024 * 1024),
            **{f"{k}_mb": v / (1024 * 1024) for k, v in by_type.items()},
        }

    def print_status(self) -> None:
        """Print checkpoint status summary."""
        checkpoints = self.list_checkpoints()
        usage = self.get_disk_usage()

        print(f"\n[Checkpoint Status: {self.checkpoint_dir}]")
        print(f"  Total checkpoints: {len(checkpoints)}")
        print(f"  Disk usage: {usage['total_mb']:.1f} MB")
        print(f"    - Networks: {usage['network_mb']:.1f} MB")
        print(f"    - States: {usage['state_mb']:.1f} MB")
        print(f"    - Buffers: {usage['buffer_mb']:.1f} MB")

        if checkpoints:
            print(f"\n  Checkpoints:")
            for prefix, num in checkpoints[-10:]:  # Last 10
                is_ms = self._is_milestone(num)
                ms_str = " (milestone)" if is_ms else ""
                print(f"    - {prefix}{ms_str}")

            if len(checkpoints) > 10:
                print(f"    ... and {len(checkpoints) - 10} more")
