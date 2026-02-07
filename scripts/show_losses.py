#!/usr/bin/env python3
"""Show loss history from saved checkpoints."""

import argparse
import pickle
from pathlib import Path
from typing import Any


def load_state(path: Path) -> dict[str, Any]:
    """Load a state.pkl file."""
    try:
        with open(path, "rb") as f:
            state: dict[str, Any] = pickle.load(f)
            return state
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}


def main() -> None:
    """Show loss history CLI."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Show loss history from checkpoints"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing checkpoints "
        "(default: models for pretrained, checkpoints for iteration)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="pretrained",
        choices=["pretrained", "iteration"],
        help="Checkpoint type (default: pretrained)",
    )
    args: argparse.Namespace = parser.parse_args()

    # Default directory based on checkpoint type
    if args.dir is None:
        args.dir = "checkpoints" if args.type == "iteration" else "models"

    checkpoint_dir: Path = Path(args.dir)
    if not checkpoint_dir.exists():
        print(f"Directory not found: {checkpoint_dir}")
        return

    # Find all state.pkl files (support both regular and "saved_" prefixed files)
    patterns: list[str]
    if args.type == "pretrained":
        patterns = [
            "pretrained_best_*_state.pkl",
            "saved_pretrained_best_*_state.pkl",
        ]
    else:
        patterns = [
            "iteration_*_state.pkl",
            "saved_iteration_*_state.pkl",
        ]

    # Collect files from all patterns
    state_files_set: set[Path] = set()
    for pattern in patterns:
        state_files_set.update(checkpoint_dir.glob(pattern))

    # Sort by checkpoint number
    def get_checkpoint_num(path: Path) -> int:
        """Extract checkpoint number from filename."""
        try:
            parts: list[str] = path.stem.split("_")
            return int(parts[-2])
        except (ValueError, IndexError):
            return 0

    state_files: list[Path] = sorted(state_files_set, key=get_checkpoint_num)

    if not state_files:
        print(f"No state files found matching {patterns} in {checkpoint_dir}")
        return

    # Print header
    print()
    print("=" * 90)
    print("LOSS HISTORY")
    print("=" * 90)
    print()
    print(
        f"  {'#':<4} {'Ep/It':<6} {'Train Loss':<12} {'Val Loss':<12} "
        f"{'Train P':<10} {'Train V':<10} {'Val P':<10} {'Val V':<10}"
    )
    print("  " + "-" * 84)

    # Load and display each checkpoint
    for state_file in state_files:
        state: dict[str, Any] = load_state(state_file)
        if not state:
            continue

        # Extract checkpoint number from filename
        # e.g., "pretrained_best_8_state"
        name: str = state_file.stem
        parts: list[str] = name.split("_")
        checkpoint_num: int | str
        try:
            # Get the number before "_state"
            checkpoint_num = int(parts[-2])
        except (ValueError, IndexError):
            checkpoint_num = "?"

        # epoch for pretrained, iteration for training checkpoints
        epoch: int | str = state.get("epoch", state.get("iteration", "?"))
        train_loss: float | None = state.get("train_loss", None)
        val_loss: float | None = state.get("val_loss", state.get("best_val_loss", None))
        train_policy: float | None = state.get("train_policy", None)
        train_value: float | None = state.get("train_value", None)
        val_policy: float | None = state.get("val_policy", None)
        val_value: float | None = state.get("val_value", None)

        # Format values
        def fmt(v: float | None) -> str:
            """Format a float value or return '-' if None."""
            return f"{v:.5f}" if v is not None else "-"

        print(
            f"  {checkpoint_num:<4} {epoch:<6} "
            f"{fmt(train_loss):<12} {fmt(val_loss):<12} "
            f"{fmt(train_policy):<10} {fmt(train_value):<10} "
            f"{fmt(val_policy):<10} {fmt(val_value):<10}"
        )

    print("  " + "-" * 84)
    print()

    # Show notes
    print("  Note: '-' indicates data not available")
    if args.type == "iteration":
        print("        (AlphaZero training has no validation set)")
    print()


if __name__ == "__main__":
    main()
