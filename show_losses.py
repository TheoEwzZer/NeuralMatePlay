#!/usr/bin/env python3
"""Show loss history from saved checkpoints."""

import argparse
import pickle
from pathlib import Path


def load_state(path: Path) -> dict:
    """Load a state.pkl file."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Show loss history from checkpoints")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing checkpoints (default: models for pretrained, checkpoints for iteration)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="pretrained",
        choices=["pretrained", "iteration"],
        help="Checkpoint type (default: pretrained)",
    )
    args = parser.parse_args()

    # Default directory based on checkpoint type
    if args.dir is None:
        args.dir = "checkpoints" if args.type == "iteration" else "models"

    checkpoint_dir = Path(args.dir)
    if not checkpoint_dir.exists():
        print(f"Directory not found: {checkpoint_dir}")
        return

    # Find all state.pkl files (support both regular and "saved_" prefixed files)
    if args.type == "pretrained":
        patterns = ["pretrained_best_*_state.pkl", "saved_pretrained_best_*_state.pkl"]
    else:
        patterns = ["iteration_*_state.pkl", "saved_iteration_*_state.pkl"]

    # Collect files from all patterns
    state_files_set = set()
    for pattern in patterns:
        state_files_set.update(checkpoint_dir.glob(pattern))

    # Sort by checkpoint number
    def get_checkpoint_num(path):
        try:
            parts = path.stem.split("_")
            return int(parts[-2])
        except (ValueError, IndexError):
            return 0

    state_files = sorted(state_files_set, key=get_checkpoint_num)

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
        state = load_state(state_file)
        if not state:
            continue

        # Extract checkpoint number from filename
        name = state_file.stem  # e.g., "pretrained_best_8_state"
        parts = name.split("_")
        try:
            checkpoint_num = int(parts[-2])  # Get the number before "_state"
        except (ValueError, IndexError):
            checkpoint_num = "?"

        # epoch for pretrained, iteration for training checkpoints
        epoch = state.get("epoch", state.get("iteration", "?"))
        train_loss = state.get("train_loss", None)
        val_loss = state.get("val_loss", state.get("best_val_loss", None))
        train_policy = state.get("train_policy", None)
        train_value = state.get("train_value", None)
        val_policy = state.get("val_policy", None)
        val_value = state.get("val_value", None)

        # Format values
        def fmt(v):
            return f"{v:.5f}" if v is not None else "-"

        print(
            f"  {checkpoint_num:<4} {epoch:<6} {fmt(train_loss):<12} {fmt(val_loss):<12} "
            f"{fmt(train_policy):<10} {fmt(train_value):<10} {fmt(val_policy):<10} {fmt(val_value):<10}"
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
