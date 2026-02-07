"""
Initialize a new neural network with random weights.

Creates a DualHeadNetwork with the specified architecture and saves it to a file.
Useful for testing, benchmarking, or starting fresh training.
"""

import argparse
import os
import sys
from typing import Literal

# Ensure proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.alphazero.network import DualHeadNetwork
from src.alphazero.spatial_encoding import get_num_planes
from src.config import Config


def main() -> Literal[1] | Literal[0]:
    parser = argparse.ArgumentParser(
        description="Initialize a new neural network with random weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default network (72 planes, 128 filters, 8 blocks)
  ./neural_mate_init -o models/random.pt

  # Create with custom architecture
  ./neural_mate_init -o models/small.pt --filters 64 --blocks 6

  # Use config file for architecture
  ./neural_mate_init -o models/network.pt --config config/config.json
""",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="models/network.pt",
        help="Output path for the network file (default: models/network.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json for network architecture (optional)",
    )
    parser.add_argument(
        "--filters",
        type=int,
        default=None,
        help="Number of filters in residual blocks (default: 128)",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=None,
        help="Number of residual blocks (default: 8)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing file without asking",
    )

    args: argparse.Namespace = parser.parse_args()

    # Load config if specified
    if args.config and os.path.exists(args.config):
        print(f"Loading config from {args.config}...")
        config: Config = Config.load(args.config)
        num_filters: int = config.network.num_filters
        num_blocks: int = config.network.num_residual_blocks
    else:
        # Use defaults
        num_filters = 128
        num_blocks = 8

    # Override with CLI arguments if provided
    if args.filters is not None:
        num_filters = args.filters
    if args.blocks is not None:
        num_blocks = args.blocks

    # Fixed 72 input planes (48 history + 12 metadata + 8 semantic + 4 tactical)
    num_input_planes: int = get_num_planes()

    # Check if output file exists
    if os.path.exists(args.output) and not args.force:
        response: str = input(f"File {args.output} already exists. Overwrite? [y/N] ")
        if response.lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    print()
    print("=" * 60)
    print("  INITIALIZING NEW NEURAL NETWORK")
    print("=" * 60)
    print()
    print("Architecture:")
    print(
        f"  Input planes:       {num_input_planes} (48 history + 24 metadata/semantic/tactical)"
    )
    print(f"  Filters:            {num_filters}")
    print(f"  Residual blocks:    {num_blocks}")
    print("  Heads:              Policy, WDL, Phase, MovesLeft")
    print()

    # Create the network
    print("Creating network with random weights...")
    network = DualHeadNetwork(
        num_input_planes=num_input_planes,
        num_filters=num_filters,
        num_residual_blocks=num_blocks,
    )

    # Count parameters
    total_params: int = sum(p.numel() for p in network.parameters())
    trainable_params: int = sum(
        p.numel() for p in network.parameters() if p.requires_grad
    )

    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    print(f"  Model size (approx):   {total_params * 4 / 1024 / 1024:.1f} MB")
    print()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the network
    print(f"Saving to {args.output}...")
    network.save(args.output)

    # Verify file was created
    if os.path.exists(args.output):
        file_size: float = os.path.getsize(args.output) / 1024 / 1024
        print(f"  File size: {file_size:.1f} MB")
        print()
        print("Done! Network initialized with random weights.")
        print()
        print("Next steps:")
        print("  - Pretrain:  ./neural_mate_pretrain")
        print(f"  - Train:     ./neural_mate_train --network {args.output}")
        print(f"  - Play:      ./neural_mate_play --network {args.output}")
        print(f"  - Diagnose:  ./neural_mate_diagnose --network {args.output}")
    else:
        print("ERROR: Failed to save network!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
