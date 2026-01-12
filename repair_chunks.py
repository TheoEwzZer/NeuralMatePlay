#!/usr/bin/env python3
"""
Repair metadata.json for chunk recovery after interrupted preprocessing.

Usage:
    python repair_chunks.py                    # Auto-detect and repair
    python repair_chunks.py --estimate-games   # Estimate games from positions
    python repair_chunks.py --set-file "data/lichess_elite_2020-08.pgn" --games 10000
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pretraining.chunk_manager import ChunkManager


def count_positions(chunks_dir: str) -> int:
    """Count total positions in existing chunks."""
    total = 0
    for chunk_path in ChunkManager.iter_chunk_paths(chunks_dir):
        try:
            chunk = ChunkManager.load_chunk(chunk_path)
            total += len(chunk["states"])
        except Exception as e:
            print(f"Error loading {chunk_path}: {e}")
    return total


def estimate_games(positions: int, positions_per_game: int = 77) -> int:
    """Estimate number of games from positions (avg ~77 positions/game for master games)."""
    return positions // positions_per_game


def main():
    parser = argparse.ArgumentParser(description="Repair chunk metadata for resume")
    parser.add_argument(
        "--chunks-dir", default="data/chunks", help="Chunks directory"
    )
    parser.add_argument(
        "--estimate-games", action="store_true",
        help="Estimate games from positions (77 pos/game average)"
    )
    parser.add_argument(
        "--set-file", type=str,
        help="Set the current file being processed (for resume)"
    )
    parser.add_argument(
        "--games", type=int, default=0,
        help="Number of games already processed in current file"
    )
    parser.add_argument(
        "--processed-files", nargs="*", default=[],
        help="List of fully processed PGN files"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be written without saving"
    )

    args = parser.parse_args()

    # Check if chunks exist
    chunk_paths = list(ChunkManager.iter_chunk_paths(args.chunks_dir))
    if not chunk_paths:
        print(f"No chunks found in {args.chunks_dir}")
        return 1

    print(f"Found {len(chunk_paths)} chunks")

    # Count positions
    print("Counting positions (this may take a while)...")
    total_positions = count_positions(args.chunks_dir)
    print(f"Total positions: {total_positions:,}")

    # Estimate games if requested
    estimated_games = estimate_games(total_positions)
    print(f"Estimated games: ~{estimated_games:,} (assuming 77 pos/game)")

    # Build metadata
    metadata = {
        "num_chunks": len(chunk_paths),
        "total_examples": total_positions,
        "chunk_size": 20000,  # Default
        "games_processed": args.games if args.set_file else estimated_games,
        "processed_files": args.processed_files,
        "current_file": args.set_file,
        "current_file_games": args.games if args.set_file else 0,
    }

    print("\nMetadata to write:")
    print(json.dumps(metadata, indent=2))

    if args.dry_run:
        print("\n(Dry run - not saving)")
        return 0

    # Confirm
    print("\nWrite this metadata? [y/N] ", end="")
    response = input().strip().lower()
    if response != "y":
        print("Aborted")
        return 1

    # Save
    metadata_path = os.path.join(args.chunks_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved to {metadata_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
