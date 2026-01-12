#!/usr/bin/env python3
"""
Migration script to remove deprecated phases and moves_left fields from chunks.

Usage:
    python migrate_chunks.py [chunks_dir]

Default chunks_dir: data/chunks
"""

import os
import sys
import h5py
import tempfile
import shutil
from pathlib import Path


def migrate_chunk(chunk_path: str) -> bool:
    """
    Remove phases and moves_left from a chunk file.

    Returns True if migration was needed, False if already clean.
    """
    with h5py.File(chunk_path, "r") as f:
        has_phases = "phases" in f
        has_moves_left = "moves_left" in f

        if not has_phases and not has_moves_left:
            return False  # Already clean

        # Read required data
        states = f["states"][:]
        policy_indices = f["policy_indices"][:]
        values = f["values"][:]

    # Write to temp file, then replace
    temp_fd, temp_path = tempfile.mkstemp(suffix=".h5")
    os.close(temp_fd)

    try:
        with h5py.File(temp_path, "w") as f:
            f.create_dataset("states", data=states, compression="lzf")
            f.create_dataset("policy_indices", data=policy_indices)
            f.create_dataset("values", data=values)

        # Replace original with migrated
        shutil.move(temp_path, chunk_path)
        return True
    except Exception as e:
        # Cleanup temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def main():
    chunks_dir = sys.argv[1] if len(sys.argv) > 1 else "data/chunks"

    if not os.path.exists(chunks_dir):
        print(f"Error: Directory '{chunks_dir}' not found")
        sys.exit(1)

    # Find all chunk files
    chunk_files = sorted(Path(chunks_dir).glob("chunk_*.h5"))

    if not chunk_files:
        print(f"No chunk files found in {chunks_dir}")
        sys.exit(1)

    print(f"Found {len(chunk_files)} chunks to migrate")

    migrated = 0
    skipped = 0
    errors = 0

    for i, chunk_path in enumerate(chunk_files):
        try:
            if migrate_chunk(str(chunk_path)):
                migrated += 1
            else:
                skipped += 1

            print(
                f"  Progress: {i + 1}/{len(chunk_files)} ({migrated} migrated, {skipped} skipped)"
            )

        except Exception as e:
            print(f"  Error migrating {chunk_path}: {e}")
            errors += 1

    print(f"\nMigration complete:")
    print(f"  Migrated: {migrated}")
    print(f"  Already clean: {skipped}")
    print(f"  Errors: {errors}")

    if errors == 0:
        print("\nYou can now delete this script: migrate_chunks.py")


if __name__ == "__main__":
    main()
