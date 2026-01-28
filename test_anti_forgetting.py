#!/usr/bin/env python3
"""
Test script to compare anti-forgetting mechanisms.

Tests 4 configurations:
1. NONE: No anti-forgetting (baseline)
2. EWC: EWC only
3. REPLAY: Replay Buffer only
4. BOTH: EWC + Replay Buffer

Usage:
    python test_anti_forgetting.py                    # Run all tests
    python test_anti_forgetting.py --only NONE EWC   # Run specific tests
    python test_anti_forgetting.py --chunks 100      # Use 100 chunks
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# The 4 test configurations
CONFIGS = {
    "NONE": {
        "ewc_enabled": False,
        "tactical_replay_enabled": False,
        "description": "No anti-forgetting (baseline)",
    },
    "EWC": {
        "ewc_enabled": True,
        "tactical_replay_enabled": False,
        "ewc_lambda": 0.02,  # Reduced from 0.4 to allow convergence
        "description": "EWC only",
    },
    "REPLAY": {
        "ewc_enabled": False,
        "tactical_replay_enabled": True,
        "description": "Replay Buffer only",
    },
    "BOTH": {
        "ewc_enabled": True,
        "tactical_replay_enabled": True,
        "ewc_lambda": 0.02,  # Reduced from 0.4 to allow convergence
        "description": "EWC + Replay Buffer",
    },
}


def load_config(config_path: str) -> dict:
    """Load JSON config file."""
    with open(config_path, "r") as f:
        return json.load(f)


def save_config(config: dict, config_path: str) -> None:
    """Save JSON config file."""
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def run_training(
    config_name: str,
    base_config_path: str,
    output_dir: str,
    chunks: int,
    epochs: int,
) -> tuple[bool, str]:
    """Run training with specified config. Returns (success, model_path)."""
    print(f"\n{'='*60}")
    print(f"Training: {config_name} - {CONFIGS[config_name]['description']}")
    print(f"Chunks: {chunks}, Epochs: {epochs}")
    print(f"{'='*60}\n")

    config_settings = CONFIGS[config_name]

    # Load and modify config
    config = load_config(base_config_path)
    if "pretraining" not in config:
        config["pretraining"] = {}

    # Keep original PGN paths to reuse existing chunks!
    # Only modify the anti-forgetting settings
    config["pretraining"]["ewc_enabled"] = config_settings["ewc_enabled"]
    config["pretraining"]["tactical_replay_enabled"] = config_settings[
        "tactical_replay_enabled"
    ]

    # Propagate EWC parameters if specified
    if "ewc_lambda" in config_settings:
        config["pretraining"]["ewc_lambda"] = config_settings["ewc_lambda"]

    # Reduce buffer capacity and ratio for quick tests (fewer chunks = smaller buffer)
    config["pretraining"]["tactical_replay_capacity"] = chunks * 50
    config["pretraining"]["tactical_replay_ratio"] = 0.10

    # Make sure we use existing chunks directory
    if "chunks_dir" not in config["pretraining"]:
        config["pretraining"]["chunks_dir"] = "data/chunks"

    # Save temp config
    os.makedirs(output_dir, exist_ok=True)

    # Clean up any existing state to prevent cross-test contamination
    for state_file in ["ewc_state.pt", "replay_buffer.npz"]:
        state_path = os.path.join(output_dir, state_file)
        if os.path.exists(state_path):
            os.remove(state_path)
            print(f"  Cleaned up: {state_path}")

    temp_config_path = os.path.join(output_dir, f"config_{config_name.lower()}.json")
    save_config(config, temp_config_path)

    # Output model path
    model_output = os.path.join(output_dir, f"model_{config_name.lower()}.pt")

    # Find all PGN files (same as original training) to reuse existing chunks
    pgn_files = sorted(glob.glob("data/lichess_*.pgn"))
    if not pgn_files:
        pgn_files = ["data/lichess_elite_2020-08.pgn"]  # Fallback

    # Build command
    cmd = (
        [
            sys.executable,
            "-m",
            "src.pretraining.pretrain",
            "--config",
            temp_config_path,
            "--pgn",
        ]
        + pgn_files
        + [  # Pass all PGN files to reuse existing chunks
            "--epochs",
            str(epochs),
            "--max-chunks",
            str(chunks),
            "--output",
            model_output,
        ]
    )

    print(f"Running: {' '.join(cmd)}\n")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ {config_name} completed in {elapsed:.1f}s")

        # Find the best model
        best_pattern = f"model_{config_name.lower()}_best_*_network.pt"
        best_models = list(Path(output_dir).glob(best_pattern))
        if best_models:
            model_path = str(sorted(best_models)[-1])
        else:
            # Try to find any matching model
            pattern = f"model_{config_name.lower()}*network.pt"
            models = list(Path(output_dir).glob(pattern))
            model_path = str(sorted(models)[-1]) if models else model_output

        return True, model_path

    except subprocess.CalledProcessError as e:
        print(f"\n✗ {config_name} FAILED: {e}")
        return False, ""


def run_diagnostic(model1_path: str, model2_path: str) -> str:
    """Run diagnostic comparison between two models."""
    cmd = [
        sys.executable,
        "diagnose_network.py",
        "-m",
        model1_path,
        model2_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error: {e}"


def extract_summary(diagnostic_output: str) -> dict:
    """Extract key metrics from diagnostic output."""
    summary = {"weighted_avg": None, "weighted_wins": None}

    for line in diagnostic_output.split("\n"):
        if "WEIGHTED AVG" in line:
            parts = line.split()
            # Find percentages
            for i, p in enumerate(parts):
                if "%" in p:
                    if summary["weighted_avg"] is None:
                        summary["weighted_avg"] = p
                    else:
                        summary["weighted_avg_2"] = p
        elif "WEIGHTED WINS" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p.isdigit():
                    if summary["weighted_wins"] is None:
                        summary["weighted_wins"] = int(p)
                    else:
                        summary["weighted_wins_2"] = int(p)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Test anti-forgetting mechanisms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_anti_forgetting.py                     # Full test (400 chunks, 5 epochs)
  python test_anti_forgetting.py --chunks 50        # Quick test (50 chunks)
  python test_anti_forgetting.py --only NONE EWC    # Compare only NONE vs EWC
  python test_anti_forgetting.py --skip-training    # Only run diagnostics
        """,
    )
    parser.add_argument(
        "--config", default="config/config.json", help="Base config file path"
    )
    parser.add_argument(
        "--chunks", type=int, default=400, help="Chunks per epoch (default: 400)"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs (default: 5)"
    )
    parser.add_argument(
        "--output-dir",
        default="models/anti_forgetting_test",
        help="Output directory for test models",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only run diagnostics on existing models",
    )
    parser.add_argument(
        "--only",
        choices=list(CONFIGS.keys()),
        nargs="+",
        help="Only run specific configs (e.g., --only NONE EWC)",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configs to test
    configs_to_test = args.only if args.only else list(CONFIGS.keys())

    print(f"\n{'#'*60}")
    print(f"# Anti-Forgetting Mechanism Comparison Test")
    print(f"{'#'*60}")
    print(f"\nConfigurations to test:")
    for name in configs_to_test:
        print(f"  - {name}: {CONFIGS[name]['description']}")
    print(f"\nSettings:")
    print(f"  Chunks per epoch: {args.chunks}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Output directory: {args.output_dir}")

    trained_models = {}

    if not args.skip_training:
        # Run training for each config
        for config_name in configs_to_test:
            success, model_path = run_training(
                config_name,
                args.config,
                args.output_dir,
                args.chunks,
                args.epochs,
            )

            if success and model_path:
                trained_models[config_name] = model_path
                print(f"  Model saved: {model_path}")
            else:
                print(f"  ✗ {config_name} training failed")
    else:
        # Find existing models
        print("\nLooking for existing models...")
        for config_name in configs_to_test:
            pattern = f"model_{config_name.lower()}*network.pt"
            models = list(Path(args.output_dir).glob(pattern))
            if models:
                trained_models[config_name] = str(sorted(models)[-1])
                print(f"  Found: {config_name} -> {trained_models[config_name]}")
            else:
                print(f"  Not found: {config_name}")

    # Run comparisons
    print(f"\n{'#'*60}")
    print(f"# Diagnostic Comparisons")
    print(f"{'#'*60}")

    if len(trained_models) < 2:
        print("\nNot enough models to compare. Need at least 2.")
        return 1

    # Results table
    results = {}

    # Compare each config against baseline (NONE)
    baseline = "NONE" if "NONE" in trained_models else list(trained_models.keys())[0]
    print(f"\nBaseline: {baseline}")

    for config_name in trained_models:
        if config_name == baseline:
            continue

        print(f"\n--- {config_name} vs {baseline} ---")
        output = run_diagnostic(trained_models[config_name], trained_models[baseline])

        # Print relevant lines
        for line in output.split("\n"):
            if any(
                x in line
                for x in [
                    "WEIGHTED AVG",
                    "WEIGHTED WINS",
                    "Mate in 1",
                    "Tactics",
                    "Material",
                ]
            ):
                print(line)

        results[config_name] = extract_summary(output)

    # Final summary
    print(f"\n{'#'*60}")
    print(f"# SUMMARY")
    print(f"{'#'*60}")

    print(f"\n{'Config':<10} {'Description':<30} {'Model Path'}")
    print("-" * 80)
    for name, path in trained_models.items():
        desc = CONFIGS[name]["description"]
        print(f"{name:<10} {desc:<30} {os.path.basename(path)}")

    if results:
        print(f"\n{'Config':<10} vs {baseline:<8} {'Weighted Avg':<15} {'Wins'}")
        print("-" * 50)
        for config_name, summary in results.items():
            avg = summary.get("weighted_avg", "N/A")
            wins = summary.get("weighted_wins", "N/A")
            print(f"{config_name:<10} vs {baseline:<8} {str(avg):<15} {wins}")

    print("\n" + "=" * 60)
    print("To run full comparison manually:")
    if len(trained_models) >= 2:
        models_list = list(trained_models.values())
        print(f"  python diagnose_network.py -m {models_list[0]} {models_list[1]}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
