#!/usr/bin/env python3
"""
Main entry point for the Neural Network Diagnostic Suite.
"""

import sys
import os
import time
import argparse

# Add src to path for imports
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root_dir, "src"))
sys.path.insert(0, _root_dir)

from .core import (
    Colors,
    TestResults,
    header,
    load_network_from_path,
    load_latest_network,
    analyze_network_architecture,
)
from .tests import ALL_TESTS
from .comparison import run_comparison, compare_two_models


def run_all_tests(network, iteration: int) -> TestResults:
    """Run all diagnostic tests."""
    results = TestResults()

    print(
        f"\n{Colors.BOLD}Running comprehensive diagnostics for iteration {iteration}{Colors.ENDC}"
    )

    # Architecture analysis first
    analyze_network_architecture(network, results)

    # Run all tests
    for test_func, test_name, _ in ALL_TESTS:
        test_func(network, results)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive network diagnostic with detailed LLM-friendly output."
    )
    parser.add_argument(
        "checkpoint_dir",
        nargs="?",
        default="checkpoints",
        help="Directory containing checkpoint files (default: checkpoints)",
    )
    parser.add_argument(
        "--iteration",
        "-i",
        type=int,
        default=None,
        help="Specific iteration to test (default: latest)",
    )
    parser.add_argument(
        "--compare", "-c", action="store_true", help="Compare multiple checkpoints"
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs=2,
        metavar=("MODEL1", "MODEL2"),
        help="Compare two specific model files (e.g., --models model1.pt model2.pt)",
    )
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Quick comparison (fewer tests)"
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["train", "pretrain"],
        default="train",
        help="Checkpoint type: 'train' for iteration_X, 'pretrain' for pretrained_best_X (default: train)",
    )
    args = parser.parse_args()

    # Handle colors
    if not sys.stdout.isatty():
        Colors.disable()

    print(header("NEURAL NETWORK DIAGNOSTIC SUITE"))
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Two-model comparison mode
    if args.models:
        model1, model2 = args.models
        if not os.path.exists(model1):
            print(f"{Colors.RED}[ERROR] Model not found: {model1}{Colors.ENDC}")
            return
        if not os.path.exists(model2):
            print(f"{Colors.RED}[ERROR] Model not found: {model2}{Colors.ENDC}")
            return
        compare_two_models(model1, model2, quick=args.quick)
        return

    print(f"  Checkpoint directory: {args.checkpoint_dir}")

    # Comparison mode
    if args.compare:
        run_comparison(args.checkpoint_dir, args.type)
        return

    # Load network
    try:
        if args.iteration:
            # Build path based on checkpoint type
            if args.type == "pretrain":
                path = os.path.join(
                    args.checkpoint_dir, f"pretrained_best_{args.iteration}_network.pt"
                )
            else:
                path = os.path.join(
                    args.checkpoint_dir, f"iteration_{args.iteration}_network.pt"
                )
            network = load_network_from_path(path)
            if network is None:
                return
            iteration = args.iteration
        else:
            result = load_latest_network(args.checkpoint_dir, args.type)
            if result is None:
                return
            network, iteration = result
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(
                f"\n{Colors.RED}[ERROR] Checkpoint has incompatible architecture{Colors.ENDC}"
            )
            print(
                f"  The checkpoint was trained with a different network configuration."
            )
            print(f"  This can happen when the value head architecture was changed.")
            print(f"\n  Details: {str(e)[:200]}...")
            return
        raise

    # Run tests
    results = run_all_tests(network, iteration)

    # Print summary
    results.print_summary()

    print(f"\n{Colors.DIM}Run with --compare to compare across iterations{Colors.ENDC}")


if __name__ == "__main__":
    main()
