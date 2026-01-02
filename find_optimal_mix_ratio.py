#!/usr/bin/env python3
"""
Hyperparameter search for optimal pretrain_mix_ratio.

Tests multiple mix ratios and compares results against pretrained model
using the diagnostic suite.

Usage:
    python find_optimal_mix_ratio.py
    python find_optimal_mix_ratio.py --iterations 10 --ratios 0.1,0.2,0.3
    python find_optimal_mix_ratio.py --pretrained models/pretrained_best_7_network.pt
"""

import argparse
import json
import os
import sys
import shutil
import time
from dataclasses import dataclass
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np


@dataclass
class RatioResult:
    """Results for a single mix ratio experiment."""

    ratio: float
    iterations_trained: int
    final_loss: float
    final_policy_loss: float
    final_value_loss: float
    diagnostic_score: float
    vs_pretrained_score: float
    test_scores: dict
    training_time: float


def run_experiment(
    ratio: float,
    iterations: int,
    pretrained_path: str,
    base_config_path: str,
    output_dir: str,
    pretrained_scores: Optional[dict] = None,  # Cached pretrained scores
    verbose: bool = True,
) -> Optional[RatioResult]:
    """
    Run a single experiment with a specific mix ratio.

    Args:
        ratio: The pretrain_mix_ratio to test
        iterations: Number of training iterations
        pretrained_path: Path to pretrained model for comparison
        base_config_path: Path to base config.json
        output_dir: Directory for this experiment's checkpoints
        verbose: Print progress

    Returns:
        RatioResult with all metrics
    """
    from alphazero import DualHeadNetwork, AlphaZeroTrainer
    from config import Config
    from diagnostics.comparison import run_all_tests_on_network

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: pretrain_mix_ratio = {ratio}")
    print(f"{'='*60}")

    # Load and modify config
    config = Config.load(base_config_path)
    config.training.pretrain_mix_ratio = ratio
    config.training.checkpoint_path = output_dir
    config.training.iterations = iterations

    # Speed optimizations for hyperparameter search
    config.training.games_per_iteration = 30  # Fewer games
    config.training.num_simulations = 100  # Faster MCTS (vs 400 in config)
    config.training.min_buffer_size = 500  # Start training sooner
    config.training.arena_interval = iterations + 1  # Skip arena during search

    # Clear previous experiment data
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load pretrained network as starting point
    if verbose:
        print(f"  Loading pretrained model: {pretrained_path}")

    try:
        network = DualHeadNetwork.load(pretrained_path)
    except Exception as e:
        print(f"  ERROR: Could not load pretrained model: {e}")
        return None

    # Sync history_length
    network_history_length = (network.num_input_planes - 12) // 12 - 1
    config.training.history_length = network_history_length

    # Create trainer
    trainer = AlphaZeroTrainer(network, config.training)

    # Training stats
    training_stats = {
        "final_loss": 0,
        "final_policy": 0,
        "final_value": 0,
    }

    def callback(data: dict):
        phase = data.get("phase", "")
        if phase == "self_play":
            games = data.get("games_played", 0)
            total = data.get("total_games", 0)
            print(f"\r    Self-play: {games}/{total}", end="", flush=True)
        elif phase == "training":
            epoch = data.get("epoch", 0)
            epochs = data.get("epochs", 0)
            loss = data.get("total_loss", 0)
            print(
                f"\r    Training epoch {epoch}/{epochs} | Loss: {loss:.4f}      ",
                end="",
                flush=True,
            )
        elif phase == "iteration_complete":
            stats = data.get("stats", {})
            ts = stats.get("training_stats", {})
            training_stats["final_loss"] = ts.get("avg_total_loss", 0)
            training_stats["final_policy"] = ts.get("avg_policy_loss", 0)
            training_stats["final_value"] = ts.get("avg_value_loss", 0)
            print()

    # Run training
    if verbose:
        print(f"  Training {iterations} iterations with mix_ratio={ratio}...")

    start_time = time.time()

    for i in range(iterations):
        print(f"  Iteration {i+1}/{iterations}")
        trainer.train_iteration(callback)

    training_time = time.time() - start_time

    if verbose:
        print(f"\n  Training completed in {training_time:.1f}s")
        print(f"  Final loss: {training_stats['final_loss']:.4f}")

    # Run diagnostics on trained model
    if verbose:
        print("\n  Running diagnostics on trained model...")

    trained_scores = run_all_tests_on_network(
        trainer.network, silent=True, show_progress=False
    )
    trained_avg = np.mean(list(trained_scores.values()))

    # Use cached pretrained scores or compute if not provided
    if pretrained_scores is None:
        if verbose:
            print("  Running diagnostics on pretrained model...")
        pretrained_network = DualHeadNetwork.load(pretrained_path)
        pretrained_scores = run_all_tests_on_network(
            pretrained_network, silent=True, show_progress=False
        )
    pretrained_avg = np.mean(list(pretrained_scores.values()))

    # Calculate vs_pretrained score (how much we retained)
    vs_pretrained = trained_avg / pretrained_avg if pretrained_avg > 0 else 0

    if verbose:
        print(f"\n  Results for ratio={ratio}:")
        print(f"    Trained avg score:    {trained_avg*100:.1f}%")
        print(f"    Pretrained avg score: {pretrained_avg*100:.1f}%")
        print(f"    Retention:            {vs_pretrained*100:.1f}%")

    return RatioResult(
        ratio=ratio,
        iterations_trained=iterations,
        final_loss=training_stats["final_loss"],
        final_policy_loss=training_stats["final_policy"],
        final_value_loss=training_stats["final_value"],
        diagnostic_score=trained_avg,
        vs_pretrained_score=vs_pretrained,
        test_scores=trained_scores,
        training_time=training_time,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal pretrain_mix_ratio by testing multiple values"
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4",
        help="Comma-separated list of ratios to test (default: 0.0,0.1,0.2,0.3,0.4)",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=10,
        help="Training iterations per experiment (default: 10)",
    )
    parser.add_argument(
        "--pretrained",
        "-p",
        type=str,
        default="models/pretrained_best_7_network.pt",
        help="Path to pretrained model for comparison",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/config.json",
        help="Base config file to use",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="experiments/mix_ratio_search",
        help="Output directory for experiments",
    )

    args = parser.parse_args()

    # Parse ratios
    ratios = [float(r.strip()) for r in args.ratios.split(",")]

    # Validate paths
    if not os.path.exists(args.pretrained):
        print(f"ERROR: Pretrained model not found: {args.pretrained}")
        return 1

    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        return 1

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  PRETRAIN MIX RATIO HYPERPARAMETER SEARCH")
    print("=" * 60)
    print(f"  Ratios to test: {ratios}")
    print(f"  Iterations per experiment: {args.iterations}")
    print(f"  Pretrained model: {args.pretrained}")
    print(f"  Config: {args.config}")
    print(f"  Output: {args.output}")
    print("=" * 60)

    # Pre-compute pretrained diagnostics ONCE (saves ~30s per experiment)
    from alphazero import DualHeadNetwork
    from diagnostics.comparison import run_all_tests_on_network

    print("\n  Pre-computing pretrained model diagnostics...")
    pretrained_network = DualHeadNetwork.load(args.pretrained)
    cached_pretrained_scores = run_all_tests_on_network(
        pretrained_network, silent=True, show_progress=True
    )
    pretrained_avg = np.mean(list(cached_pretrained_scores.values()))
    print(f"  Pretrained baseline score: {pretrained_avg*100:.1f}%")
    del pretrained_network  # Free memory

    # Run experiments
    results: list[RatioResult] = []

    for ratio in ratios:
        exp_dir = os.path.join(args.output, f"ratio_{ratio:.2f}")

        result = run_experiment(
            ratio=ratio,
            iterations=args.iterations,
            pretrained_path=args.pretrained,
            base_config_path=args.config,
            output_dir=exp_dir,
            pretrained_scores=cached_pretrained_scores,
            verbose=True,
        )

        if result:
            results.append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY: PRETRAIN MIX RATIO SEARCH RESULTS")
    print("=" * 70)

    if not results:
        print("  No experiments completed successfully!")
        return 1

    # Sort by diagnostic score (higher is better)
    results.sort(key=lambda r: r.diagnostic_score, reverse=True)

    print(
        f"\n  {'Ratio':<8} {'Diag Score':>12} {'Retention':>12} {'Final Loss':>12} {'Time':>10}"
    )
    print("  " + "-" * 58)

    for r in results:
        marker = " <-- BEST" if r == results[0] else ""
        print(
            f"  {r.ratio:<8.2f} {r.diagnostic_score*100:>11.1f}% "
            f"{r.vs_pretrained_score*100:>11.1f}% "
            f"{r.final_loss:>12.4f} "
            f"{r.training_time:>9.1f}s{marker}"
        )

    print("  " + "-" * 58)

    # Best result
    best = results[0]
    print(f"\n  OPTIMAL RATIO: {best.ratio}")
    print(f"  - Diagnostic score: {best.diagnostic_score*100:.1f}%")
    print(f"  - Retention vs pretrained: {best.vs_pretrained_score*100:.1f}%")

    # Per-test comparison for best vs worst
    if len(results) > 1:
        worst = results[-1]
        print(
            f"\n  Test-by-test comparison (best={best.ratio} vs worst={worst.ratio}):"
        )
        print(f"  {'Test':<22} {'Best':>10} {'Worst':>10} {'Delta':>10}")
        print("  " + "-" * 55)

        for test_name in best.test_scores:
            best_score = best.test_scores.get(test_name, 0)
            worst_score = worst.test_scores.get(test_name, 0)
            delta = best_score - worst_score
            delta_str = f"+{delta*100:.0f}%" if delta > 0 else f"{delta*100:.0f}%"
            print(
                f"  {test_name:<22} {best_score*100:>9.0f}% {worst_score*100:>9.0f}% {delta_str:>10}"
            )

    # Save results to JSON
    results_file = os.path.join(args.output, "results.json")
    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "ratios": ratios,
            "iterations": args.iterations,
            "pretrained": args.pretrained,
        },
        "results": [
            {
                "ratio": r.ratio,
                "diagnostic_score": r.diagnostic_score,
                "vs_pretrained_score": r.vs_pretrained_score,
                "final_loss": r.final_loss,
                "training_time": r.training_time,
                "test_scores": r.test_scores,
            }
            for r in results
        ],
        "optimal_ratio": best.ratio,
    }

    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n  Results saved to: {results_file}")

    # Recommendation
    print("\n" + "=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)
    print(f"  Set pretrain_mix_ratio to {best.ratio} in your config.json:")
    print(f'    "pretrain_mix_ratio": {best.ratio}')
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
