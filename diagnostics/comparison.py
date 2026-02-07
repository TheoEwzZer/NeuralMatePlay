"""Multi-checkpoint comparison utilities."""

from __future__ import annotations

import sys
import os
import io
import contextlib
import time
from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np

if TYPE_CHECKING:
    from src.alphazero.network import DualHeadNetwork

# Ensure proper imports
_root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from src.alphazero import DualHeadNetwork as _DualHeadNetwork
from .core import Colors, TestResults, header, get_available_checkpoints
from .tests import ALL_TESTS


def run_all_tests_on_network(
    network: DualHeadNetwork, show_progress: bool = False
) -> dict[str, float]:
    """
    Run all tests on a single network and return scores.
    Network is loaded once, all tests run on it.

    Args:
        network: The neural network to test
        silent: If True, suppress all test output
        show_progress: If True, show test names and scores as they run

    Returns:
        dict mapping test_name -> score
    """
    scores: dict[str, float] = {}

    test_func: Callable[..., float]
    test_name: str
    i: int
    for i, (test_func, test_name, _) in enumerate(ALL_TESTS):
        if show_progress:
            print(
                f"      [{i+1:>2}/{len(ALL_TESTS)}] {test_name:<22}",
                end=" ",
                flush=True,
            )

        dummy_results: TestResults = TestResults()

        f: io.StringIO = io.StringIO()
        with contextlib.redirect_stdout(f):
            score: float = test_func(network, dummy_results)

        scores[test_name] = score

        if show_progress:
            color: str
            if score >= 0.7:
                color = Colors.GREEN
            elif score >= 0.4:
                color = Colors.YELLOW
            else:
                color = Colors.RED
            print(f"{color}{score*100:>5.0f}%{Colors.ENDC}")

    return scores


def compare_two_models(model1_path: str, model2_path: str) -> None:
    """
    Compare two specific model files on all tests.
    OPTIMIZED: Each model is loaded once, then all tests run on it.
    """
    print(header("COMPARING TWO MODELS"))

    # Get model names for display
    name1: str = os.path.basename(model1_path)
    name2: str = os.path.basename(model2_path)

    print(f"  Running {len(ALL_TESTS)} tests on each...")

    # Load and test model 1
    print(f"\n  {Colors.BOLD}Model 1: {name1}{Colors.ENDC}")
    try:
        network1: DualHeadNetwork = _DualHeadNetwork.load(model1_path)
    except Exception as e:
        print(f"    {Colors.RED}[ERROR] Cannot load: {e}{Colors.ENDC}")
        return

    start: float = time.time()
    scores1: dict[str, float] = run_all_tests_on_network(network1, show_progress=True)
    elapsed1: float = time.time() - start
    avg1: float = float(np.mean(list(scores1.values())))
    if avg1 >= 0.7:
        color1: str = Colors.GREEN
    elif avg1 >= 0.4:
        color1 = Colors.YELLOW
    else:
        color1 = Colors.RED
    print(f"      {'─'*35}")
    print(f"      Average: {color1}{avg1*100:.0f}%{Colors.ENDC} ({elapsed1:.1f}s)")
    # Free memory
    del network1

    # Load and test model 2
    print(f"\n  {Colors.BOLD}Model 2: {name2}{Colors.ENDC}")
    try:
        network2: DualHeadNetwork = _DualHeadNetwork.load(model2_path)
    except Exception as e:
        print(f"    {Colors.RED}[ERROR] Cannot load: {e}{Colors.ENDC}")
        return

    start = time.time()
    scores2: dict[str, float] = run_all_tests_on_network(network2, show_progress=True)
    elapsed2: float = time.time() - start
    avg2: float = float(np.mean(list(scores2.values())))
    if avg2 >= 0.7:
        color2: str = Colors.GREEN
    elif avg2 >= 0.4:
        color2 = Colors.YELLOW
    else:
        color2 = Colors.RED
    print(f"      {'─'*35}")
    print(f"      Average: {color2}{avg2*100:.0f}%{Colors.ENDC} ({elapsed2:.1f}s)")
    # Free memory
    del network2

    # Build results structure
    results: dict[str, list[tuple[str, float]]] = {}
    weights: dict[str, float] = {}
    test_name: str
    weight: float
    for _, test_name, weight in ALL_TESTS:
        weights[test_name] = weight
        results[test_name] = [
            (name1, scores1.get(test_name, 0)),
            (name2, scores2.get(test_name, 0)),
        ]

    # Summary table
    print(header("COMPARISON SUMMARY"))

    # Truncate names for header
    h1: str = name1[:15] + ".." if len(name1) > 17 else name1
    h2: str = name2[:15] + ".." if len(name2) > 17 else name2

    print(f"\n  {'Test':<18} {'Wt':>4} {h1:>15} {h2:>15}  {'Winner':>10}")
    print("  " + "-" * 75)

    model1_wins: float = 0
    model2_wins: float = 0
    model1_weighted: float = 0
    model2_weighted: float = 0
    total_weight: float = 0

    scores_list: list[tuple[str, float]]
    for test_name, scores_list in results.items():
        score1: float = scores_list[0][1]
        score2: float = scores_list[1][1]
        weight = weights.get(test_name, 1.0)

        model1_weighted += score1 * weight
        model2_weighted += score2 * weight
        total_weight += weight

        # Colors for scores
        if score1 >= 0.7:
            c1: str = Colors.GREEN
        elif score1 >= 0.4:
            c1 = Colors.YELLOW
        else:
            c1 = Colors.RED

        if score2 >= 0.7:
            c2: str = Colors.GREEN
        elif score2 >= 0.4:
            c2 = Colors.YELLOW
        else:
            c2 = Colors.RED

        # Weight indicator
        wt_str: str = f"x{weight:.0f}" if weight > 1 else ""

        # Determine winner (weighted wins count more)
        diff: float = score2 - score1
        winner: str
        if diff > 0.05:
            winner = f"{Colors.CYAN}-> M2{Colors.ENDC}"
            # Weighted wins
            model2_wins += weight
        elif diff < -0.05:
            winner = f"{Colors.CYAN}<- M1{Colors.ENDC}"
            # Weighted wins
            model1_wins += weight
        else:
            winner = f"{Colors.DIM}tie{Colors.ENDC}"

        # Highlight high-weight tests (pad BEFORE adding ANSI codes)
        name_padded: str = f"{test_name:<18}"
        name_display: str
        if weight > 1:
            name_display = f"{Colors.BOLD}{name_padded}{Colors.ENDC}"
        else:
            name_display = name_padded

        s1: str = f"{c1}{score1*100:>13.0f}%{Colors.ENDC}"
        s2: str = f"{c2}{score2*100:>13.0f}%{Colors.ENDC}"
        print(f"  {name_display} {wt_str:>4} {s1} {s2}  {winner:>10}")

    print("  " + "-" * 75)

    # Weighted averages
    wavg1: float = model1_weighted / total_weight if total_weight > 0 else 0
    wavg2: float = model2_weighted / total_weight if total_weight > 0 else 0

    if wavg1 >= 0.7:
        c1 = Colors.GREEN
    elif wavg1 >= 0.4:
        c1 = Colors.YELLOW
    else:
        c1 = Colors.RED

    if wavg2 >= 0.7:
        c2 = Colors.GREEN
    elif wavg2 >= 0.4:
        c2 = Colors.YELLOW
    else:
        c2 = Colors.RED

    print(
        f"  {'WEIGHTED AVG':<26} {'':>4} {c1}{wavg1*100:>13.0f}%{Colors.ENDC} {c2}{wavg2*100:>13.0f}%{Colors.ENDC}"
    )
    print(f"  {'WEIGHTED WINS':<26} {'':>4} {model1_wins:>14.0f} {model2_wins:>14.0f}")

    # Overall verdict based on weighted scores
    print()
    if model1_wins == model2_wins:
        if wavg1 > wavg2:
            print(
                f"  {Colors.BOLD}{Colors.YELLOW}Tie on wins, but {name1} has higher weighted average{Colors.ENDC}"
            )
        elif wavg2 > wavg1:
            print(
                f"  {Colors.BOLD}{Colors.YELLOW}Tie on wins, but {name2} has higher weighted average{Colors.ENDC}"
            )
        else:
            print(
                f"  {Colors.BOLD}{Colors.YELLOW}Models are evenly matched{Colors.ENDC}"
            )


def run_comparison(checkpoint_dir: str, checkpoint_type: str = "train") -> None:
    """
    Run comparison across all checkpoints with ALL tests.
    OPTIMIZED: Each checkpoint is loaded only once, then all tests run on it.
    """
    print(header("MULTI-CHECKPOINT COMPARISON"))

    checkpoints: list[tuple[int, str]] = get_available_checkpoints(
        checkpoint_dir, checkpoint_type
    )

    if not checkpoints:
        type_name: str = "pretrain" if checkpoint_type == "pretrain" else "train"
        print(
            f"  {Colors.RED}No {type_name} checkpoints found in {checkpoint_dir}!{Colors.ENDC}"
        )
        return

    print(f"  Found {len(checkpoints)} checkpoint(s)")
    print(f"  Running {len(ALL_TESTS)} tests on each...\n")

    # Structure: {iteration: {test_name: score}}
    all_results: dict[int, dict[str, float]] = {}
    sorted_iters: list[int] = []

    total_start: float = time.time()

    iteration: int
    path: str
    for iteration, path in checkpoints:
        print(f"\n  {Colors.BOLD}Checkpoint {iteration}{Colors.ENDC}")

        try:
            network: DualHeadNetwork = _DualHeadNetwork.load(path)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(
                    f"    {Colors.YELLOW}Skipped (incompatible architecture){Colors.ENDC}"
                )
                continue
            raise

        # Run all tests on this network (loaded once!) with progress
        start: float = time.time()
        scores: dict[str, float] = run_all_tests_on_network(network, show_progress=True)
        elapsed: float = time.time() - start

        all_results[iteration] = scores
        sorted_iters.append(iteration)

        # Show summary for this checkpoint
        avg_score: float = float(np.mean(list(scores.values())))
        color: str
        if avg_score >= 0.7:
            color = Colors.GREEN
        elif avg_score >= 0.4:
            color = Colors.YELLOW
        else:
            color = Colors.RED
        print(f"      {'-'*35}")
        print(
            f"      Average: {color}{avg_score*100:.0f}%{Colors.ENDC} ({elapsed:.1f}s)"
        )

        # Free memory
        del network

    total_elapsed: float = time.time() - total_start

    if not all_results:
        print(f"\n  {Colors.RED}No compatible checkpoints found!{Colors.ENDC}")
        return

    print(f"\n  Total time: {total_elapsed:.1f}s")

    # Get test names in order
    test_names: list[str] = [name for _, name, _ in ALL_TESTS]
    test_weights: dict[str, float] = {name: weight for _, name, weight in ALL_TESTS}

    # Print summary table
    print(header("COMPARISON SUMMARY"))

    # Header
    iter_headers: str = "".join(f"{it:>8}" for it in sorted_iters)
    print(f"\n  {'Test':<22} {iter_headers}   {'Trend':>6}")
    print("  " + "-" * (24 + len(sorted_iters) * 8 + 8))

    # Rows - one per test
    test_name: str
    for test_name in test_names:
        weight: float = test_weights.get(test_name, 1.0)

        # Highlight high-weight tests
        name_display: str
        if weight > 1:
            name_display = f"{Colors.BOLD}{test_name:<22}{Colors.ENDC}"
        else:
            name_display = f"{test_name:<22}"

        row: str = f"  {name_display}"

        scores_for_trend: list[float] = []
        it: int
        for it in sorted_iters:
            if it in all_results and test_name in all_results[it]:
                sc: float = all_results[it][test_name]
                scores_for_trend.append(sc)
                if sc >= 0.7:
                    color = Colors.GREEN
                elif sc >= 0.4:
                    color = Colors.YELLOW
                else:
                    color = Colors.RED
                row += f" {color}{sc*100:>6.0f}%{Colors.ENDC}"
            else:
                row += f" {Colors.DIM}     -{Colors.ENDC}"

        # Trend (5% threshold for progressive scoring sensitivity)
        trend: str
        if len(scores_for_trend) >= 2:
            diff: float = scores_for_trend[-1] - scores_for_trend[0]
            if diff > 0.05:
                trend = f"{Colors.GREEN}+{Colors.ENDC}"
            elif diff < -0.05:
                trend = f"{Colors.RED}-{Colors.ENDC}"
            else:
                trend = f"{Colors.YELLOW}={Colors.ENDC}"
        else:
            trend = "-"
        row += f"   {trend:>6}"
        print(row)

    print("  " + "-" * (24 + len(sorted_iters) * 8 + 8))

    # Overall average per iteration
    print(f"\n  {'AVERAGE':<22}", end="")
    for it in sorted_iters:
        if it in all_results:
            avg: float = float(np.mean(list(all_results[it].values())))
            if avg >= 0.7:
                color = Colors.GREEN
            elif avg >= 0.4:
                color = Colors.YELLOW
            else:
                color = Colors.RED
            print(f" {color}{avg*100:>6.0f}%{Colors.ENDC}", end="")
        else:
            print(f" {Colors.DIM}     -{Colors.ENDC}", end="")

    # Trend for average
    if len(sorted_iters) >= 2:
        first_avg: float = float(np.mean(list(all_results[sorted_iters[0]].values())))
        last_avg: float = float(np.mean(list(all_results[sorted_iters[-1]].values())))
        diff = last_avg - first_avg
        if diff > 0.05:
            trend = f"{Colors.GREEN}+{Colors.ENDC}"
        elif diff < -0.05:
            trend = f"{Colors.RED}-{Colors.ENDC}"
        else:
            trend = f"{Colors.YELLOW}={Colors.ENDC}"
        print(f"   {trend:>6}")
    else:
        print()

    print(
        f"\n  {Colors.DIM}Use without --compare for detailed single checkpoint analysis{Colors.ENDC}"
    )
