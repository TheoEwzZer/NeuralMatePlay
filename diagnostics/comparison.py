"""Multi-checkpoint comparison utilities."""

import sys
import os
import io
import contextlib
import time
import numpy as np

# Ensure proper imports
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
if os.path.join(_root_dir, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_root_dir, "src"))

from alphazero import DualHeadNetwork
from .core import Colors, TestResults, header, subheader, get_available_checkpoints
from .tests import ALL_TESTS


def run_all_tests_on_network(
    network, silent: bool = True, show_progress: bool = False
) -> dict:
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
    scores = {}

    for i, (test_func, test_name, _) in enumerate(ALL_TESTS):
        if show_progress:
            print(
                f"      [{i+1:>2}/{len(ALL_TESTS)}] {test_name:<22}",
                end=" ",
                flush=True,
            )

        dummy_results = TestResults()

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            score = test_func(network, dummy_results)

        scores[test_name] = score

        if show_progress:
            if score >= 0.7:
                color = Colors.GREEN
            elif score >= 0.4:
                color = Colors.YELLOW
            else:
                color = Colors.RED
            print(f"{color}{score*100:>5.0f}%{Colors.ENDC}")

    return scores


def compare_two_models(model1_path: str, model2_path: str, quick: bool = False):
    """
    Compare two specific model files on all tests.
    OPTIMIZED: Each model is loaded once, then all tests run on it.
    """
    print(header("COMPARING TWO MODELS"))

    # Get model names for display
    name1 = os.path.basename(model1_path)
    name2 = os.path.basename(model2_path)

    print(f"  Running {len(ALL_TESTS)} tests on each...")

    # Load and test model 1
    print(f"\n  {Colors.BOLD}Model 1: {name1}{Colors.ENDC}")
    try:
        network1 = DualHeadNetwork.load(model1_path)
    except Exception as e:
        print(f"    {Colors.RED}[ERROR] Cannot load: {e}{Colors.ENDC}")
        return

    start = time.time()
    scores1 = run_all_tests_on_network(network1, silent=True, show_progress=True)
    elapsed1 = time.time() - start
    avg1 = np.mean(list(scores1.values()))
    color1 = (
        Colors.GREEN if avg1 >= 0.7 else (Colors.YELLOW if avg1 >= 0.4 else Colors.RED)
    )
    print(f"      {'─'*35}")
    print(f"      Average: {color1}{avg1*100:.0f}%{Colors.ENDC} ({elapsed1:.1f}s)")
    del network1  # Free memory

    # Load and test model 2
    print(f"\n  {Colors.BOLD}Model 2: {name2}{Colors.ENDC}")
    try:
        network2 = DualHeadNetwork.load(model2_path)
    except Exception as e:
        print(f"    {Colors.RED}[ERROR] Cannot load: {e}{Colors.ENDC}")
        return

    start = time.time()
    scores2 = run_all_tests_on_network(network2, silent=True, show_progress=True)
    elapsed2 = time.time() - start
    avg2 = np.mean(list(scores2.values()))
    color2 = (
        Colors.GREEN if avg2 >= 0.7 else (Colors.YELLOW if avg2 >= 0.4 else Colors.RED)
    )
    print(f"      {'─'*35}")
    print(f"      Average: {color2}{avg2*100:.0f}%{Colors.ENDC} ({elapsed2:.1f}s)")
    del network2  # Free memory

    # Build results structure
    results = {}
    weights = {}
    for _, test_name, weight in ALL_TESTS:
        weights[test_name] = weight
        results[test_name] = [
            (name1, scores1.get(test_name, 0)),
            (name2, scores2.get(test_name, 0)),
        ]

    # Summary table
    print(header("COMPARISON SUMMARY"))

    # Truncate names for header
    h1 = name1[:15] + ".." if len(name1) > 17 else name1
    h2 = name2[:15] + ".." if len(name2) > 17 else name2

    print(f"\n  {'Test':<18} {'Wt':>4} {h1:>15} {h2:>15}  {'Winner':>10}")
    print("  " + "-" * 75)

    model1_wins = 0
    model2_wins = 0
    model1_weighted = 0
    model2_weighted = 0
    total_weight = 0

    for test_name, scores in results.items():
        score1 = scores[0][1]
        score2 = scores[1][1]
        weight = weights.get(test_name, 1.0)

        model1_weighted += score1 * weight
        model2_weighted += score2 * weight
        total_weight += weight

        # Colors for scores
        c1 = (
            Colors.GREEN
            if score1 >= 0.7
            else (Colors.YELLOW if score1 >= 0.4 else Colors.RED)
        )
        c2 = (
            Colors.GREEN
            if score2 >= 0.7
            else (Colors.YELLOW if score2 >= 0.4 else Colors.RED)
        )

        # Weight indicator
        wt_str = f"x{weight:.0f}" if weight > 1 else ""

        # Determine winner (weighted wins count more)
        diff = score2 - score1
        if diff > 0.05:
            winner = f"{Colors.CYAN}→ M2{Colors.ENDC}"
            model2_wins += weight  # Weighted wins
        elif diff < -0.05:
            winner = f"{Colors.CYAN}M1 ←{Colors.ENDC}"
            model1_wins += weight  # Weighted wins
        else:
            winner = f"{Colors.DIM}tie{Colors.ENDC}"

        # Highlight high-weight tests (pad BEFORE adding ANSI codes)
        name_padded = f"{test_name:<18}"
        if weight > 1:
            name_display = f"{Colors.BOLD}{name_padded}{Colors.ENDC}"
        else:
            name_display = name_padded

        print(
            f"  {name_display} {wt_str:>4} {c1}{score1*100:>13.0f}%{Colors.ENDC} {c2}{score2*100:>13.0f}%{Colors.ENDC}  {winner:>10}"
        )

    print("  " + "-" * 75)

    # Weighted averages
    wavg1 = model1_weighted / total_weight if total_weight > 0 else 0
    wavg2 = model2_weighted / total_weight if total_weight > 0 else 0

    c1 = (
        Colors.GREEN
        if wavg1 >= 0.7
        else (Colors.YELLOW if wavg1 >= 0.4 else Colors.RED)
    )
    c2 = (
        Colors.GREEN
        if wavg2 >= 0.7
        else (Colors.YELLOW if wavg2 >= 0.4 else Colors.RED)
    )

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


def run_comparison(checkpoint_dir: str, checkpoint_type: str = "train"):
    """
    Run comparison across all checkpoints with ALL tests.
    OPTIMIZED: Each checkpoint is loaded only once, then all tests run on it.
    """
    print(header("MULTI-CHECKPOINT COMPARISON"))

    checkpoints = get_available_checkpoints(checkpoint_dir, checkpoint_type)

    if not checkpoints:
        type_name = "pretrain" if checkpoint_type == "pretrain" else "train"
        print(
            f"  {Colors.RED}No {type_name} checkpoints found in {checkpoint_dir}!{Colors.ENDC}"
        )
        return

    print(f"  Found {len(checkpoints)} checkpoint(s)")
    print(f"  Running {len(ALL_TESTS)} tests on each...\n")

    # Structure: {iteration: {test_name: score}}
    all_results = {}
    sorted_iters = []

    total_start = time.time()

    for iteration, path in checkpoints:
        print(f"\n  {Colors.BOLD}Checkpoint {iteration}{Colors.ENDC}")

        try:
            network = DualHeadNetwork.load(path)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(
                    f"    {Colors.YELLOW}Skipped (incompatible architecture){Colors.ENDC}"
                )
                continue
            raise

        # Run all tests on this network (loaded once!) with progress
        start = time.time()
        scores = run_all_tests_on_network(network, silent=True, show_progress=True)
        elapsed = time.time() - start

        all_results[iteration] = scores
        sorted_iters.append(iteration)

        # Show summary for this checkpoint
        avg_score = np.mean(list(scores.values()))
        if avg_score >= 0.7:
            color = Colors.GREEN
        elif avg_score >= 0.4:
            color = Colors.YELLOW
        else:
            color = Colors.RED
        print(f"      {'─'*35}")
        print(
            f"      Average: {color}{avg_score*100:.0f}%{Colors.ENDC} ({elapsed:.1f}s)"
        )

        # Free memory
        del network

    total_elapsed = time.time() - total_start

    if not all_results:
        print(f"\n  {Colors.RED}No compatible checkpoints found!{Colors.ENDC}")
        return

    print(f"\n  Total time: {total_elapsed:.1f}s")

    # Get test names in order
    test_names = [name for _, name, _ in ALL_TESTS]
    test_weights = {name: weight for _, name, weight in ALL_TESTS}

    # Print summary table
    print(header("COMPARISON SUMMARY"))

    # Header
    iter_headers = "".join(f"{it:>8}" for it in sorted_iters)
    print(f"\n  {'Test':<22} {iter_headers}   {'Trend':>6}")
    print("  " + "-" * (24 + len(sorted_iters) * 8 + 8))

    # Rows - one per test
    for test_name in test_names:
        weight = test_weights.get(test_name, 1.0)

        # Highlight high-weight tests
        if weight > 1:
            name_display = f"{Colors.BOLD}{test_name:<22}{Colors.ENDC}"
        else:
            name_display = f"{test_name:<22}"

        row = f"  {name_display}"

        scores_for_trend = []
        for it in sorted_iters:
            if it in all_results and test_name in all_results[it]:
                sc = all_results[it][test_name]
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

        # Trend
        if len(scores_for_trend) >= 2:
            diff = scores_for_trend[-1] - scores_for_trend[0]
            if diff > 0.1:
                trend = f"{Colors.GREEN}↑{Colors.ENDC}"
            elif diff < -0.1:
                trend = f"{Colors.RED}↓{Colors.ENDC}"
            else:
                trend = f"{Colors.YELLOW}→{Colors.ENDC}"
        else:
            trend = "-"
        row += f"   {trend:>6}"
        print(row)

    print("  " + "-" * (24 + len(sorted_iters) * 8 + 8))

    # Overall average per iteration
    print(f"\n  {'AVERAGE':<22}", end="")
    for it in sorted_iters:
        if it in all_results:
            avg = np.mean(list(all_results[it].values()))
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
        first_avg = np.mean(list(all_results[sorted_iters[0]].values()))
        last_avg = np.mean(list(all_results[sorted_iters[-1]].values()))
        diff = last_avg - first_avg
        if diff > 0.05:
            trend = f"{Colors.GREEN}↑{Colors.ENDC}"
        elif diff < -0.05:
            trend = f"{Colors.RED}↓{Colors.ENDC}"
        else:
            trend = f"{Colors.YELLOW}→{Colors.ENDC}"
        print(f"   {trend:>6}")
    else:
        print()

    print(
        f"\n  {Colors.DIM}Use without --compare for detailed single checkpoint analysis{Colors.ENDC}"
    )
