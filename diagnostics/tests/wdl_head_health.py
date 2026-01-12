"""Test: WDL Head Health (MCTS Critical)."""

import numpy as np
import chess

from ..core import (
    Colors,
    TestResults,
    header,
    subheader,
    ok,
    fail,
    warn,
    encode_for_network,
)


def test_wdl_head_health(network, results: TestResults):
    """
    Test if the WDL head is healthy enough for MCTS to work.

    A collapsed WDL head outputs ~0 for all positions, making MCTS blind.
    This test is CRITICAL because MCTS = Policy × Value, so a broken
    WDL head renders the entire system useless regardless of policy quality.

    Tests:
    1. Variance across diverse positions (should be high)
    2. Range spanning most of [-1, 1]
    3. Clear separation between winning/losing positions
    """
    print(header("TEST: WDL Head Health (MCTS Critical)"))

    # Diverse test positions: clearly winning, clearly losing, neutral
    test_positions = [
        # Clearly winning for White
        ("k7/8/1K6/8/8/8/8/7Q w - - 0 1", "White has Q vs nothing", 1.0),
        ("8/8/8/8/8/2k5/8/KR6 w - - 0 1", "White has R vs nothing", 0.9),
        ("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "White +Q", 0.8),
        # Neutral
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Start", 0.0),
        (
            "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            "Early game",
            0.0,
        ),
        # Clearly losing for White (winning for Black)
        ("RNBQKBNR/PPPPPPPP/8/8/8/8/pppppppp/rnb1kbnr w kq - 0 1", "White -Q", -0.8),
        ("K7/8/1k6/8/8/8/8/7q w - - 0 1", "Black has Q vs nothing", -1.0),
        ("8/8/8/8/8/2K5/8/kr6 w - - 0 1", "Black has R vs nothing", -0.9),
    ]

    print(subheader("Value Distribution Analysis"))
    print(f"\n  {'Position':<25} {'Expected':>10} {'Actual':>10} {'Status':>10}")
    print("  " + "-" * 60)

    values = []
    expected = []
    correct_direction = 0

    for fen, desc, expected_sign in test_positions:
        board = chess.Board(fen)
        state = encode_for_network(board, network)
        _, value = network.predict_single(state)
        values.append(value)
        expected.append(expected_sign)

        # Check if direction is correct
        if expected_sign > 0.3:
            is_correct = value > 0.05
            status = ok("") if is_correct else fail("")
        elif expected_sign < -0.3:
            is_correct = value < -0.05
            status = ok("") if is_correct else fail("")
        else:
            is_correct = abs(value) < 0.5
            status = ok("") if is_correct else warn("")

        if is_correct:
            correct_direction += 1

        print(f"  {desc:<25} {expected_sign:>+10.2f} {value:>+10.4f} {status}")

        results.add_diagnostic("wdl_health", f"{desc}_expected", expected_sign)
        results.add_diagnostic("wdl_health", f"{desc}_actual", float(value))

    print("  " + "-" * 60)

    # Calculate health metrics
    value_std = np.std(values)
    value_range = max(values) - min(values)
    value_mean = np.mean(values)

    # Separation between winning and losing
    winning_values = [v for v, e in zip(values, expected) if e > 0.3]
    losing_values = [v for v, e in zip(values, expected) if e < -0.3]

    if winning_values and losing_values:
        separation = np.mean(winning_values) - np.mean(losing_values)
    else:
        separation = 0.0

    print(subheader("Health Metrics"))
    print(
        f"  Value std deviation:    {value_std:.4f}  {'(OK)' if value_std > 0.1 else '(COLLAPSED!)'}"
    )
    print(
        f"  Value range:            {value_range:.4f}  {'(OK)' if value_range > 0.3 else '(TOO NARROW!)'}"
    )
    print(f"  Value mean:             {value_mean:+.4f}")
    print(
        f"  Win/Loss separation:    {separation:.4f}  {'(OK)' if separation > 0.2 else '(CANNOT DISTINGUISH!)'}"
    )
    print(f"  Correct directions:     {correct_direction}/{len(test_positions)}")

    # Store diagnostics
    results.add_diagnostic("wdl_health", "std_deviation", float(value_std))
    results.add_diagnostic("wdl_health", "range", float(value_range))
    results.add_diagnostic("wdl_health", "mean", float(value_mean))
    results.add_diagnostic("wdl_health", "separation", float(separation))
    results.add_diagnostic("wdl_health", "correct_directions", correct_direction)
    results.add_diagnostic("wdl_health", "all_values", [float(v) for v in values])

    # Calculate health score (0-1)
    # Weights: std (40%), range (30%), separation (30%)
    std_score = min(1.0, value_std / 0.3)  # Perfect if std >= 0.3
    range_score = min(1.0, value_range / 1.0)  # Perfect if range >= 1.0
    sep_score = min(1.0, max(0, separation) / 0.5)  # Perfect if separation >= 0.5

    health_score = 0.4 * std_score + 0.3 * range_score + 0.3 * sep_score

    # Critical failure detection
    is_collapsed = value_std < 0.05 or value_range < 0.1

    print(subheader("MCTS Usability Score"))

    if is_collapsed:
        print(f"\n  {Colors.RED}{Colors.BOLD}[!] WDL HEAD COLLAPSED [!]{Colors.ENDC}")
        print(
            f"  {Colors.RED}MCTS is effectively BLIND - cannot distinguish positions{Colors.ENDC}"
        )
        print(
            f"  {Colors.RED}Network will make random-like moves regardless of policy quality{Colors.ENDC}"
        )
        health_score = 0.0  # Force zero score
        results.add_issue(
            "CRITICAL",
            "wdl_head",
            "WDL HEAD COLLAPSED - MCTS UNUSABLE",
            f"std={value_std:.4f}, range={value_range:.4f}. Network outputs ~constant value for all positions.",
        )
        results.add_recommendation(
            0,  # Highest priority
            "URGENT: Retrain from scratch or fix WDL head architecture",
            "Collapsed WDL head makes MCTS blind. Games are effectively random.",
        )
    else:
        bar_len = int(health_score * 20)
        if health_score >= 0.7:
            color = Colors.GREEN
            status = "HEALTHY"
        elif health_score >= 0.4:
            color = Colors.YELLOW
            status = "DEGRADED"
        else:
            color = Colors.RED
            status = "POOR"

        bar = color + "#" * bar_len + Colors.DIM + "-" * (20 - bar_len) + Colors.ENDC
        print(
            f"\n  Health: [{bar}] {health_score*100:.0f}% - {color}{status}{Colors.ENDC}"
        )

        if health_score < 0.7:
            results.add_issue(
                "HIGH" if health_score < 0.4 else "MEDIUM",
                "wdl_head",
                f"WDL head health is {status.lower()} ({health_score*100:.0f}%)",
                f"std={value_std:.4f}, range={value_range:.4f}, separation={separation:.4f}",
            )

    # Visual: show value distribution
    print(subheader("Value Distribution"))
    print(f"\n  Min: {min(values):+.4f}  Max: {max(values):+.4f}")

    # ASCII histogram
    bins = 10
    hist, edges = np.histogram(values, bins=bins, range=(-1, 1))
    max_count = max(hist) if max(hist) > 0 else 1

    print(f"\n  Distribution of values across test positions:")
    for i in range(bins):
        bar_width = int(hist[i] / max_count * 30) if max_count > 0 else 0
        label = f"{edges[i]:+.1f} to {edges[i+1]:+.1f}"
        bar = "#" * bar_width
        print(f"    {label}: {bar} ({hist[i]})")

    passed = health_score >= 0.5 and not is_collapsed
    results.add("WDL Head Health", passed, health_score, 1.0)

    return health_score
