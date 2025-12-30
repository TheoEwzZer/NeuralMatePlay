"""Test: Opening Development."""

import numpy as np
import chess

from alphazero.move_encoding import decode_move
from ..core import (
    Colors,
    TestResults,
    header,
    subheader,
    ok,
    fail,
    warn,
    predict_for_board,
)


def test_development(network, results: TestResults):
    """Test if network develops pieces reasonably in opening."""
    print(header("TEST: Opening Development"))

    board = chess.Board()

    print(board)
    print("\n  Starting position - analyzing opening move preferences")

    policy, value = predict_for_board(board, network)

    # Categorize opening moves
    excellent_moves = {"e2e4", "d2d4"}  # Central pawn pushes
    good_moves = {"g1f3", "c2c4", "b1c3"}  # Development
    okay_moves = {"e2e3", "d2d3", "g2g3", "b2b3"}  # Slow but playable
    dubious_moves = {"h2h4", "a2a4", "g1h3", "b1a3", "f2f4"}  # Edge moves

    top_10 = np.argsort(policy)[::-1][:10]

    print(f"\n  Value evaluation: {value:+.4f}")
    print(f"\n  {'Rank':<6} {'Move':<8} {'Prob':>8} {'Category':<15}")
    print("  " + "-" * 45)

    move_analysis = {"excellent": 0, "good": 0, "okay": 0, "dubious": 0, "other": 0}

    excellent_prob = 0
    good_prob = 0
    dubious_prob = 0

    for i, idx in enumerate(top_10):
        move = decode_move(idx, board)
        prob = policy[idx]
        if move:
            uci = move.uci()
            if uci in excellent_moves:
                category = "EXCELLENT"
                color = Colors.GREEN
                move_analysis["excellent"] += 1
                excellent_prob += prob
            elif uci in good_moves:
                category = "Good"
                color = Colors.GREEN
                move_analysis["good"] += 1
                good_prob += prob
            elif uci in okay_moves:
                category = "Okay"
                color = Colors.YELLOW
                move_analysis["okay"] += 1
            elif uci in dubious_moves:
                category = "Dubious"
                color = Colors.RED
                move_analysis["dubious"] += 1
                dubious_prob += prob
            else:
                category = ""
                color = ""
                move_analysis["other"] += 1

            end_color = Colors.ENDC if color else ""
            print(f"  {color}{i+1:<6} {uci:<8} {prob*100:>7.2f}% {category}{end_color}")

    # Analysis
    print(subheader("Opening Move Analysis"))
    print(f"  Probability on excellent moves (e4, d4):    {excellent_prob*100:>6.1f}%")
    print(f"  Probability on good moves (Nf3, c4, Nc3):   {good_prob*100:>6.1f}%")
    print(f"  Probability on dubious moves (h4, a4, etc): {dubious_prob*100:>6.1f}%")

    results.add_diagnostic("opening", "excellent_prob", float(excellent_prob))
    results.add_diagnostic("opening", "good_prob", float(good_prob))
    results.add_diagnostic("opening", "dubious_prob", float(dubious_prob))
    results.add_diagnostic("opening", "value_eval", float(value))

    # Check top move quality
    top_move = decode_move(top_10[0], board)
    top_uci = top_move.uci() if top_move else None

    if top_uci in excellent_moves:
        print(f"\n  {ok(f'Excellent opening choice: {top_uci}')}")
        passed = True
        score = 1.0
    elif top_uci in good_moves:
        print(f"\n  {ok(f'Good opening choice: {top_uci}')}")
        passed = True
        score = 0.8
    elif top_uci in okay_moves:
        print(f"\n  {warn(f'Acceptable but passive opening: {top_uci}')}")
        passed = True
        score = 0.5
    else:
        print(f"\n  {fail(f'Poor opening choice: {top_uci}')}")
        passed = False
        score = 0.2
        results.add_issue(
            "MEDIUM",
            "opening",
            f"Network plays dubious opening move {top_uci}",
            f"Probability: {policy[top_10[0]]*100:.1f}%, dubious moves total: {dubious_prob*100:.1f}%",
        )

    # Additional concern if dubious moves have high probability
    if dubious_prob > 0.3:
        results.add_issue(
            "MEDIUM",
            "opening",
            f"High probability ({dubious_prob*100:.0f}%) on dubious opening moves",
            "Network may play h4, a4, etc. frequently",
        )

    results.add("Opening Development", passed, score, 1.0)
    return score
