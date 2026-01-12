"""Test: Exchange Evaluation."""

import numpy as np
import chess

from ..core import (
    TestResults,
    header,
    subheader,
    ok,
    fail,
    warn,
    predict_for_board,
)


# Test positions for piece exchange evaluation
TEST_POSITIONS = [
    # === ROOK VS MINOR PIECES ===
    {
        "name": "Rook vs Bishop+Knight",
        "fen": "8/8/8/4k3/8/4K3/R7/3bn3 w - - 0 1",
        "description": "Rook vs B+N - slightly worse for rook",
        "expected_eval": "negative",  # Two minors slightly better
    },
    {
        "name": "Rook+Pawn vs Two Minors",
        "fen": "8/8/8/4k3/4P3/4K3/R7/3bn3 w - - 0 1",
        "description": "R+P vs B+N - roughly equal",
        "expected_eval": "neutral",
    },
    # === BISHOP PAIR ===
    {
        "name": "Bishop Pair Advantage",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 1",
        "description": "Both sides have bishops - normal",
        "expected_eval": "neutral",
    },
    {
        "name": "Bishop Pair vs Knights",
        "fen": "8/8/8/4k3/8/4K3/BB6/3nn3 w - - 0 1",
        "description": "Two bishops vs two knights - bishops usually better",
        "expected_eval": "positive",
    },
    # === QUEEN VS PIECES ===
    {
        "name": "Queen vs Rook+Minor",
        "fen": "8/8/8/4k3/8/4K3/Q7/3rn3 w - - 0 1",
        "description": "Queen vs R+N - roughly equal",
        "expected_eval": "neutral",
    },
    {
        "name": "Queen vs Two Rooks",
        "fen": "8/8/8/4k3/8/4K3/Q7/1rr5 w - - 0 1",
        "description": "Queen vs 2 Rooks - rooks slightly better",
        "expected_eval": "negative",
    },
    # === MINOR PIECE IMBALANCES ===
    {
        "name": "Bishop vs Knight (Open)",
        "fen": "8/8/4k3/8/8/4K3/B7/4n3 w - - 0 1",
        "description": "Bishop vs Knight open - bishop often better",
        "expected_eval": "positive",
    },
    {
        "name": "Bishop vs Knight (Closed)",
        "fen": "8/pp6/1p6/1Pk5/1P6/1P6/1P6/B3K3 w - - 0 1",
        "description": "Closed position - bishop blocked",
        "expected_eval": "neutral",  # Bad bishop
    },
    # === MATERIAL IMBALANCES ===
    {
        "name": "Three Pawns for Piece",
        "fen": "8/8/4k3/8/PPP5/4K3/8/4n3 w - - 0 1",
        "description": "3 pawns vs knight - depends on position",
        "expected_eval": "neutral",
    },
    {
        "name": "Exchange Up",
        "fen": "8/8/4k3/8/8/4K3/R7/4b3 w - - 0 1",
        "description": "Rook vs Bishop - exchange up",
        "expected_eval": "positive",
    },
]


def test_exchange_eval(network, results: TestResults):
    """Test if network evaluates piece exchanges correctly."""
    print(header("TEST: Exchange Evaluation"))

    passed = 0
    total = len(TEST_POSITIONS)

    for test in TEST_POSITIONS:
        board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        policy, value = predict_for_board(board, network)

        expected = test["expected_eval"]
        print(f"\n  Value: {value:+.4f} (expected: {expected})")

        if expected == "positive":
            if value > 0.1:
                print(f"  {ok('Correctly sees advantage')}")
                passed += 1
            elif value > -0.1:
                print(f"  {warn('Slight underestimation')}")
                passed += 0.5
            else:
                print(f"  {fail('Wrong evaluation')}")

        elif expected == "negative":
            if value < -0.1:
                print(f"  {ok('Correctly sees disadvantage')}")
                passed += 1
            elif value < 0.1:
                print(f"  {warn('Slight underestimation')}")
                passed += 0.5
            else:
                print(f"  {fail('Wrong evaluation')}")

        elif expected == "neutral":
            if abs(value) < 0.25:
                print(f"  {ok('Correctly sees balance')}")
                passed += 1
            elif abs(value) < 0.4:
                print(f"  {warn('Slight imbalance detected')}")
                passed += 0.5
            else:
                print(f"  {fail('Wrong evaluation - should be neutral')}")

        results.add_diagnostic("exchange_eval", f"{test['name']}_value", float(value))

    score = passed / total
    results.add_diagnostic("exchange_eval", "total_tested", total)
    results.add_diagnostic("exchange_eval", "correct", passed)
    results.add("Exchange Evaluation", score >= 0.4, score, 1.0)

    return score
