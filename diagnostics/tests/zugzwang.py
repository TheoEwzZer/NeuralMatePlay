"""Test: Zugzwang Recognition."""

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


# Zugzwang positions - any move loses
TEST_POSITIONS = [
    # === KING AND PAWN ZUGZWANGS ===
    {
        "name": "Basic King Zugzwang",
        "fen": "8/8/8/8/8/1k6/8/1K6 w - - 0 1",
        "description": "White to move loses opposition",
        "side_to_move_disadvantage": True,
    },
    {
        "name": "Opposition",
        "fen": "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
        "description": "White has opposition - winning",
        "side_to_move_disadvantage": False,  # White wants to move here
    },
    {
        "name": "Triangulation Setup",
        "fen": "8/8/8/8/4k3/8/4PK2/8 w - - 0 1",
        "description": "White can triangulate",
        "side_to_move_disadvantage": False,
    },
    # === PIECE ZUGZWANGS ===
    {
        "name": "Rook vs Pawn Zugzwang",
        "fen": "8/1R6/7k/6p1/8/8/8/6K1 b - - 0 1",
        "description": "Black must move king or lose pawn",
        "side_to_move_disadvantage": True,
    },
    {
        "name": "Bishop Zugzwang",
        "fen": "8/8/8/8/8/k7/b7/1K6 w - - 0 1",
        "description": "Stalemate themes",
        "side_to_move_disadvantage": False,
    },
    # === RECIPROCAL ZUGZWANGS ===
    {
        "name": "Mutual Zugzwang",
        "fen": "8/8/1p6/1P6/1K6/8/8/1k6 w - - 0 1",
        "description": "Whoever moves loses",
        "side_to_move_disadvantage": True,
    },
    {
        "name": "Complex Zugzwang",
        "fen": "6k1/5p2/5Pp1/8/7K/8/8/8 b - - 0 1",
        "description": "Black to move is zugzwang",
        "side_to_move_disadvantage": True,
    },
    # === EVALUATION POSITIONS ===
    {
        "name": "Near Zugzwang",
        "fen": "8/8/8/3k4/8/3K4/3P4/8 w - - 0 1",
        "description": "Close to zugzwang position",
        "side_to_move_disadvantage": False,
    },
]


def test_zugzwang(network, results: TestResults):
    """Test if network recognizes zugzwang positions."""
    print(header("TEST: Zugzwang Recognition"))

    passed = 0
    total = len(TEST_POSITIONS)

    for test in TEST_POSITIONS:
        board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        policy, value = predict_for_board(board, network)

        # In zugzwang, the side to move is at disadvantage
        # Value should be negative if it's disadvantageous to move
        print(f"\n  Value: {value:+.4f}")
        print(f"  Side to move disadvantaged: {test['side_to_move_disadvantage']}")

        # Check if evaluation matches zugzwang understanding
        if test["side_to_move_disadvantage"]:
            # Value should be negative (bad for side to move)
            if value < 0.0:
                print(f"  {ok('Recognizes disadvantage of moving')}")
                passed += 1
            elif value < 0.15:
                print(f"  {warn('Slight recognition of zugzwang')}")
                passed += 0.5
            else:
                print(f"  {fail('Misses zugzwang - thinks position is good')}")
        else:
            # Side to move has advantage
            if value >= -0.1:
                print(f"  {ok('Correctly evaluates position')}")
                passed += 1
            else:
                print(f"  {warn('Evaluation differs')}")
                passed += 0.5

        results.add_diagnostic("zugzwang", f"{test['name']}_value", float(value))
        results.add_diagnostic(
            "zugzwang",
            f"{test['name']}_correct",
            (test["side_to_move_disadvantage"] and value < 0)
            or (not test["side_to_move_disadvantage"] and value >= -0.1),
        )

    score = passed / total
    results.add_diagnostic("zugzwang", "total_tested", total)
    results.add_diagnostic("zugzwang", "correct", passed)
    results.add("Zugzwang", score >= 0.4, score, 1.0)

    return score
