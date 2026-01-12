"""Test: Fortress Recognition."""

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


# Fortress positions - draw despite material deficit
TEST_POSITIONS = [
    # === ROOK FORTRESSES ===
    {
        "name": "Rook vs Queen Fortress",
        "fen": "8/8/8/8/8/1k6/1r6/1K4Q1 b - - 0 1",
        "description": "Rook can hold vs Queen in some cases",
        "expected_eval": "losing",  # Usually losing but drawable
        "material_diff": -4,  # Down a lot but fortress possible
    },
    {
        "name": "Rook Pawn Fortress",
        "fen": "8/8/8/8/8/k7/P7/K7 w - - 0 1",
        "description": "Wrong color bishop can't win",
        "expected_eval": "draw",  # Can be drawn
        "material_diff": 1,
    },
    # === BISHOP FORTRESSES ===
    {
        "name": "Wrong Color Bishop",
        "fen": "7k/8/7K/8/B7/8/7P/8 w - - 0 1",
        "description": "Bishop wrong color for h-pawn",
        "expected_eval": "draw",  # Famous draw
        "material_diff": 4,
    },
    {
        "name": "Opposite Color Bishops",
        "fen": "8/8/4k3/4p3/4P3/3BK3/8/5b2 w - - 0 1",
        "description": "Opposite bishops - likely draw",
        "expected_eval": "draw",
        "material_diff": 0,
    },
    # === KNIGHT FORTRESSES ===
    {
        "name": "Knight vs Pawns Fortress",
        "fen": "8/8/8/4k3/4n3/4K3/3PP3/8 w - - 0 1",
        "description": "Knight blockade",
        "expected_eval": "draw",
        "material_diff": 1,
    },
    # === EVALUATION POSITIONS ===
    {
        "name": "Material Advantage (Not Fortress)",
        "fen": "8/8/8/4k3/8/4K3/4Q3/8 w - - 0 1",
        "description": "Queen vs nothing - winning",
        "expected_eval": "winning",
        "material_diff": 9,
    },
    {
        "name": "Rook Endgame (Not Fortress)",
        "fen": "8/8/8/4k3/8/4K3/4R3/8 w - - 0 1",
        "description": "Rook vs nothing - winning",
        "expected_eval": "winning",
        "material_diff": 5,
    },
    {
        "name": "Piece Down (Should Lose)",
        "fen": "8/8/8/4k3/4n3/4K3/8/8 w - - 0 1",
        "description": "Knight up - Black winning",
        "expected_eval": "losing",
        "material_diff": -3,
    },
]


def test_fortress(network, results: TestResults):
    """Test if network recognizes fortress positions."""
    print(header("TEST: Fortress Recognition"))

    passed = 0
    total = len(TEST_POSITIONS)

    for test in TEST_POSITIONS:
        board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        policy, value = predict_for_board(board, network)

        expected = test["expected_eval"]
        material = test["material_diff"]

        print(f"\n  Value: {value:+.4f}")
        print(f"  Material: {material:+d}")
        print(f"  Expected: {expected}")

        # Check evaluation
        if expected == "winning":
            if value > 0.3:
                print(f"  {ok('Correctly sees winning position')}")
                passed += 1
            elif value > 0.0:
                print(f"  {warn('Sees advantage but underestimates')}")
                passed += 0.5
            else:
                print(f"  {fail('Misses winning position')}")

        elif expected == "losing":
            if value < -0.3:
                print(f"  {ok('Correctly sees losing position')}")
                passed += 1
            elif value < 0.0:
                print(f"  {warn('Sees disadvantage but underestimates')}")
                passed += 0.5
            else:
                print(f"  {fail('Misses losing position')}")

        elif expected == "draw":
            # Fortress should be evaluated close to 0 despite material
            if abs(value) < 0.4:
                print(f"  {ok('Recognizes drawish fortress')}")
                passed += 1
            elif abs(value) < 0.6:
                print(f"  {warn('Partial fortress recognition')}")
                passed += 0.5
            else:
                print(f"  {fail('Misses fortress - evaluates on material')}")

        results.add_diagnostic("fortress", f"{test['name']}_value", float(value))
        results.add_diagnostic("fortress", f"{test['name']}_material", material)

    score = passed / total
    results.add_diagnostic("fortress", "total_tested", total)
    results.add_diagnostic("fortress", "correct", passed)
    results.add("Fortress", score >= 0.4, score, 1.0)

    return score
