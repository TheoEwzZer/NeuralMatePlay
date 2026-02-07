"""Test: Good vs Bad Bishop."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

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

if TYPE_CHECKING:
    from src.alphazero.network import DualHeadNetwork


# Test positions for bishop quality evaluation
TEST_POSITIONS: list[dict[str, Any]] = [
    # === GOOD BISHOPS ===
    {
        "name": "Active Bishop (Open Diagonal)",
        "fen": "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 1",
        "description": "Black bishop on b4 is active",
        "expected_eval": "neutral",  # Both have active pieces
        "bishop_quality": "active",
    },
    {
        "name": "Fianchetto Bishop",
        "fen": "rnbqk1nr/ppppppbp/6p1/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        "description": "Fianchettoed bishop on g7",
        "expected_eval": "neutral",
        "bishop_quality": "fianchetto",
    },
    {
        "name": "Open Diagonal Control",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "description": "Bishop on c4 attacks f7",
        "expected_eval": "positive",
        "bishop_quality": "attacking",
    },
    # === BAD BISHOPS ===
    {
        "name": "Bad Bishop (Blocked)",
        "fen": "rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 0 1",
        "description": "French structure - bishops blocked",
        "expected_eval": "positive",  # White has space
        "bishop_quality": "blocked",
    },
    {
        "name": "Hemmed In Bishop",
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        "description": "f1 bishop needs development",
        "expected_eval": "neutral",
        "bishop_quality": "undeveloped",
    },
    {
        "name": "Pawns on Bishop Color",
        "fen": "8/pp3ppp/4p3/3pP3/3P4/4B3/PPP2PPP/8 w - - 0 1",
        "description": "White pawns on light squares with light bishop",
        "expected_eval": "neutral",  # Bad bishop structure
        "bishop_quality": "same_color_pawns",
    },
    # === BISHOP PAIR ===
    {
        "name": "Bishop Pair Advantage",
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1",
        "description": "Both sides have bishop pair",
        "expected_eval": "neutral",
        "bishop_quality": "pair_vs_pair",
    },
    {
        "name": "Bishop Pair vs Knights",
        "fen": "8/8/4k3/8/8/4K3/BB6/1nn5 w - - 0 1",
        "description": "Two bishops vs two knights",
        "expected_eval": "positive",  # Bishops usually better
        "bishop_quality": "pair_vs_knights",
    },
    # === OPPOSITE COLOR BISHOPS ===
    {
        "name": "Opposite Color Bishops",
        "fen": "8/8/4k3/8/8/4K3/B7/5b2 w - - 0 1",
        "description": "Opposite colored bishops - drawish",
        "expected_eval": "neutral",
        "bishop_quality": "opposite_color",
    },
    {
        "name": "OCB with Pawns",
        "fen": "8/pp3ppp/4pk2/8/8/4PK2/PP3PPP/5B2 w - - 0 1",
        "description": "Opposite bishops with pawns - still drawish",
        "expected_eval": "neutral",
        "bishop_quality": "opposite_color_endgame",
    },
]


def test_bishop_quality(network: DualHeadNetwork, results: TestResults) -> float:
    """Test if network evaluates bishop quality correctly."""
    print(header("TEST: Good vs Bad Bishop"))

    passed: float = 0
    total: int = len(TEST_POSITIONS)

    for test in TEST_POSITIONS:
        board: chess.Board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        policy: np.ndarray
        value: float
        policy, value = predict_for_board(board, network)

        expected: str = test["expected_eval"]
        quality: str = test["bishop_quality"]

        print(f"\n  Bishop quality: {quality}")
        print(f"  Value: {value:+.4f} (expected: {expected})")

        if expected == "positive":
            if value > 0.0:
                print(f"  {ok('Correctly sees bishop advantage')}")
                passed += 1
            elif value > -0.15:
                print(f"  {warn('Slight underestimation')}")
                passed += 0.5
            else:
                print(f"  {fail('Misses bishop advantage')}")

        elif expected == "negative":
            if value < 0.0:
                print(f"  {ok('Correctly sees bishop disadvantage')}")
                passed += 1
            elif value < 0.15:
                print(f"  {warn('Slight underestimation')}")
                passed += 0.5
            else:
                print(f"  {fail('Misses bishop disadvantage')}")

        elif expected == "neutral":
            if abs(value) < 0.3:
                print(f"  {ok('Correctly sees balanced position')}")
                passed += 1
            else:
                print(f"  {warn('Sees significant imbalance')}")
                passed += 0.5

        results.add_diagnostic("bishop_quality", f"{test['name']}_value", float(value))
        results.add_diagnostic("bishop_quality", f"{test['name']}_type", quality)

    score: float = passed / total
    results.add_diagnostic("bishop_quality", "total_tested", total)
    results.add_diagnostic("bishop_quality", "correct", passed)
    results.add("Bishop Quality", score >= 0.4, score, 1.0)

    return score
