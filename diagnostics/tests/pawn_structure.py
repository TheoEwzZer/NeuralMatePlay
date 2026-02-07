"""Test: Pawn Structure Understanding."""

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


# Test positions for pawn structure evaluation
TEST_POSITIONS: list[dict[str, Any]] = [
    # === DOUBLED PAWNS ===
    {
        "name": "Doubled Pawns (White)",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "description": "Starting position - no doubled pawns",
        "structure_type": "normal",
        "expected_eval": "neutral",
    },
    {
        "name": "Doubled c-Pawns",
        "fen": "rnbqkbnr/pp1ppppp/8/8/8/2P5/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        "description": "White has doubled c-pawns potential",
        "structure_type": "doubled",
        "expected_eval": "neutral",  # Not doubled yet
    },
    # === ISOLATED PAWNS ===
    {
        "name": "Isolated d-Pawn",
        "fen": "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 1",
        "description": "Symmetrical d-pawns",
        "structure_type": "isolated_d",
        "expected_eval": "neutral",
    },
    {
        "name": "IQP Position",
        "fen": "rnbqkb1r/ppp2ppp/4pn2/8/3P4/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 1",
        "description": "Isolated Queen Pawn structure",
        "structure_type": "iqp",
        "expected_eval": "neutral",  # Dynamic play compensates
    },
    # === PAWN CHAINS ===
    {
        "name": "Strong Pawn Chain",
        "fen": "rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 0 1",
        "description": "French Defense pawn chain",
        "structure_type": "chain",
        "expected_eval": "positive",  # White has space
    },
    {
        "name": "Pawn Chain Base",
        "fen": "rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/2N5/PPP2PPP/R1BQKBNR b KQkq - 0 1",
        "description": "Attack the base of the chain",
        "structure_type": "chain",
        "expected_eval": "positive",
    },
    # === BACKWARD PAWNS ===
    {
        "name": "Backward Pawn",
        "fen": "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 1",
        "description": "Potential backward pawn on d5",
        "structure_type": "backward",
        "expected_eval": "positive",  # White has better structure
    },
    # === PASSED PAWN CREATION ===
    {
        "name": "Create Passer",
        "fen": "8/pp3ppp/8/3P4/8/8/PP3PPP/8 w - - 0 1",
        "description": "White has potential passed pawn",
        "structure_type": "passed",
        "expected_eval": "positive",
    },
    # === PAWN MAJORITY ===
    {
        "name": "Queenside Majority",
        "fen": "8/ppp2ppp/8/8/8/8/PPPP1PPP/8 w - - 0 1",
        "description": "Queenside vs Kingside majority",
        "structure_type": "majority",
        "expected_eval": "neutral",
    },
    {
        "name": "Central Majority",
        "fen": "8/pp3ppp/8/3PP3/8/8/PP3PPP/8 w - - 0 1",
        "description": "Central pawn majority",
        "structure_type": "central_majority",
        "expected_eval": "positive",
    },
]


def test_pawn_structure(network: DualHeadNetwork, results: TestResults) -> float:
    """Test if network understands pawn structure."""
    print(header("TEST: Pawn Structure"))

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
        structure: str = test["structure_type"]

        print(f"\n  Structure type: {structure}")
        print(f"  Value: {value:+.4f} (expected: {expected})")

        if expected == "positive":
            if value > 0.0:
                print(f"  {ok('Correctly sees structural advantage')}")
                passed += 1
            elif value > -0.15:
                print(f"  {warn('Slight underestimation')}")
                passed += 0.5
            else:
                print(f"  {fail('Misses structural advantage')}")

        elif expected == "negative":
            if value < 0.0:
                print(f"  {ok('Correctly sees structural weakness')}")
                passed += 1
            elif value < 0.15:
                print(f"  {warn('Slight underestimation')}")
                passed += 0.5
            else:
                print(f"  {fail('Misses structural weakness')}")

        elif expected == "neutral":
            if abs(value) < 0.25:
                print(f"  {ok('Correctly sees balanced structure')}")
                passed += 1
            else:
                print(f"  {warn('Sees imbalance in balanced structure')}")
                passed += 0.5

        results.add_diagnostic("pawn_structure", f"{test['name']}_value", float(value))
        results.add_diagnostic("pawn_structure", f"{test['name']}_type", structure)

    score: float = passed / total
    results.add_diagnostic("pawn_structure", "total_tested", total)
    results.add_diagnostic("pawn_structure", "correct", passed)
    results.add("Pawn Structure", score >= 0.4, score, 1.0)

    return score
