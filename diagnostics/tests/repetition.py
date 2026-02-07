"""Test: Repetition Awareness."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

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

if TYPE_CHECKING:
    from src.alphazero.network import DualHeadNetwork


# Test positions for repetition awareness
TEST_POSITIONS: list[dict[str, Any]] = [
    # === SEEK DRAW (when losing) ===
    {
        "name": "Seek Perpetual (Losing)",
        "fen": "6k1/5ppp/8/8/8/8/5PPP/q5K1 w - - 0 1",
        "description": "White is losing - should seek perpetual if possible",
        "situation": "losing",
        "expected_behavior": "drawing_resource",
    },
    {
        "name": "Force Repetition",
        "fen": "5rk1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1",
        "description": "Down material - perpetual check?",
        "situation": "evaluate",
        "expected_behavior": "check_checks",
    },
    # === AVOID DRAW (when winning) ===
    {
        "name": "Avoid Stalemate",
        "fen": "7k/8/6K1/8/8/8/8/7Q w - - 0 1",
        "bad_moves": ["h1h7"],  # Would be stalemate
        "description": "Don't stalemate when winning",
        "situation": "winning",
        "expected_behavior": "avoid_draw",
    },
    {
        "name": "Press Advantage",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 1",
        "description": "Don't repeat moves when better",
        "situation": "slight_advantage",
        "expected_behavior": "progress",
    },
    # === EVALUATION IN REPEATED POSITIONS ===
    {
        "name": "Equal Position",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "description": "Starting position - balanced",
        "situation": "equal",
        "expected_behavior": "neutral_eval",
    },
    {
        "name": "Perpetual Check Available",
        "fen": "5k2/5p2/5K2/8/8/8/8/4Q3 w - - 0 1",
        "expected_moves": ["e1e8", "e1f2"],  # Can deliver checks
        "description": "Perpetual check possibility",
        "situation": "evaluate",
        "expected_behavior": "check_checks",
    },
    # === THREEFOLD SETUP ===
    {
        "name": "Near Threefold",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "description": "Position that could repeat",
        "situation": "slight_advantage",
        "expected_behavior": "progress",
    },
    {
        "name": "Defend and Hold",
        "fen": "8/8/8/8/8/5k2/8/4RK2 w - - 0 1",
        "description": "Winning position - don't repeat",
        "situation": "winning",
        "expected_behavior": "progress",
    },
]


def test_repetition(network: DualHeadNetwork, results: TestResults) -> float:
    """Test if network handles repetition correctly."""
    print(header("TEST: Repetition Awareness"))

    passed: float = 0
    total: int = len(TEST_POSITIONS)

    for test in TEST_POSITIONS:
        board: chess.Board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        policy: np.ndarray
        value: float
        policy, value = predict_for_board(board, network)

        top_5: np.ndarray = np.argsort(policy)[::-1][:5]
        behavior: str = test["expected_behavior"]
        situation: str = test["situation"]

        print(f"\n  Value: {value:+.4f}")
        print(f"  Situation: {situation}")

        # Show top moves
        print(f"  {'Rank':<6} {'Move':<8} {'Prob':>8}")
        print("  " + "-" * 25)

        top_move: chess.Move | None = None
        for i, idx in enumerate(top_5):
            move = decode_move(idx, board)
            prob = policy[idx]
            if move:
                if i == 0:
                    top_move = move
                print(f"  {i+1:<6} {move.uci():<8} {prob*100:>7.2f}%")

        # Check bad moves if specified
        if "bad_moves" in test:
            bad_set: set[str] = set(test["bad_moves"])
            if top_move and top_move.uci() in bad_set:
                print(f"\n  {fail('Plays drawing/stalemating move when winning!')}")
                results.add_diagnostic("repetition", f"{test['name']}_bad_move", True)
            else:
                print(f"\n  {ok('Avoids bad drawing move')}")
                passed += 1
                results.add_diagnostic("repetition", f"{test['name']}_bad_move", False)
            continue

        # Check expected moves if specified
        if "expected_moves" in test:
            expected_set: set[str] = set(test["expected_moves"])
            found_at: int | None = None
            for i, idx in enumerate(top_5):
                move = decode_move(idx, board)
                if move and move.uci() in expected_set:
                    found_at = i + 1
                    break

            if found_at == 1:
                print(f"\n  {ok('Finds expected move')}")
                passed += 1
            elif found_at:
                print(f"\n  {warn(f'Expected move at rank {found_at}')}")
                passed += 0.5
            else:
                print(f"\n  {warn('Different move choice')}")
                passed += 0.5
            continue

        # Evaluate based on situation
        if situation == "winning" and value > 0.3:
            print(f"\n  {ok('Correctly sees winning position')}")
            passed += 1
        elif situation == "losing" and value < -0.3:
            print(f"\n  {ok('Correctly sees losing position')}")
            passed += 1
        elif situation == "equal" and abs(value) < 0.2:
            print(f"\n  {ok('Correctly sees equal position')}")
            passed += 1
        elif situation == "slight_advantage" and value > 0.0:
            print(f"\n  {ok('Sees slight advantage')}")
            passed += 1
        elif situation == "evaluate":
            print(f"\n  {warn('Position evaluated')}")
            passed += 0.5
        else:
            print(f"\n  {warn('Evaluation differs from expected')}")
            passed += 0.5

        results.add_diagnostic("repetition", f"{test['name']}_value", float(value))

    score: float = passed / total
    results.add_diagnostic("repetition", "total_tested", total)
    results.add_diagnostic("repetition", "correct", passed)
    results.add("Repetition Awareness", score >= 0.4, score, 1.0)

    return score
