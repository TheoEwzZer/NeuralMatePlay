"""Test: King Safety."""

import numpy as np
import chess

from alphazero.move_encoding import decode_move
from ..core import (
    TestResults,
    header,
    subheader,
    ok,
    fail,
    warn,
    predict_for_board,
)


# Test positions for king safety evaluation
TEST_POSITIONS = [
    # === SAFE VS UNSAFE KING ===
    {
        "name": "Castled King (Safe)",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 0 1",
        "expected_eval": "positive",  # White castled, safer
        "description": "White has castled, Black hasn't",
    },
    {
        "name": "Exposed King (Unsafe)",
        "fen": "rnbq1bnr/ppppkppp/8/4p3/4P3/8/PPPPKPPP/RNBQ1BNR w - - 0 1",
        "expected_eval": "neutral",  # Both kings exposed
        "description": "Both kings in center - dangerous",
    },
    {
        "name": "King in Center vs Castled",
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 0 1",
        "expected_eval": "positive",  # White advantage (castled)
        "description": "White castled, Black king stuck",
    },
    # === SHOULD CASTLE ===
    {
        "name": "Must Castle (Kingside)",
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "expected_moves": ["e1g1"],  # O-O
        "description": "White should castle kingside",
        "test_type": "move",
    },
    {
        "name": "Must Castle (Queenside)",
        "fen": "r3kbnr/pppqpppp/2n5/3p4/3P1B2/2N5/PPP1PPPP/R3KBNR w KQkq - 0 1",
        "expected_moves": ["e1c1"],  # O-O-O
        "description": "Queenside castle is possible",
        "test_type": "move",
    },
    # === PAWN SHIELD ===
    {
        "name": "Intact Pawn Shield",
        "fen": "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w - - 0 1",
        "expected_eval": "neutral",  # Both have good shields
        "description": "Both kings have intact pawn shields",
    },
    {
        "name": "Broken Pawn Shield",
        "fen": "r1bq1rk1/pppp1p1p/2n2np1/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w - - 0 1",
        "expected_eval": "positive",  # Black's shield weakened (g6)
        "description": "Black has weakened kingside (g6)",
    },
    # === ATTACK ON KING ===
    {
        "name": "King Under Attack",
        "fen": "r1bq1rk1/pppp1ppp/2n2n2/4p3/2B1P2Q/5N2/PPPP1PPP/RNB2RK1 b - - 0 1",
        "expected_eval": "negative",  # Black under pressure
        "description": "White has attacking chances with Qh4",
    },
    {
        "name": "Greek Gift Setup",
        "fen": "r1bq1rk1/ppp2ppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 1",
        "expected_eval": "neutral",
        "description": "Potential Bxh7+ sacrifice position",
    },
    {
        "name": "Open File Toward King",
        "fen": "r1bq1rk1/ppp2ppp/2n1pn2/3p4/3P4/2NBPN2/PPP2PPP/R1BQK2R w KQ - 0 1",
        "expected_eval": "positive",  # White has initiative
        "description": "White can open lines toward Black's king",
    },
]


def test_king_safety(network, results: TestResults):
    """Test if network evaluates king safety correctly."""
    print(header("TEST: King Safety"))

    passed = 0
    total = 0

    for test in TEST_POSITIONS:
        board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        policy, value = predict_for_board(board, network)

        # Check if it's a move test or evaluation test
        if "expected_moves" in test:
            total += 1
            top_5 = np.argsort(policy)[::-1][:5]

            print(f"\n  Value: {value:+.4f}")
            print(f"  Expected: {test['expected_moves']}")

            found_at = None
            expected_set = set(test["expected_moves"])

            for i, idx in enumerate(top_5):
                move = decode_move(idx, board)
                if move and move.uci() in expected_set:
                    found_at = i + 1
                    break

            if found_at == 1:
                print(f"  {ok('Correct castle/safety move!')}")
                passed += 1
            elif found_at:
                print(f"  {warn(f'Safety move at rank {found_at}')}")
                passed += 0.5
            else:
                print(f"  {fail('Misses king safety move')}")

            results.add_diagnostic("king_safety", f"{test['name']}_rank", found_at)

        else:
            # Evaluation test - progressive scoring
            total += 1
            expected = test["expected_eval"]

            print(f"\n  Value: {value:+.4f} (expected: {expected})")

            if expected == "positive":
                if value > 0.1:
                    print(f"  {ok('Correct positive evaluation')}")
                    passed += 1.0
                elif value > 0.0:
                    print(f"  {warn('Weak positive')}")
                    passed += 0.7
                elif value > -0.15:
                    print(f"  {warn('Near neutral')}")
                    passed += 0.3
                else:
                    print(f"  {fail(f'Wrong: got {value:+.4f}')}")
            elif expected == "negative":
                if value < -0.1:
                    print(f"  {ok('Correct negative evaluation')}")
                    passed += 1.0
                elif value < 0.0:
                    print(f"  {warn('Weak negative')}")
                    passed += 0.7
                elif value < 0.15:
                    print(f"  {warn('Near neutral')}")
                    passed += 0.3
                else:
                    print(f"  {fail(f'Wrong: got {value:+.4f}')}")
            else:  # neutral
                if abs(value) < 0.15:
                    print(f"  {ok('Correct neutral evaluation')}")
                    passed += 1.0
                elif abs(value) < 0.3:
                    print(f"  {warn('Slightly off neutral')}")
                    passed += 0.7
                else:
                    print(f"  {fail(f'Wrong: got {value:+.4f}')}")

            results.add_diagnostic("king_safety", f"{test['name']}_value", float(value))

    score = passed / total if total > 0 else 0
    results.add_diagnostic("king_safety", "total_tested", total)
    results.add_diagnostic("king_safety", "correct", passed)
    results.add("King Safety", score >= 0.5, score, 1.0)

    return score
