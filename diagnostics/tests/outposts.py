"""Test: Outpost Recognition."""

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


# Test positions for outpost understanding
TEST_POSITIONS = [
    # === KNIGHT OUTPOSTS ===
    {
        "name": "Classic d5 Outpost",
        "fen": "rnbqkb1r/pp3ppp/4pn2/2ppN3/3P4/2N5/PPP1PPPP/R1BQKB1R w KQkq - 0 1",
        "description": "Knight on d5 outpost",
        "expected_eval": "positive",
        "outpost_type": "knight_d5",
    },
    {
        "name": "e5 Outpost",
        "fen": "rnbqkb1r/ppp2ppp/4pn2/3pN3/3P4/8/PPP1PPPP/RNBQKB1R w KQkq - 0 1",
        "description": "Knight on e5",
        "expected_eval": "positive",
        "outpost_type": "knight_e5",
    },
    {
        "name": "Occupy Outpost",
        "fen": "rnbqkb1r/pp3ppp/4pn2/2pp4/3P4/2N2N2/PPP1PPPP/R1BQKB1R w KQkq - 0 1",
        "expected_moves": ["c3d5", "f3e5"],  # Jump to outpost
        "description": "Knight should occupy outpost",
        "outpost_type": "occupy",
    },
    # === BISHOP OUTPOSTS ===
    {
        "name": "Bishop Outpost",
        "fen": "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N1B3/PP2PPPP/R2QKBNR w KQkq - 0 1",
        "description": "Bishop on strong diagonal",
        "expected_eval": "neutral",  # Both have active bishops
        "outpost_type": "bishop",
    },
    # === WEAK SQUARES ===
    {
        "name": "Weak d6 Square",
        "fen": "rnbqkb1r/pp3ppp/3ppn2/2p5/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 1",
        "description": "d6 is weak - potential outpost",
        "expected_eval": "positive",
        "outpost_type": "weak_square",
    },
    {
        "name": "Weak f5 Square",
        "fen": "rnbqkb1r/pppp1p1p/5np1/4p3/2P1P3/2N5/PP1P1PPP/R1BQKBNR w KQkq - 0 1",
        "description": "f5 is weakened by g6",
        "expected_eval": "positive",
        "outpost_type": "weak_square",
    },
    # === EVALUATION POSITIONS ===
    {
        "name": "No Outposts",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "description": "Starting position - no outposts",
        "expected_eval": "neutral",
        "outpost_type": "none",
    },
    {
        "name": "Dominated Outpost",
        "fen": "r1bqkb1r/pp3ppp/2n1pn2/2ppN3/3P4/2N1P3/PPP2PPP/R1BQKB1R w KQkq - 0 1",
        "description": "Strong central knight",
        "expected_eval": "positive",
        "outpost_type": "dominant",
    },
]


def test_outposts(network, results: TestResults):
    """Test if network recognizes outposts."""
    print(header("TEST: Outpost Recognition"))

    passed = 0
    total = 0

    for test in TEST_POSITIONS:
        board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        policy, value = predict_for_board(board, network)

        outpost_type = test["outpost_type"]
        print(f"\n  Outpost type: {outpost_type}")

        if "expected_moves" in test:
            total += 1
            top_5 = np.argsort(policy)[::-1][:5]
            expected_set = set(test["expected_moves"])

            print(f"  Value: {value:+.4f}")
            print(f"  {'Rank':<6} {'Move':<8} {'Prob':>8} {'Outpost?':<12}")
            print("  " + "-" * 40)

            found_at = None
            for i, idx in enumerate(top_5):
                move = decode_move(idx, board)
                prob = policy[idx]
                if move:
                    uci = move.uci()
                    is_outpost = uci in expected_set
                    if is_outpost and found_at is None:
                        found_at = i + 1
                    out_str = "OUTPOST" if is_outpost else ""
                    color = Colors.GREEN if is_outpost else ""
                    end_color = Colors.ENDC if color else ""
                    print(f"  {color}{i+1:<6} {uci:<8} {prob*100:>7.2f}% {out_str}{end_color}")

            if found_at == 1:
                print(f"\n  {ok('Occupies outpost!')}")
                passed += 1
            elif found_at:
                print(f"\n  {warn(f'Outpost move at rank {found_at}')}")
                passed += 0.5
            else:
                print(f"\n  {fail('Misses outpost')}")

            results.add_diagnostic("outposts", f"{test['name']}_rank", found_at)

        else:
            total += 1
            expected = test["expected_eval"]
            print(f"  Value: {value:+.4f} (expected: {expected})")

            if expected == "positive" and value > 0.0:
                print(f"  {ok('Sees outpost advantage')}")
                passed += 1
            elif expected == "negative" and value < 0.0:
                print(f"  {ok('Sees outpost disadvantage')}")
                passed += 1
            elif expected == "neutral" and abs(value) < 0.3:
                print(f"  {ok('Correctly evaluates position')}")
                passed += 1
            else:
                print(f"  {warn('Evaluation differs')}")
                passed += 0.5

            results.add_diagnostic("outposts", f"{test['name']}_value", float(value))

    score = passed / total if total > 0 else 0
    results.add_diagnostic("outposts", "total_tested", total)
    results.add_diagnostic("outposts", "correct", passed)
    results.add("Outposts", score >= 0.4, score, 1.0)

    return score
