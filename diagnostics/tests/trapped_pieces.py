"""Test: Trapped Pieces Recognition."""

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


# Test positions with trapped pieces
TEST_POSITIONS = [
    # === TRAPPED BISHOP ===
    {
        "name": "Trapped Bishop h7",
        "fen": "r1bqk2r/ppppnppp/2n5/4p3/2B5/b4N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "expected_eval": "positive",  # Black's bishop on a3 is trapped
        "description": "Black bishop trapped on a3",
        "trap_type": "evaluation",
    },
    {
        "name": "Bishop Trapped on h7",
        "fen": "rnbqk2r/ppppp1pp/5n2/5p2/2B5/4PN2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "expected_moves": ["c4f7"],  # Can trap bishop on f7 sac
        "description": "Bxf7+ wins material",
        "trap_type": "move",
    },
    # === TRAPPED KNIGHT ===
    {
        "name": "Knight Trapped on Rim",
        "fen": "rnbqkb1r/pppppppp/8/8/8/N7/PPPPPPPP/R1BQKBNR w KQkq - 0 1",
        "expected_eval": "neutral",  # Na3 is awkward but not trapped
        "description": "Knight on a3 is misplaced",
        "trap_type": "evaluation",
    },
    {
        "name": "Trap Enemy Knight",
        "fen": "rnbqkb1r/ppp1pppp/3p4/8/3nP3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 1",
        "expected_moves": ["c3d5", "f3d4"],  # Win the knight
        "description": "Can win the trapped knight on d4",
        "trap_type": "move",
    },
    # === TRAPPED ROOK ===
    {
        "name": "Rook Trapped in Corner",
        "fen": "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K1NR w KQkq - 0 1",
        "expected_eval": "negative",  # White rook trapped by own knight
        "description": "White's h1 rook blocked by knight",
        "trap_type": "evaluation",
    },
    {
        "name": "Trap Rook with Bishop",
        "fen": "r4rk1/ppp2ppp/3p4/8/8/2B5/PPP2PPP/R4RK1 w - - 0 1",
        "expected_moves": ["c3a5", "c3f6"],  # Active bishop
        "description": "Bishop can restrict rook",
        "trap_type": "move",
    },
    # === TRAPPED QUEEN ===
    {
        "name": "Queen Trap",
        "fen": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1",
        "expected_moves": ["g4g5"],  # Traps queen
        "description": "g5 traps the queen",
        "trap_type": "move",
    },
    {
        "name": "Avoid Queen Trap",
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        "expected_eval": "neutral",  # Normal position
        "description": "Standard opening - no traps",
        "trap_type": "evaluation",
    },
    # === GENERAL PIECE RESTRICTION ===
    {
        "name": "Pieces Restricted",
        "fen": "r1bqkbnr/pppppppp/2n5/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        "expected_eval": "positive",  # White slightly better developed
        "description": "Normal development - slight White edge",
        "trap_type": "evaluation",
    },
    {
        "name": "Cramped Position",
        "fen": "r1bqkbnr/pppppppp/2n5/8/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
        "expected_eval": "negative",  # Black is slightly cramped
        "description": "Black has less space",
        "trap_type": "evaluation",
    },
]


def test_trapped_pieces(network, results: TestResults):
    """Test if network recognizes trapped pieces."""
    print(header("TEST: Trapped Pieces"))

    passed = 0
    total = 0

    for test in TEST_POSITIONS:
        board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        policy, value = predict_for_board(board, network)

        if test["trap_type"] == "move":
            total += 1
            top_5 = np.argsort(policy)[::-1][:5]
            expected_set = set(test["expected_moves"])

            print(f"\n  Value: {value:+.4f}")
            print(f"  {'Rank':<6} {'Move':<8} {'Prob':>8} {'Traps?':<10}")
            print("  " + "-" * 35)

            found_at = None
            for i, idx in enumerate(top_5):
                move = decode_move(idx, board)
                prob = policy[idx]
                if move:
                    uci = move.uci()
                    is_trap = uci in expected_set
                    if is_trap and found_at is None:
                        found_at = i + 1
                    trap_str = "TRAP!" if is_trap else ""
                    color = Colors.GREEN if is_trap else ""
                    end_color = Colors.ENDC if color else ""
                    print(f"  {color}{i+1:<6} {uci:<8} {prob*100:>7.2f}% {trap_str}{end_color}")

            if found_at == 1:
                print(f"\n  {ok('Finds the trapping move!')}")
                passed += 1
            elif found_at:
                print(f"\n  {warn(f'Trapping move at rank {found_at}')}")
                passed += 0.5
            else:
                print(f"\n  {fail('Misses the trap')}")

            results.add_diagnostic("trapped_pieces", f"{test['name']}_rank", found_at)

        else:  # evaluation
            total += 1
            expected = test["expected_eval"]
            print(f"\n  Value: {value:+.4f} (expected: {expected})")

            if expected == "positive" and value > 0.0:
                print(f"  {ok('Correct evaluation')}")
                passed += 1
            elif expected == "negative" and value < 0.0:
                print(f"  {ok('Correct evaluation')}")
                passed += 1
            elif expected == "neutral" and abs(value) < 0.3:
                print(f"  {ok('Correct neutral evaluation')}")
                passed += 1
            else:
                print(f"  {warn(f'Evaluation mismatch')}")
                passed += 0.5

            results.add_diagnostic("trapped_pieces", f"{test['name']}_value", float(value))

    score = passed / total if total > 0 else 0
    results.add_diagnostic("trapped_pieces", "total_tested", total)
    results.add_diagnostic("trapped_pieces", "correct", passed)
    results.add("Trapped Pieces", score >= 0.4, score, 1.0)

    return score
