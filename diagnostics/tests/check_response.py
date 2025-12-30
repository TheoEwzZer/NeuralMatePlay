"""Test: Check Response."""

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
    dim,
    predict_for_board,
)


def test_check_response(network, results: TestResults):
    """Test if the network properly responds to check."""
    print(header("TEST: Check Response"))

    test_positions = [
        {
            "name": "Queen Check - Block or Capture",
            "fen": "rnbqkbnr/ppppp1pp/8/5p2/4P2q/8/PPPP2PP/RNBQKBNR w KQkq - 0 1",
            "description": "Queen gives check on h4 via diagonal (e2-f2 empty), block with g3 or move king",
        },
        {
            "name": "Bishop Check - Must Respond",
            "fen": "rnbqk1nr/pppp1ppp/8/4p3/1b2P3/8/PPP2PPP/RNBQKBNR w KQkq - 0 1",
            "description": "Bishop gives check on b4 via diagonal (d2-e2 empty), block or move king",
        },
        {
            "name": "Knight Check - King Must Move",
            "fen": "rnbqkb1r/pppppppp/8/8/8/5n2/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "description": "Knight gives check from f3, only king moves work",
        },
    ]

    passed = 0
    total = 0

    for test in test_positions:
        board = chess.Board(test["fen"])

        if not board.is_check():
            name = test["name"]
            print(f"  {dim(f'Skipping {name}: not actually in check')}")
            continue

        total += 1
        print(subheader(test["name"]))
        print(board)
        print(f"\n  {test['description']}")

        # All legal moves escape check
        legal_moves = list(board.legal_moves)
        print(f"  Legal moves (all escape check): {len(legal_moves)}")

        policy, value = predict_for_board(board, network)

        top_idx = np.argmax(policy)
        top_move = decode_move(top_idx, board)
        top_prob = policy[top_idx]

        # Show top 5 moves
        top_5 = np.argsort(policy)[::-1][:5]
        legal_found_at = None

        print(f"\n  Value evaluation: {value:+.4f}")
        print(f"\n  {'Rank':<6} {'Move':<8} {'Prob':>8} {'Legal?':<10}")
        print("  " + "-" * 35)

        for i, idx in enumerate(top_5):
            move = decode_move(idx, board)
            prob = policy[idx]
            if move:
                is_legal = move in legal_moves
                if is_legal and legal_found_at is None:
                    legal_found_at = i + 1
                legal_str = "LEGAL" if is_legal else "ILLEGAL!"
                color = Colors.GREEN if is_legal else Colors.RED
                print(
                    f"  {color}{i+1:<6} {move.uci():<8} {prob*100:>7.2f}% {legal_str}{Colors.ENDC}"
                )

        results.add_diagnostic(
            "check_response", f"{test['name']}_legal_rank", legal_found_at
        )
        results.add_diagnostic(
            "check_response",
            f"{test['name']}_top_legal",
            top_move in legal_moves if top_move else False,
        )

        if top_move and top_move in legal_moves:
            print(f"\n  {ok(f'Network responds to check: {top_move.uci()}')}")
            passed += 1
        else:
            if legal_found_at:
                print(f"\n  {warn(f'Legal move at rank {legal_found_at}')}")
            else:
                print(f"\n  {fail('Network plays ILLEGAL move under check!')}")
            results.add_issue(
                "CRITICAL",
                "legality",
                f"Network plays illegal move when in check ({test['name']})",
                f"Top move {top_move.uci() if top_move else 'None'} is not legal",
            )

    if total == 0:
        print(warn("No valid check positions to test"))
        results.add("Check Response", False, 0.0, 1.0)
        return 0.0

    score = passed / total
    results.add_diagnostic("check_response", "total_tested", total)
    results.add_diagnostic("check_response", "correct_responses", passed)
    results.add("Check Response", passed == total, score, 1.0)
    return score
