"""Test: Mate in 1."""

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


def test_mate_in_one(network, results: TestResults):
    """Test if the network can find mate in 1."""
    print(header("TEST: Mate in 1"))

    test_positions = [
        {
            "name": "Back Rank Mate",
            "fen": "6k1/5ppp/8/8/8/8/8/4K2R w K - 0 1",
            "description": "Rh8#",
        },
        {
            "name": "Queen Ladder Mate",
            "fen": "k7/8/1K6/8/8/8/8/7Q w - - 0 1",
            "description": "Qa8#",
        },
        {
            "name": "Support Mate",
            "fen": "6k1/5Npp/8/8/8/8/8/4K2Q w - - 0 1",
            "description": "Qh8#",
        },
        {
            "name": "Arabian Mate",
            "fen": "7k/5N1p/8/6R1/8/8/8/4K3 w - - 0 1",
            "description": "Rg8#",
        },
    ]

    passed = 0
    total_valid = 0

    for test in test_positions:
        board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        # Find all mate-in-1 moves
        mate_moves = []
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                mate_moves.append(move)
            board.pop()

        if not mate_moves:
            print(f"  {dim('(No mate in 1 in this position, skipping)')}")
            results.add_diagnostic("mate_in_1", f"{test['name']}_skipped", True)
            continue

        total_valid += 1
        print(f"\n  Mating moves: {[m.uci() for m in mate_moves]}")

        # Get network's prediction (with proper perspective handling)
        policy, value = predict_for_board(board, network)

        top_idx = np.argmax(policy)
        top_move = decode_move(top_idx, board)
        top_prob = policy[top_idx]

        print(f"  Value evaluation: {value:+.4f}")

        # Check all top moves
        top_5 = np.argsort(policy)[::-1][:5]
        mate_found_at = None

        print(f"\n  {'Rank':<6} {'Move':<8} {'Prob':>8} {'Mate?':<10}")
        print("  " + "-" * 35)

        for i, idx in enumerate(top_5):
            move = decode_move(idx, board)
            prob = policy[idx]
            if move:
                is_mate = move in mate_moves
                if is_mate and mate_found_at is None:
                    mate_found_at = i + 1
                mate_str = "MATE!" if is_mate else ""
                color = Colors.GREEN if is_mate else ""
                end_color = Colors.ENDC if is_mate else ""
                print(
                    f"  {color}{i+1:<6} {move.uci():<8} {prob*100:>7.2f}% {mate_str}{end_color}"
                )

        # Store diagnostics
        results.add_diagnostic("mate_in_1", f"{test['name']}_mate_rank", mate_found_at)
        results.add_diagnostic("mate_in_1", f"{test['name']}_top_prob", float(top_prob))
        results.add_diagnostic("mate_in_1", f"{test['name']}_value", float(value))

        if top_move and top_move in mate_moves:
            print(
                f"\n  {ok(f'Network finds mate: {top_move.uci()} ({top_prob*100:.1f}%)')}"
            )
            passed += 1
        else:
            if mate_found_at:
                print(
                    f"\n  {warn(f'Mate found at rank {mate_found_at}, not prioritized')}"
                )
                results.add_issue(
                    "HIGH",
                    "tactics",
                    f"Network sees {test['name']} but doesn't prioritize it",
                    f"Mate at rank {mate_found_at}",
                )
            else:
                print(f"\n  {fail('Network misses the mate completely')}")
                results.add_issue(
                    "CRITICAL",
                    "tactics",
                    f"Network completely misses {test['name']}",
                    "Mate in 1 not in top 5 moves",
                )

    if total_valid == 0:
        print(warn("No valid mate-in-1 positions tested"))
        results.add("Mate in 1", False, 0.0, 1.0)
        return 0.0

    score = passed / total_valid
    results.add_diagnostic("mate_in_1", "total_tested", total_valid)
    results.add_diagnostic("mate_in_1", "mates_found", passed)
    results.add("Mate in 1", passed >= total_valid * 0.5, score, 1.0)

    if score < 0.5:
        results.add_recommendation(
            2,
            "Train on tactical puzzles with mate patterns",
            f"Network only finds {passed}/{total_valid} mate-in-1 positions",
        )

    return score
