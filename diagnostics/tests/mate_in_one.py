"""Test: Mate in 1."""

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
    dim,
    predict_for_board,
)

if TYPE_CHECKING:
    from src.alphazero.network import DualHeadNetwork


def test_mate_in_one(network: DualHeadNetwork, results: TestResults) -> float:
    """Test if the network can find mate in 1."""
    print(header("TEST: Mate in 1"))

    test_positions: list[dict[str, Any]] = [
        # === BACK RANK MATES (4 positions) ===
        {
            "name": "Back Rank Mate (Queen)",
            "fen": "6k1/5ppp/8/8/8/8/8/3QK2R w - - 0 1",
            "description": "Qd8#",
        },
        {
            "name": "Back Rank Mate (Rook)",
            "fen": "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1",
            "description": "Ra8#",
        },
        {
            "name": "Back Rank Mate (Rook 2)",
            "fen": "6k1/5ppp/8/8/8/8/4R3/4K3 w - - 0 1",
            "description": "Re8#",
        },
        {
            "name": "Back Rank Mate (Queen 2)",
            "fen": "5k2/5ppp/8/8/8/8/8/3QK3 w - - 0 1",
            "description": "Qd8#",
        },
        # === QUEEN MATES (6 positions) ===
        {
            "name": "Queen Ladder Mate",
            "fen": "k7/8/1K6/8/8/8/8/7Q w - - 0 1",
            "description": "Qh8# or Qb7#",
        },
        {
            "name": "King + Queen Mate",
            "fen": "k7/8/1K6/8/8/8/8/Q7 w - - 0 1",
            "description": "Qa7#",
        },
        {
            "name": "Queen Mate (edge)",
            "fen": "k7/8/K7/8/8/8/1Q6/8 w - - 0 1",
            "description": "Qb8#",
        },
        {
            "name": "Queen Corner Mate",
            "fen": "k7/1K6/8/8/8/8/8/Q7 w - - 0 1",
            "description": "Qc8#",
        },
        {
            "name": "Queen Support Mate",
            "fen": "6k1/8/5KQ1/8/8/8/8/8 w - - 0 1",
            "description": "Qg7#",
        },
        {
            "name": "Queen Diagonal Mate",
            "fen": "k7/8/1K6/8/8/8/8/4Q3 w - - 0 1",
            "description": "Qe8#",
        },
        # === ROOK MATES (3 positions) ===
        {
            "name": "Rook Corridor Mate",
            "fen": "3R3k/8/7K/8/8/8/8/8 w - - 0 1",
            "description": "Rf8#",
        },
        {
            "name": "Rook + King Mate",
            "fen": "k7/8/1K6/8/8/8/8/R7 w - - 0 1",
            "description": "Ra8#",
        },
        {
            "name": "Double Rook Mate",
            "fen": "6k1/5ppp/8/8/8/8/8/3RKR2 w - - 0 1",
            "description": "Rd8#",
        },
        # === SPECIAL MATES (4 positions) ===
        {
            "name": "Arabian Mate",
            "fen": "7k/7p/5N2/6R1/8/8/8/4K3 w - - 0 1",
            "description": "Rg8#",
        },
        {
            "name": "Smothered Mate",
            "fen": "r5rk/6pp/7N/8/8/8/8/4K3 w - - 0 1",
            "description": "Nf7#",
        },
        {
            "name": "Queen Corner Mate 2",
            "fen": "7k/5K2/8/8/8/8/8/7Q w - - 0 1",
            "description": "Qf8#",
        },
        {
            "name": "Queen Edge Mate",
            "fen": "7k/8/6K1/8/8/8/8/7Q w - - 0 1",
            "description": "Qh7#",
        },
    ]

    passed: float = 0
    total_valid: int = 0

    for test in test_positions:
        board: chess.Board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        # Find all mate-in-1 moves
        mate_moves: list[chess.Move] = []
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
        policy: np.ndarray
        value: float
        policy, value = predict_for_board(board, network)

        top_idx: int = np.argmax(policy)
        top_move: chess.Move | None = decode_move(top_idx, board)
        top_prob: float = policy[top_idx]

        print(f"  Value evaluation: {value:+.4f}")

        # Check all top moves
        top_5: np.ndarray = np.argsort(policy)[::-1][:5]
        mate_found_at: int | None = None

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

        # Progressive scoring based on rank
        if top_move and top_move in mate_moves:
            print(
                f"\n  {ok(f'Network finds mate: {top_move.uci()} ({top_prob*100:.1f}%)')}"
            )
            passed += 1.0
        elif mate_found_at == 2:
            print(f"\n  {warn(f'Mate at rank 2')}")
            passed += 0.7
            results.add_issue(
                "HIGH",
                "tactics",
                f"Network sees {test['name']} but doesn't prioritize it",
                f"Mate at rank {mate_found_at}",
            )
        elif mate_found_at == 3:
            print(f"\n  {warn(f'Mate at rank 3')}")
            passed += 0.5
            results.add_issue(
                "HIGH",
                "tactics",
                f"Network sees {test['name']} but doesn't prioritize it",
                f"Mate at rank {mate_found_at}",
            )
        elif mate_found_at:
            print(f"\n  {warn(f'Mate at rank {mate_found_at}')}")
            passed += 0.3
            results.add_issue(
                "HIGH",
                "tactics",
                f"Network sees {test['name']} but ranks it low",
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

    score: float = passed / total_valid
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
