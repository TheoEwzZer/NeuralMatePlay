"""Test: Check Response."""

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


def test_check_response(network: DualHeadNetwork, results: TestResults) -> float:
    """Test if the network properly responds to check."""
    print(header("TEST: Check Response"))

    test_positions: list[dict[str, Any]] = [
        # === BASIC CHECKS (4 positions) ===
        {
            "name": "Queen Check (diagonal)",
            "fen": "rnbqkbnr/ppppp1pp/8/5p2/4P2q/8/PPPP2PP/RNBQKBNR w KQkq - 0 1",
            "description": "Queen gives check on h4 via diagonal",
        },
        {
            "name": "Rook Check (file)",
            "fen": "4r3/8/8/8/4K3/8/8/8 w - - 0 1",
            "description": "Rook gives check on file",
        },
        {
            "name": "Knight Check",
            "fen": "8/8/8/3n4/8/4K3/8/8 w - - 0 1",
            "description": "Knight gives check from d5",
        },
        {
            "name": "Bishop Check",
            "fen": "rnbqk1nr/pppp1ppp/8/4p3/1b2P3/8/PPP2PPP/RNBQKBNR w KQkq - 0 1",
            "description": "Bishop gives check on b4 diagonal",
        },
        # === DOUBLE CHECKS - King MUST move (3 positions) ===
        {
            "name": "Double Check (B+N)",
            "fen": "r1bqkb1r/pppp1Bpp/2n2N2/4p3/8/8/PPPP1PPP/RNBQK2R b KQkq - 0 1",
            "description": "Double check from Bf7 and Nf6 - king must move",
        },
        {
            "name": "Double Check (R+B)",
            "fen": "4k3/8/3N4/7B/8/8/8/4K3 b - - 0 1",
            "description": "Double check from Nd6 and Bh5 - king must move",
        },
        {
            "name": "Double Check (R+N)",
            "fen": "4k3/8/3N4/8/4R3/8/8/4K3 b - - 0 1",
            "description": "Double check from Nd6 and Re4 on file",
        },
        # === DISCOVERED CHECKS (3 positions) ===
        {
            "name": "Discovered Check (Bishop reveals)",
            "fen": "r1bqk2r/pppp1Bpp/2n2n2/4p3/4P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 1",
            "description": "Bf7 gives check (classic fried liver setup)",
        },
        {
            "name": "Discovered Check (Rook reveals)",
            "fen": "4k3/8/8/8/4R3/8/8/4K3 b - - 0 1",
            "description": "Rook gives check on open file",
        },
        {
            "name": "Scholar's Mate Qh5+",
            "fen": "rnbqkbnr/pppp2pp/5p2/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 1",
            "description": "Qh5+ in Scholar's mate pattern - g6 or Ke7",
        },
        # === MUST BLOCK OR CAPTURE (3 positions) ===
        {
            "name": "Bishop Check (block options)",
            "fen": "rnbqk1nr/ppp2ppp/8/3pp3/B7/8/PPPP1PPP/RNBQK1NR b KQkq - 0 1",
            "description": "Ba4+ can be blocked by Bd7/Nc6/c6 or Ke7/Kf8",
        },
        {
            "name": "Knight Check (capture option)",
            "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/8/3n4/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            "description": "Nd3+ can be captured Bxd3/cxd3 or Ke2",
        },
        {
            "name": "Only One Legal Move",
            "fen": "4k3/4Q3/8/8/8/8/8/4K3 b - - 0 1",
            "description": "King must move - only escape Ke7",
        },
        # === COMPLEX CHECK POSITIONS (3 positions) ===
        {
            "name": "Knight Fork Check",
            "fen": "r1bqk2r/pppp1ppp/2n2N2/2b1p3/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 1",
            "description": "Nf6+ forks king and queen - must respond",
        },
        {
            "name": "Rook Check Endgame",
            "fen": "8/8/8/8/r7/8/8/K7 w - - 0 1",
            "description": "King must escape rook check",
        },
        {
            "name": "Queen Check (file)",
            "fen": "4k3/8/8/4Q3/8/8/8/4K3 b - - 0 1",
            "description": "Qe5+ on file - multiple king escapes",
        },
    ]

    passed: float = 0
    total: int = 0

    for test in test_positions:
        board: chess.Board = chess.Board(test["fen"])

        if not board.is_check():
            name: str = test["name"]
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

        # Progressive scoring based on rank of legal move
        if top_move and top_move in legal_moves:
            print(f"\n  {ok(f'Network responds to check: {top_move.uci()}')}")
            passed += 1.0
        elif legal_found_at == 2:
            print(f"\n  {warn(f'Legal move at rank 2')}")
            passed += 0.7
        elif legal_found_at == 3:
            print(f"\n  {warn(f'Legal move at rank 3')}")
            passed += 0.5
        elif legal_found_at:
            print(f"\n  {warn(f'Legal move at rank {legal_found_at}')}")
            passed += 0.3
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
