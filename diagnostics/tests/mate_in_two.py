"""Test: Mate in 2."""

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


# Test positions with mate in 2 (17 positions like mate_in_one)
TEST_POSITIONS: list[dict[str, Any]] = [
    # === QUEEN SACRIFICE MATES (4 positions) ===
    {
        "name": "Scholar's Mate Pattern",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
        "first_move": "h5f7",
        "description": "Qxf7+ Ke7, Qxe5#",
    },
    {
        "name": "Queen Sacrifice Back Rank",
        "fen": "r1b2rk1/ppppqppp/2n5/8/8/8/PPPP1PPP/R1BQR1K1 w - - 0 1",
        "first_move": "e1e7",
        "description": "Rxe7 Rxe7, Qd8#",
    },
    {
        "name": "Queen Corridor Setup",
        "fen": "6k1/5ppp/8/8/8/5Q2/5PPP/6K1 w - - 0 1",
        "first_move": "f3f6",
        "description": "Qf6 gxf6, Qg7#",
    },
    {
        "name": "Queen and Bishop Mate",
        "fen": "r1bqk2r/ppp2ppp/2n5/3np3/2B5/5Q2/PPPP1PPP/RNB1K2R w KQkq - 0 1",
        "first_move": "f3f7",
        "description": "Qxf7+ Kd8, Qf8#",
    },
    # === BACK RANK MATES (4 positions) ===
    {
        "name": "Double Rook Back Rank",
        "fen": "6k1/5ppp/8/8/8/8/5PPP/R3R1K1 w - - 0 1",
        "first_move": "e1e8",
        "description": "Re8+ Rxe8, Rxe8#",
    },
    {
        "name": "Rook and Queen Back Rank",
        "fen": "3r2k1/5ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1",
        "first_move": "d1d8",
        "description": "Qxd8+ Rxd8, Rxd8#",
    },
    {
        "name": "Heavy Pieces Back Rank",
        "fen": "2r3k1/5ppp/8/8/8/8/5PPP/R2Q2K1 w - - 0 1",
        "first_move": "d1d8",
        "description": "Qd8+ Rxd8, Rxd8#",
    },
    {
        "name": "Rook Lift Back Rank",
        "fen": "6k1/5ppp/4R3/8/8/8/5PPP/4R1K1 w - - 0 1",
        "first_move": "e6e8",
        "description": "Re8+ Rxe8, Rxe8#",
    },
    # === KNIGHT MATES (3 positions) ===
    {
        "name": "Smothered Mate Classic",
        "fen": "6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1",
        "first_move": "g5f7",
        "description": "Nf7+ Kg8, Nh6#",
    },
    {
        "name": "Knight Fork to Mate",
        "fen": "r3k2r/ppp2ppp/8/4N3/8/8/PPP2PPP/R3K2R w KQkq - 0 1",
        "first_move": "e5f7",
        "description": "Nf7 Kd7, Nxh8#",
    },
    {
        "name": "Double Knight Mate",
        "fen": "4k3/8/5N2/4N3/8/8/8/4K3 w - - 0 1",
        "first_move": "f6d7",
        "description": "Nd7 Kf7, Ne5#",
    },
    # === BISHOP MATES (2 positions) ===
    {
        "name": "Bishop and Rook Mate",
        "fen": "4k3/8/4B3/8/8/8/8/4RK2 w - - 0 1",
        "first_move": "e1e8",
        "description": "Re8+ Kd7, Rd8#",
    },
    {
        "name": "Two Bishops Mate",
        "fen": "4k3/8/3BB3/8/8/8/8/4K3 w - - 0 1",
        "first_move": "d6c7",
        "description": "Bc7 Kf7, Bd5#",
    },
    # === SPECIAL PATTERNS (4 positions) ===
    {
        "name": "Arabian Mate Setup",
        "fen": "7k/8/5N1R/8/8/8/8/6K1 w - - 0 1",
        "first_move": "h6h7",
        "description": "Rh7+ Kg8, Rg7#",
    },
    {
        "name": "Anastasia's Mate Setup",
        "fen": "4rrk1/5Npp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        "first_move": "a1a8",
        "description": "Ra8 Rxa8, Nh6#",
    },
    {
        "name": "Opera Mate Setup",
        "fen": "3rkb1r/ppp2ppp/8/8/8/8/PPP2PPP/R1B1R1K1 w - - 0 1",
        "first_move": "e1e8",
        "description": "Re8+ Bxe8, Rxe8#",
    },
    {
        "name": "Boden's Mate Setup",
        "fen": "2kr4/ppp5/8/8/2B5/1B6/PPP5/2K5 w - - 0 1",
        "first_move": "c4a6",
        "description": "Ba6+ bxa6, Bxa6#",
    },
]


def _is_mate_in_n(
    board: chess.Board, n: int, memo: dict[str, chess.Move | None] | None = None
) -> chess.Move | None:
    """Check if there's a forced mate in n moves. Returns the first move if found."""
    if memo is None:
        memo = {}

    board_key: str = board.fen()
    if board_key in memo:
        return memo[board_key]

    if n <= 0:
        memo[board_key] = None
        return None

    for move in board.legal_moves:
        board.push(move)

        if board.is_checkmate():
            board.pop()
            memo[board_key] = move
            return move

        if n > 1 and not board.is_game_over():
            # Opponent's best defense
            can_escape: bool = False
            for defense in board.legal_moves:
                board.push(defense)
                if not _is_mate_in_n(board, n - 1, memo):
                    can_escape = True
                board.pop()
                if can_escape:
                    break

            if not can_escape:
                board.pop()
                memo[board_key] = move
                return move

        board.pop()

    memo[board_key] = None
    return None


def test_mate_in_two(network: DualHeadNetwork, results: TestResults) -> float:
    """Test if the network can find mate in 2."""
    print(header("TEST: Mate in 2"))

    passed: float = 0
    total_valid: int = 0

    for test in TEST_POSITIONS:
        board: chess.Board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        # Verify there's actually a mate in 2
        mating_move: chess.Move | None = _is_mate_in_n(board, 2)

        if not mating_move and test["first_move"]:
            # Try the suggested first move
            try:
                suggested: chess.Move = chess.Move.from_uci(test["first_move"])
                if suggested in board.legal_moves:
                    print(f"\n  Expected first move: {test['first_move']}")
                else:
                    print(f"\n  (Position for evaluation)")
                    continue
            except:
                continue

        if mating_move:
            total_valid += 1
            print(f"\n  Mating sequence starts with: {mating_move.uci()}")

            policy: np.ndarray
            value: float
            policy, value = predict_for_board(board, network)

            top_5: np.ndarray = np.argsort(policy)[::-1][:5]

            print(f"  Value: {value:+.4f}")
            print(f"\n  {'Rank':<6} {'Move':<8} {'Prob':>8} {'Mate?':<10}")
            print("  " + "-" * 35)

            mate_found_at: int | None = None
            for i, idx in enumerate(top_5):
                move = decode_move(idx, board)
                prob = policy[idx]
                if move:
                    is_mate_move = move == mating_move
                    if is_mate_move and mate_found_at is None:
                        mate_found_at = i + 1
                    mate_str = "MATE!" if is_mate_move else ""
                    color = Colors.GREEN if is_mate_move else ""
                    end_color = Colors.ENDC if color else ""
                    print(
                        f"  {color}{i+1:<6} {move.uci():<8} {prob*100:>7.2f}% {mate_str}{end_color}"
                    )

            # Progressive scoring based on rank
            if mate_found_at == 1:
                print(f"\n  {ok('Network finds the mating move!')}")
                passed += 1.0
            elif mate_found_at == 2:
                print(f"\n  {warn(f'Mating move at rank 2')}")
                passed += 0.75
            elif mate_found_at == 3:
                print(f"\n  {warn(f'Mating move at rank 3')}")
                passed += 0.5
            elif mate_found_at:
                print(f"\n  {warn(f'Mating move at rank {mate_found_at}')}")
                passed += 0.25
            else:
                print(f"\n  {fail('Network misses the mate in 2')}")

            results.add_diagnostic("mate_in_2", f"{test['name']}_rank", mate_found_at)
        else:
            # Evaluation position
            policy: np.ndarray
            value: float
            policy, value = predict_for_board(board, network)
            print(f"\n  Value evaluation: {value:+.4f}")

    if total_valid == 0:
        print(warn("No valid mate-in-2 positions tested"))
        results.add("Mate in 2", False, 0.0, 1.0)
        return 0.0

    score: float = passed / total_valid
    results.add_diagnostic("mate_in_2", "total_tested", total_valid)
    results.add_diagnostic("mate_in_2", "found", passed)
    results.add("Mate in 2", score >= 0.3, score, 1.0)

    return score
