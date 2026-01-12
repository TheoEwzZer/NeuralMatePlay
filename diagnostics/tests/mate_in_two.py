"""Test: Mate in 2."""

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


# Test positions with mate in 2
TEST_POSITIONS = [
    # === QUEEN MATES ===
    {
        "name": "Queen Sacrifice Mate",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
        "first_move": "h5f7",  # Qxf7+
        "description": "Qxf7+ Kxf7, then mate follows",
    },
    {
        "name": "Back Rank Setup",
        "fen": "6k1/5ppp/8/8/8/8/5PPP/R3R1K1 w - - 0 1",
        "first_move": "e1e8",  # Re8+
        "description": "Re8+ Rxe8, Ra8#",
    },
    {
        "name": "Queen Corridor",
        "fen": "6k1/5ppp/8/8/8/5Q2/5PPP/6K1 w - - 0 1",
        "first_move": "f3f6",  # Qf6
        "description": "Qf6 threatens Qg7#",
    },
    # === ROOK MATES ===
    {
        "name": "Double Rook Mate",
        "fen": "6k1/5ppp/8/8/8/8/5PPP/R5RK w - - 0 1",
        "first_move": "a1a8",  # Ra8+
        "description": "Ra8+ Rxa8, Rg8#",
    },
    {
        "name": "Rook Lift",
        "fen": "6k1/5ppp/4R3/8/8/8/5PPP/6K1 w - - 0 1",
        "first_move": "e6e8",  # Re8+
        "description": "Re8+ Kh7, Rh8#",
    },
    # === KNIGHT MATES ===
    {
        "name": "Smothered Mate Setup",
        "fen": "r4rk1/5Npp/8/8/8/8/5PPP/6K1 w - - 0 1",
        "first_move": "f7h6",  # Nh6+
        "description": "Nh6+ Kh8, Qg8# (or similar)",
    },
    {
        "name": "Knight and Queen",
        "fen": "r1bqkbnr/pppp1Npp/8/8/8/8/PPPPQPPP/RNB1KB1R b KQkq - 0 1",
        "first_move": None,  # Black to move, evaluate position
        "description": "White has Nxh8 or Qh5+ threats",
    },
    # === BISHOP MATES ===
    {
        "name": "Bishop and Queen",
        "fen": "r1bqk2r/pppp1Bpp/2n2n2/2b1p3/4P3/3P4/PPP2PPP/RNBQK1NR b KQkq - 0 1",
        "first_move": None,  # Evaluation position
        "description": "After Bxf7+ Ke7",
    },
]


def _is_mate_in_n(board, n, memo=None):
    """Check if there's a forced mate in n moves. Returns the first move if found."""
    if memo is None:
        memo = {}

    board_key = board.fen()
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
            can_escape = False
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


def test_mate_in_two(network, results: TestResults):
    """Test if the network can find mate in 2."""
    print(header("TEST: Mate in 2"))

    passed = 0
    total_valid = 0

    for test in TEST_POSITIONS:
        board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        # Verify there's actually a mate in 2
        mating_move = _is_mate_in_n(board, 2)

        if not mating_move and test["first_move"]:
            # Try the suggested first move
            try:
                suggested = chess.Move.from_uci(test["first_move"])
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

            policy, value = predict_for_board(board, network)

            top_5 = np.argsort(policy)[::-1][:5]

            print(f"  Value: {value:+.4f}")
            print(f"\n  {'Rank':<6} {'Move':<8} {'Prob':>8} {'Mate?':<10}")
            print("  " + "-" * 35)

            mate_found_at = None
            for i, idx in enumerate(top_5):
                move = decode_move(idx, board)
                prob = policy[idx]
                if move:
                    is_mate_move = (move == mating_move)
                    if is_mate_move and mate_found_at is None:
                        mate_found_at = i + 1
                    mate_str = "MATE!" if is_mate_move else ""
                    color = Colors.GREEN if is_mate_move else ""
                    end_color = Colors.ENDC if color else ""
                    print(f"  {color}{i+1:<6} {move.uci():<8} {prob*100:>7.2f}% {mate_str}{end_color}")

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
            policy, value = predict_for_board(board, network)
            print(f"\n  Value evaluation: {value:+.4f}")

    if total_valid == 0:
        print(warn("No valid mate-in-2 positions tested"))
        results.add("Mate in 2", False, 0.0, 1.0)
        return 0.0

    score = passed / total_valid
    results.add_diagnostic("mate_in_2", "total_tested", total_valid)
    results.add_diagnostic("mate_in_2", "found", passed)
    results.add("Mate in 2", score >= 0.3, score, 1.0)

    return score
