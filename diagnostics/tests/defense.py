"""Test: Defense (Avoid Hanging Pieces)."""

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


def test_defense(network, results: TestResults):
    """Test if the network avoids hanging its own pieces."""
    print(header("TEST: Defense (Avoid Hanging Pieces)"))

    test_positions = [
        {
            "name": "Knight under attack",
            "fen": "rnbqkbnr/ppp1pppp/8/3p4/4N3/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1",
            "threatened_square": chess.E4,
            "threatened_piece": chess.KNIGHT,
            "attacker_square": chess.D5,
        },
        {
            "name": "Bishop under attack",
            "fen": "rnbqkbnr/ppp1pppp/8/3pB3/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1",
            "threatened_square": chess.E5,
            "threatened_piece": chess.BISHOP,
            "attacker_square": chess.D5,
        },
        {
            "name": "Rook under attack",
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/7R/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1",
            "threatened_square": chess.H4,
            "threatened_piece": chess.ROOK,
            "attacker_square": chess.E5,
        },
    ]

    passed = 0

    for test in test_positions:
        board = chess.Board(test["fen"])
        print(subheader(test["name"]))
        print(board)

        piece_name = chess.piece_name(test["threatened_piece"])
        threat_sq = chess.square_name(test["threatened_square"])

        print(f"\n  Threatened: {piece_name} on {threat_sq}")
        print(f"  Attacker: pawn on {chess.square_name(test['attacker_square'])}")

        # Find good defensive moves
        good_moves = []
        for move in board.legal_moves:
            # Move the threatened piece away
            if move.from_square == test["threatened_square"]:
                good_moves.append(move)
            # Capture the attacker
            if move.to_square == test["attacker_square"] and board.is_capture(move):
                good_moves.append(move)

        print(f"  Good defensive moves: {[m.uci() for m in good_moves[:8]]}")

        # Get network's prediction (with proper perspective handling)
        policy, value = predict_for_board(board, network)

        top_idx = np.argmax(policy)
        top_move = decode_move(top_idx, board)
        top_prob = policy[top_idx]

        # Analyze top moves
        top_5 = np.argsort(policy)[::-1][:5]
        defense_found_at = None

        print(f"\n  Value evaluation: {value:+.4f}")
        print(f"\n  {'Rank':<6} {'Move':<8} {'Prob':>8} {'Defense?':<15}")
        print("  " + "-" * 40)

        for i, idx in enumerate(top_5):
            move = decode_move(idx, board)
            prob = policy[idx]
            if move:
                is_defense = move in good_moves
                if is_defense and defense_found_at is None:
                    defense_found_at = i + 1
                defense_str = "DEFENDS" if is_defense else ""
                color = Colors.GREEN if is_defense else ""
                end_color = Colors.ENDC if is_defense else ""
                print(
                    f"  {color}{i+1:<6} {move.uci():<8} {prob*100:>7.2f}% {defense_str}{end_color}"
                )

        # Store diagnostics
        results.add_diagnostic(
            "defense", f"{test['name']}_defense_rank", defense_found_at
        )
        results.add_diagnostic("defense", f"{test['name']}_top_prob", float(top_prob))

        if top_move and top_move in good_moves:
            print(f"\n  {ok(f'Network defends: {top_move.uci()}')}")
            passed += 1
        else:
            if defense_found_at:
                print(
                    f"\n  {warn(f'Defense at rank {defense_found_at}, not prioritized')}"
                )
            else:
                move = top_move.uci() if top_move else "None"
                print(f"\n  {fail(f'Network ignores the threat, plays {move}')}")
            results.add_issue(
                "HIGH",
                "defense",
                f"Network doesn't protect {piece_name} on {threat_sq}",
                f"Plays {move} instead of defending",
            )

    score = passed / len(test_positions)
    results.add_diagnostic("defense", "total_tested", len(test_positions))
    results.add_diagnostic("defense", "defenses_made", passed)
    results.add("Defense", passed == len(test_positions), score, 1.0)

    if score < 1.0:
        results.add_recommendation(
            3,
            "Add threat-awareness to training",
            f"Network defends only {passed}/{len(test_positions)} threatened pieces",
        )

    return score
