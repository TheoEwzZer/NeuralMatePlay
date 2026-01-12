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
        # === PAWN ATTACKS (4 positions) ===
        {
            "name": "Knight attacked by pawn",
            "fen": "rnbqkbnr/ppp1pppp/8/3p4/4N3/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1",
            "threatened_square": chess.E4,
            "threatened_piece": chess.KNIGHT,
            "attacker_square": chess.D5,
        },
        {
            "name": "Bishop attacked by pawn",
            "fen": "rnbqkbnr/ppp1pppp/8/3p4/2B5/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1",
            "threatened_square": chess.C4,
            "threatened_piece": chess.BISHOP,
            "attacker_square": chess.D5,
        },
        {
            "name": "Rook attacked by pawn",
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/3R4/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1",
            "threatened_square": chess.D4,
            "threatened_piece": chess.ROOK,
            "attacker_square": chess.E5,
        },
        {
            "name": "Queen attacked by pawn",
            "fen": "rnbqkbnr/pp1ppppp/8/2p5/3Q4/8/PPP1PPPP/RNB1KBNR w KQkq - 0 1",
            "threatened_square": chess.D4,
            "threatened_piece": chess.QUEEN,
            "attacker_square": chess.C5,
        },
        # === KNIGHT ATTACKS (4 positions) ===
        {
            "name": "Knight attacked by knight",
            "fen": "r1bqkbnr/pppppppp/2n5/8/3N4/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1",
            "threatened_square": chess.D4,
            "threatened_piece": chess.KNIGHT,
            "attacker_square": chess.C6,
        },
        {
            "name": "Bishop attacked by knight",
            "fen": "rnbqkb1r/pppppppp/5n2/8/4B3/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1",
            "threatened_square": chess.E4,
            "threatened_piece": chess.BISHOP,
            "attacker_square": chess.F6,
        },
        {
            "name": "Rook attacked by knight",
            "fen": "r1bqkbnr/pppppppp/2n5/4R3/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1",
            "threatened_square": chess.E5,
            "threatened_piece": chess.ROOK,
            "attacker_square": chess.C6,
        },
        {
            "name": "Queen attacked by knight",
            "fen": "rnbqkb1r/pppppppp/5n2/8/6Q1/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
            "threatened_square": chess.G4,
            "threatened_piece": chess.QUEEN,
            "attacker_square": chess.F6,
        },
        # === BISHOP ATTACKS (3 positions) ===
        {
            "name": "Knight attacked by bishop",
            "fen": "rnbqk1nr/pppppppp/8/7b/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",
            "threatened_square": chess.F3,
            "threatened_piece": chess.KNIGHT,
            "attacker_square": chess.H5,
        },
        {
            "name": "Rook attacked by bishop",
            "fen": "rnbqkbnr/pppppppp/8/4R3/8/6b1/PPPPPPPP/RNBQKBN1 w Qkq - 0 1",
            "threatened_square": chess.E5,
            "threatened_piece": chess.ROOK,
            "attacker_square": chess.G3,
        },
        {
            "name": "Queen attacked by bishop",
            "fen": "rnbqkbnr/pppppppp/8/5b2/8/3Q4/PPP1PPPP/RNB1KBNR w KQkq - 0 1",
            "threatened_square": chess.D3,
            "threatened_piece": chess.QUEEN,
            "attacker_square": chess.F5,
        },
        # === ROOK ATTACKS (2 positions) ===
        {
            "name": "Knight attacked by rook",
            "fen": "rnbqkbnr/pppppppp/8/8/4N2r/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1",
            "threatened_square": chess.E4,
            "threatened_piece": chess.KNIGHT,
            "attacker_square": chess.H4,
        },
        {
            "name": "Bishop attacked by rook",
            "fen": "rnbqkbnr/pppppppp/8/4B2r/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1",
            "threatened_square": chess.E5,
            "threatened_piece": chess.BISHOP,
            "attacker_square": chess.H5,
        },
        # === QUEEN ATTACKS (2 positions) ===
        {
            "name": "Knight attacked by queen",
            "fen": "rnb1kbnr/pppppppp/8/8/4Nq2/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1",
            "threatened_square": chess.E4,
            "threatened_piece": chess.KNIGHT,
            "attacker_square": chess.F4,
        },
        {
            "name": "Bishop attacked by queen",
            "fen": "rnb1kbnr/pppppppp/8/4Bq2/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1",
            "threatened_square": chess.E5,
            "threatened_piece": chess.BISHOP,
            "attacker_square": chess.F5,
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

        # Progressive scoring based on rank
        if top_move and top_move in good_moves:
            print(f"\n  {ok(f'Network defends: {top_move.uci()}')}")
            passed += 1.0
        elif defense_found_at == 2:
            print(f"\n  {warn(f'Defense at rank 2')}")
            passed += 0.7
        elif defense_found_at == 3:
            print(f"\n  {warn(f'Defense at rank 3')}")
            passed += 0.5
        elif defense_found_at:
            print(f"\n  {warn(f'Defense at rank {defense_found_at}')}")
            passed += 0.3
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
