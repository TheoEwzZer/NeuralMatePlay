"""Test: Free Piece Capture."""

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


def test_free_capture(network, results: TestResults):
    """Test if the network captures free pieces."""
    print(header("TEST: Free Piece Capture"))

    test_positions = [
        {
            "name": "Hanging Queen",
            "fen": "rnb1kbnr/pppppppp/8/8/3q4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
            "target_square": chess.D4,
            "target_piece": chess.QUEEN,
            "piece_value": 9,
        },
        {
            "name": "Hanging Rook",
            "fen": "rnbqkbnr/pppppppp/8/8/3r4/4P3/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            "target_square": chess.D4,
            "target_piece": chess.ROOK,
            "piece_value": 5,
        },
        {
            "name": "Hanging Bishop",
            "fen": "rnbqk1nr/pppppppp/8/8/1b6/2P5/PP1PPPPP/RNBQKBNR w KQkq - 0 1",
            "target_square": chess.B4,
            "target_piece": chess.BISHOP,
            "piece_value": 3,
        },
        {
            "name": "Hanging Knight",
            # Knight on e5, white knight on f3 can capture with Nxe5
            "fen": "rnbqkb1r/pppppppp/8/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
            "target_square": chess.E5,
            "target_piece": chess.KNIGHT,
            "piece_value": 3,
        },
    ]

    passed = 0
    total_details = []

    for test in test_positions:
        board = chess.Board(test["fen"])
        print(subheader(test["name"]))
        print(board)

        # Find captures of the target piece
        capture_moves = []
        for move in board.legal_moves:
            if move.to_square == test["target_square"] and board.is_capture(move):
                capture_moves.append(move)

        print(
            f"\n  Target: {chess.piece_name(test['target_piece'])} on {chess.square_name(test['target_square'])} (value: {test['piece_value']})"
        )
        print(f"  Capturing moves available: {[m.uci() for m in capture_moves]}")

        # Get network's prediction (with proper perspective handling)
        policy, value = predict_for_board(board, network)

        # Analyze policy distribution
        top_indices = np.argsort(policy)[::-1][:10]
        top_probs = [policy[idx] for idx in top_indices]

        print(f"\n  Value head evaluation: {value:+.4f}")
        with np.errstate(divide="ignore", invalid="ignore"):
            policy_safe = np.maximum(policy, 1e-10)
            entropy = -np.nansum(policy_safe * np.log(policy_safe))
        print(f"  Policy entropy: {entropy:.4f}")

        print(f"\n  {'Rank':<6} {'Move':<8} {'Prob':>8} {'Type':<15}")
        print("  " + "-" * 45)

        found_capture = False
        capture_rank = None
        capture_prob = None

        for i, idx in enumerate(top_indices):
            move = decode_move(idx, board)
            prob = policy[idx]
            if move:
                is_target_capture = move in capture_moves
                move_type = "*** CAPTURE ***" if is_target_capture else ""

                if is_target_capture:
                    if i == 0:
                        found_capture = True
                    if capture_rank is None:
                        capture_rank = i + 1
                        capture_prob = prob
                    color = Colors.GREEN if i == 0 else Colors.YELLOW
                    print(
                        f"  {color}{i+1:<6} {move.uci():<8} {prob*100:>7.2f}% {move_type}{Colors.ENDC}"
                    )
                else:
                    print(f"  {i+1:<6} {move.uci():<8} {prob*100:>7.2f}% {move_type}")

        # Detailed analysis
        detail = {
            "position": test["name"],
            "piece": chess.piece_name(test["target_piece"]),
            "value": test["piece_value"],
            "capture_rank": capture_rank,
            "capture_prob": capture_prob,
            "top_move_prob": top_probs[0] if top_probs else 0,
            "value_eval": value,
            "passed": found_capture,
        }
        total_details.append(detail)

        if found_capture:
            print(f"\n  {ok('Network prioritizes the capture')}")
            passed += 1
        else:
            if capture_rank:
                print(
                    f"\n  {warn(f'Capture found at rank {capture_rank} with {capture_prob*100:.1f}% prob')}"
                )
                results.add_issue(
                    "HIGH",
                    "policy",
                    f"Network sees {test['name']} capture but doesn't prioritize it",
                    f"Capture at rank {capture_rank} ({capture_prob*100:.1f}%) vs top move ({top_probs[0]*100:.1f}%)",
                )
            else:
                print(f"\n  {fail('Network does not see the capture in top 10')}")
                results.add_issue(
                    "CRITICAL",
                    "policy",
                    f"Network completely misses {test['name']} capture",
                    f"Free {chess.piece_name(test['target_piece'])} (value {test['piece_value']}) not in top 10 moves",
                )

    # Summary statistics
    print(subheader("Capture Test Summary"))

    capture_ranks = [d["capture_rank"] for d in total_details if d["capture_rank"]]
    capture_probs = [d["capture_prob"] for d in total_details if d["capture_prob"]]

    avg_capture_rank = np.mean(capture_ranks) if capture_ranks else float("nan")
    avg_capture_prob = np.mean(capture_probs) if capture_probs else 0.0

    print(f"  Positions tested:     {len(test_positions)}")
    print(f"  Captures prioritized: {passed}/{len(test_positions)}")
    print(f"  Average capture rank: {avg_capture_rank:.1f}")
    print(f"  Average capture prob: {avg_capture_prob*100:.1f}%")

    # Store diagnostics
    results.add_diagnostic("free_capture", "positions_tested", len(test_positions))
    results.add_diagnostic("free_capture", "captures_prioritized", passed)
    results.add_diagnostic("free_capture", "avg_capture_rank", avg_capture_rank)
    results.add_diagnostic("free_capture", "avg_capture_prob", avg_capture_prob)

    for d in total_details:
        results.add_diagnostic(
            "free_capture", f"{d['position']}_rank", d["capture_rank"]
        )
        results.add_diagnostic(
            "free_capture", f"{d['position']}_prob", d["capture_prob"]
        )
        results.add_diagnostic(
            "free_capture", f"{d['position']}_value_eval", d["value_eval"]
        )

    score = passed / len(test_positions)
    results.add("Free Capture", passed == len(test_positions), score, 1.0)

    if score < 1.0:
        results.add_recommendation(
            1,
            "Improve capture recognition training",
            f"Network only prioritizes {passed}/{len(test_positions)} free captures",
        )

    return score
