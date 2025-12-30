"""Test: Basic Tactics."""

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
    info,
    predict_for_board,
)


def test_tactics(network, results: TestResults):
    """Test if the network can find basic tactics."""
    print(header("TEST: Basic Tactics"))

    test_positions = [
        {
            "name": "Knight Fork Setup",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
            "tactical_move": "f3g5",
            "description": "Ng5 attacks f7 (weak point)",
            "type": "attack",
        },
        {
            "name": "Discovered Attack",
            "fen": "r1bqkb1r/pppp1Bpp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
            "tactical_move": None,
            "description": "After Bxf7+ (Italian Game trap)",
            "type": "evaluation",
        },
        {
            "name": "Central Fork",
            "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4N3/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1",
            "tactical_move": "e4d6",
            "description": "Nd6+ forks king and bishop",
            "type": "fork",
        },
    ]

    passed = 0
    tactical_found = 0

    for test in test_positions:
        board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        policy, value = predict_for_board(board, network)

        top_5 = np.argsort(policy)[::-1][:5]

        print(f"\n  Value evaluation: {value:+.4f}")
        print(f"\n  {'Rank':<6} {'Move':<8} {'Prob':>8} {'Check?':<8} {'Notes':<20}")
        print("  " + "-" * 55)

        tactical_rank = None

        for i, idx in enumerate(top_5):
            move = decode_move(idx, board)
            prob = policy[idx]
            if move:
                board.push(move)
                is_check = board.is_check()
                board.pop()

                check_str = "+" if is_check else ""
                notes = ""

                if test.get("tactical_move"):
                    if move.uci() == test["tactical_move"]:
                        notes = "TACTICAL!"
                        if tactical_rank is None:
                            tactical_rank = i + 1
                        if i == 0:
                            tactical_found += 1

                color = Colors.GREEN if notes else (Colors.YELLOW if is_check else "")
                end_color = Colors.ENDC if color else ""
                print(
                    f"  {color}{i+1:<6} {move.uci():<8} {prob*100:>7.2f}% {check_str:<8} {notes}{end_color}"
                )

        if test.get("tactical_move"):
            if tactical_rank == 1:
                print(f"\n  {ok('Tactical move found as top choice!')}")
                passed += 1
            elif tactical_rank:
                print(f"\n  {warn(f'Tactical move at rank {tactical_rank}')}")
                passed += 0.5
            else:
                print(f"\n  {fail('Tactical move not in top 5')}")
        else:
            # Evaluation-only position
            passed += 0.5
            print(f"\n  {info('Position for evaluation analysis')}")

        results.add_diagnostic(
            "tactics", f"{test['name']}_tactical_rank", tactical_rank
        )
        results.add_diagnostic("tactics", f"{test['name']}_value", float(value))

    score = passed / len(test_positions)
    results.add_diagnostic("tactics", "total_tested", len(test_positions))
    results.add_diagnostic("tactics", "tactical_found", tactical_found)
    results.add("Tactics", score >= 0.5, score, 1.0)

    if score < 0.5:
        results.add_recommendation(
            4,
            "Include tactical puzzles in training data",
            f"Network tactical score: {score*100:.0f}%",
        )

    return score
