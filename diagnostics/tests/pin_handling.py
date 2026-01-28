"""Test: Pin Handling."""

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


# Test positions with pins
TEST_POSITIONS = [
    # === ABSOLUTE PINS (to king) ===
    {
        "name": "Pin Knight to King",
        "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        "expected_moves": ["f1b5"],  # Bb5 pins knight
        "description": "Bb5 pins knight to king",
        "pin_type": "create",
    },
    {
        "name": "Exploit Absolute Pin",
        "fen": "rnbqkb1r/pppp1ppp/8/1B2p3/4n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "expected_moves": ["d2d3"],  # Attack pinned piece
        "description": "Attack the pinned knight",
        "pin_type": "exploit",
    },
    {
        "name": "Pinned Piece Can't Move",
        "fen": "rnbqk2r/pppp1ppp/5n2/1Bb1p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
        "expected_eval": "negative",  # Black's knight is pinned
        "description": "Black knight pinned to king",
        "pin_type": "evaluation",
    },
    # === RELATIVE PINS (to queen) ===
    {
        "name": "Pin to Queen",
        "fen": "rnb1kb1r/ppppqppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "expected_moves": ["c1g5"],  # Bg5 pins knight to queen
        "description": "Bg5 pins knight to queen",
        "pin_type": "create",
    },
    {
        "name": "Exploit Queen Pin",
        "fen": "rnb1kb1r/ppppqppp/5n2/4p1B1/2B1P3/5N2/PPPP1PPP/RN1QK2R w KQkq - 0 1",
        "expected_moves": ["g5f6"],  # Capture pinned piece
        "description": "Bxf6 wins the pinned knight",
        "pin_type": "exploit",
    },
    # === BREAKING A PIN ===
    {
        "name": "Break Pin by Interposition",
        "fen": "rnbqk2r/pppp1ppp/4pn2/1Bb5/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
        "expected_moves": ["c7c6"],  # Break the pin
        "description": "c6 breaks the pin",
        "pin_type": "break",
    },
    {
        "name": "Break Pin by Attack",
        "fen": "rnbqk2r/pppp1ppp/4pn2/1Bb5/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
        "expected_moves": ["c7c6", "a7a6"],  # Attack or break
        "description": "a6 or c6 deals with pin",
        "pin_type": "break",
    },
    # === PIN EVALUATION ===
    {
        "name": "Strong Pin Advantage",
        "fen": "rnbqk2r/pppp1ppp/4p3/1Bb1P3/4n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "expected_eval": "positive",  # White has pin advantage
        "description": "White has strong pin on knight",
        "pin_type": "evaluation",
    },
    {
        "name": "Multiple Pins",
        "fen": "rn1qk2r/ppp2ppp/4pn2/1B1p4/1b1P4/2N1PN2/PPP2PPP/R1BQK2R w KQkq - 0 1",
        "expected_eval": "neutral",  # Both sides have pins
        "description": "Both sides have pins - balanced",
        "pin_type": "evaluation",
    },
    {
        "name": "Pin Wins Material",
        "fen": "rnbqk2r/pppp1ppp/5n2/1Bb1p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "expected_moves": ["e1g1", "d2d3"],  # Castle or attack pin
        "description": "Position with active pin",
        "pin_type": "exploit",
    },
]


def test_pin_handling(network, results: TestResults):
    """Test if network handles pins correctly."""
    print(header("TEST: Pin Handling"))

    passed = 0
    total = 0

    for test in TEST_POSITIONS:
        board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        policy, value = predict_for_board(board, network)

        if test["pin_type"] == "evaluation":
            total += 1
            expected = test["expected_eval"]
            print(f"\n  Value: {value:+.4f} (expected: {expected})")

            if expected == "positive" and value > 0.0:
                print(f"  {ok('Correct evaluation')}")
                passed += 1
            elif expected == "negative" and value < 0.0:
                print(f"  {ok('Correct evaluation')}")
                passed += 1
            elif expected == "neutral" and abs(value) < 0.25:
                print(f"  {ok('Correct neutral evaluation')}")
                passed += 1
            else:
                print(f"  {warn('Evaluation differs from expected')}")
                passed += 0.5

            results.add_diagnostic(
                "pin_handling", f"{test['name']}_value", float(value)
            )

        else:  # move test
            total += 1
            top_5 = np.argsort(policy)[::-1][:5]
            expected_set = set(test["expected_moves"])

            print(f"\n  Value: {value:+.4f}")
            print(f"  {'Rank':<6} {'Move':<8} {'Prob':>8} {'Pin?':<10}")
            print("  " + "-" * 35)

            found_at = None
            for i, idx in enumerate(top_5):
                move = decode_move(idx, board)
                prob = policy[idx]
                if move:
                    uci = move.uci()
                    is_pin = uci in expected_set
                    if is_pin and found_at is None:
                        found_at = i + 1
                    pin_str = "PIN!" if is_pin else ""
                    color = Colors.GREEN if is_pin else ""
                    end_color = Colors.ENDC if color else ""
                    print(
                        f"  {color}{i+1:<6} {uci:<8} {prob*100:>7.2f}% {pin_str}{end_color}"
                    )

            if found_at == 1:
                print(f"\n  {ok('Correct pin handling!')}")
                passed += 1
            elif found_at:
                print(f"\n  {warn(f'Pin move at rank {found_at}')}")
                passed += 0.5
            else:
                print(f"\n  {fail('Misses pin concept')}")

            results.add_diagnostic("pin_handling", f"{test['name']}_rank", found_at)

    score = passed / total if total > 0 else 0
    results.add_diagnostic("pin_handling", "total_tested", total)
    results.add_diagnostic("pin_handling", "correct", passed)
    results.add("Pin Handling", score >= 0.4, score, 1.0)

    return score
