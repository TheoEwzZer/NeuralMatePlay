"""Test: Passed Pawn Recognition."""

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


# Test positions with passed pawns
TEST_POSITIONS = [
    # === BASIC PUSH (3 positions) ===
    {
        "name": "Push Passed Pawn (7th rank)",
        "fen": "8/4P3/8/8/8/8/8/4K2k w - - 0 1",
        "expected_moves": ["e7e8q", "e7e8r"],
        "description": "Passed pawn on 7th - must promote",
        "test_type": "push",
    },
    {
        "name": "Push Passed Pawn (6th rank)",
        "fen": "8/8/4P3/8/8/8/8/4K2k w - - 0 1",
        "expected_moves": ["e6e7"],
        "description": "Passed pawn on 6th - push to 7th",
        "test_type": "push",
    },
    {
        "name": "Connected Passers",
        "fen": "8/8/3PP3/8/8/8/8/4K2k w - - 0 1",
        "expected_moves": ["d6d7", "e6e7"],
        "description": "Connected passed pawns - push either",
        "test_type": "push",
    },
    # === PAWN RACES (3 positions) - HARDER ===
    {
        "name": "Pawn Race (White wins)",
        "fen": "8/P7/8/8/8/8/7p/4K2k w - - 0 1",
        "expected_moves": ["a7a8q", "a7a8r"],
        "description": "White promotes first - must push!",
        "test_type": "race",
    },
    {
        "name": "Pawn Race (Must Calculate)",
        "fen": "8/8/P7/8/8/8/6p1/4K2k w - - 0 1",
        "expected_moves": ["a6a7"],
        "description": "Race is close - need to push",
        "test_type": "race",
    },
    {
        "name": "Pawn Race (Tempo Critical)",
        "fen": "8/1P4p1/8/8/8/8/8/4K2k w - - 0 1",
        "expected_moves": ["b7b8q", "b7b8r"],
        "description": "One tempo ahead - promote now",
        "test_type": "race",
    },
    # === BLOCKADE POSITIONS (3 positions) - HARDER ===
    {
        "name": "Blockade with Knight",
        "fen": "8/3p4/3N4/8/8/8/8/4K2k w - - 0 1",
        "expected_moves": ["d6e4", "d6f5", "d6b5", "d6c4"],  # Keep blockade or improve
        "description": "Knight blockades - don't release",
        "test_type": "blockade",
    },
    {
        "name": "Rook Behind Passer",
        "fen": "8/4P3/8/8/8/8/8/R3K2k w - - 0 1",
        "expected_moves": ["a1e1", "a1a8", "e7e8q"],  # Support or promote
        "description": "Rook should support from behind",
        "test_type": "support",
    },
    {
        "name": "Stop Outside Passer",
        "fen": "8/p7/8/8/8/8/8/4K2k w - - 0 1",
        "expected_moves": ["e1d2", "e1d1", "e1f2"],  # King must intercept
        "description": "King must stop distant passer",
        "test_type": "block",
    },
    # === SACRIFICE TO CREATE PASSER (3 positions) - HARDER ===
    {
        "name": "Breakthrough Sacrifice",
        "fen": "8/8/ppp5/PPP5/8/8/8/4K2k w - - 0 1",
        "expected_moves": ["b5b6", "a5a6", "c5c6"],  # Breakthrough
        "description": "Pawn breakthrough creates passer",
        "test_type": "sacrifice",
    },
    {
        "name": "Create Protected Passer",
        "fen": "8/8/8/1pP5/1P6/8/8/4K2k w - - 0 1",
        "expected_moves": ["c5c6"],  # Create protected passer
        "description": "c6 creates protected passed pawn",
        "test_type": "push",
    },
    {
        "name": "Candidate Pawns",
        "fen": "8/8/3p4/2pP4/2P5/8/8/4K2k w - - 0 1",
        "expected_moves": ["d5d6", "d5e6"],  # Push or capture
        "description": "Advance candidate passed pawn",
        "test_type": "push",
    },
    # === COMPLEX ENDGAMES (3 positions) - HARDER ===
    {
        "name": "King and Pawn vs King",
        "fen": "8/8/8/8/3k4/8/4P3/4K3 w - - 0 1",
        "expected_moves": ["e1d2", "e1f2", "e2e4"],  # Opposition/support
        "description": "Key squares concept",
        "test_type": "support",
    },
    {
        "name": "Wrong Rook Pawn",
        "fen": "8/8/8/8/8/7k/7P/5K2 w - - 0 1",
        "expected_moves": ["f1g1", "f1e2", "h2h4"],  # Try to win
        "description": "Rook pawn - tricky win",
        "test_type": "support",
    },
    {
        "name": "Distant Opposition",
        "fen": "8/8/8/k7/8/8/P7/K7 w - - 0 1",
        "expected_moves": ["a1b1", "a1b2"],  # Gain opposition
        "description": "Use distant opposition to win",
        "test_type": "support",
    },
]


def test_passed_pawn(network, results: TestResults):
    """Test if network recognizes and handles passed pawns."""
    print(header("TEST: Passed Pawn Recognition"))

    passed = 0
    total = len(TEST_POSITIONS)

    for test in TEST_POSITIONS:
        board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        policy, value = predict_for_board(board, network)

        top_5 = np.argsort(policy)[::-1][:5]

        print(f"\n  Value: {value:+.4f}")
        print(f"  {'Rank':<6} {'Move':<8} {'Prob':>8} {'Expected?':<12}")
        print("  " + "-" * 40)

        found_at = None
        expected_set = set(test["expected_moves"])

        for i, idx in enumerate(top_5):
            move = decode_move(idx, board)
            prob = policy[idx]
            if move:
                uci = move.uci()
                is_expected = uci in expected_set
                if is_expected and found_at is None:
                    found_at = i + 1
                exp_str = "YES" if is_expected else ""
                color = Colors.GREEN if is_expected else ""
                end_color = Colors.ENDC if color else ""
                print(
                    f"  {color}{i+1:<6} {uci:<8} {prob*100:>7.2f}% {exp_str}{end_color}"
                )

        # Progressive scoring based on rank
        if found_at == 1:
            print(f"\n  {ok('Correct pawn handling!')}")
            passed += 1.0
        elif found_at == 2:
            print(f"\n  {warn(f'Expected move at rank 2')}")
            passed += 0.75
        elif found_at == 3:
            print(f"\n  {warn(f'Expected move at rank 3')}")
            passed += 0.5
        elif found_at:
            print(f"\n  {warn(f'Expected move at rank {found_at}')}")
            passed += 0.25
        else:
            print(f"\n  {fail('Network misses passed pawn concept')}")

        results.add_diagnostic("passed_pawn", f"{test['name']}_rank", found_at)
        results.add_diagnostic("passed_pawn", f"{test['name']}_value", float(value))

    score = passed / total
    results.add_diagnostic("passed_pawn", "total_tested", total)
    results.add_diagnostic("passed_pawn", "correct", passed)
    results.add("Passed Pawn", score >= 0.5, score, 1.0)

    return score
