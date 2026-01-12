"""Test: Prophylaxis (Preventive Moves)."""

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


# Test positions for prophylactic thinking
TEST_POSITIONS = [
    # === PREVENT OPPONENT'S PLAN ===
    {
        "name": "Prevent Ng5",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "expected_moves": ["h2h3"],  # Prevents Ng4/Bg4
        "description": "h3 prevents Bg4/Ng4 pins",
        "prophylaxis_type": "prevent_pin",
    },
    {
        "name": "Prevent Back Rank Mate",
        "fen": "3r2k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
        "expected_moves": ["g1f1", "g1h1", "h2h3", "g2g3"],  # Luft
        "description": "Create luft (escape square)",
        "prophylaxis_type": "prevent_mate",
    },
    {
        "name": "Prevent Expansion",
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        "expected_moves": ["d2d4", "b1c3", "f1c4"],  # Control center
        "description": "Prevent Black from expanding",
        "prophylaxis_type": "prevent_expansion",
    },
    # === RESTRICT OPPONENT'S PIECES ===
    {
        "name": "Restrict Knight",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1",
        "expected_moves": ["d2d3", "a2a3"],  # Slow but restrictive
        "description": "Restrict knight maneuvers",
        "prophylaxis_type": "restrict_piece",
    },
    {
        "name": "Restrict Bishop",
        "fen": "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 1",
        "expected_moves": ["c1d2", "a2a3"],  # Deal with pin
        "description": "Deal with annoying bishop",
        "prophylaxis_type": "restrict_piece",
    },
    # === PROPHYLACTIC PIECE PLACEMENT ===
    {
        "name": "Prophylactic King Move",
        "fen": "8/8/8/4k3/4p3/4K3/4P3/8 w - - 0 1",
        "expected_moves": ["e3d2", "e3f2"],  # Prepare for pawn race
        "description": "Position king for endgame",
        "prophylaxis_type": "king_placement",
    },
    {
        "name": "Prophylactic Rook Lift",
        "fen": "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w - - 0 1",
        "expected_moves": ["f1e1", "a2a3", "b2b3"],  # Improve rook
        "description": "Improve rook position",
        "prophylaxis_type": "piece_improvement",
    },
    # === PREVENT TACTICS ===
    {
        "name": "Prevent Fork",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4N3/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 1",
        "expected_moves": ["d7d6"],  # Kick the knight
        "description": "Remove dangerous knight",
        "prophylaxis_type": "prevent_tactic",
    },
    {
        "name": "Prevent Discovered Attack",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 1",
        "expected_moves": ["f8e7", "c6a5"],  # Develop safely
        "description": "Avoid discovered attack setup",
        "prophylaxis_type": "prevent_tactic",
    },
]


def test_prophylaxis(network, results: TestResults):
    """Test if network plays prophylactic moves."""
    print(header("TEST: Prophylaxis"))

    passed = 0
    total = len(TEST_POSITIONS)

    for test in TEST_POSITIONS:
        board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        policy, value = predict_for_board(board, network)

        top_5 = np.argsort(policy)[::-1][:5]
        expected_set = set(test["expected_moves"])
        prophylaxis_type = test["prophylaxis_type"]

        print(f"\n  Type: {prophylaxis_type}")
        print(f"  Value: {value:+.4f}")
        print(f"  {'Rank':<6} {'Move':<8} {'Prob':>8} {'Prophylactic?':<15}")
        print("  " + "-" * 45)

        found_at = None
        for i, idx in enumerate(top_5):
            move = decode_move(idx, board)
            prob = policy[idx]
            if move:
                uci = move.uci()
                is_prophylactic = uci in expected_set
                if is_prophylactic and found_at is None:
                    found_at = i + 1
                proph_str = "PROPHYLACTIC" if is_prophylactic else ""
                color = Colors.GREEN if is_prophylactic else ""
                end_color = Colors.ENDC if color else ""
                print(f"  {color}{i+1:<6} {uci:<8} {prob*100:>7.2f}% {proph_str}{end_color}")

        if found_at == 1:
            print(f"\n  {ok('Finds prophylactic move!')}")
            passed += 1
        elif found_at and found_at <= 3:
            print(f"\n  {warn(f'Prophylactic move at rank {found_at}')}")
            passed += 0.5
        elif found_at:
            print(f"\n  {warn(f'Prophylactic move at rank {found_at} (low)')}")
            passed += 0.25
        else:
            print(f"\n  {fail('Misses prophylactic concept')}")

        results.add_diagnostic("prophylaxis", f"{test['name']}_rank", found_at)
        results.add_diagnostic("prophylaxis", f"{test['name']}_type", prophylaxis_type)

    score = passed / total
    results.add_diagnostic("prophylaxis", "total_tested", total)
    results.add_diagnostic("prophylaxis", "correct", passed)
    results.add("Prophylaxis", score >= 0.3, score, 1.0)

    return score
