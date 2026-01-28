"""Test: Tempo Understanding."""

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


# Test positions for tempo understanding
TEST_POSITIONS = [
    # === GAIN TEMPO ===
    {
        "name": "Gain Tempo with Attack",
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        "expected_moves": ["f1c4", "f1b5", "d2d4"],  # Develop with tempo
        "description": "Develop attacking something",
        "tempo_type": "gain",
    },
    {
        "name": "Attack Undefended Piece",
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "expected_moves": ["b1c3", "d2d3", "c2c3"],  # Develop with purpose
        "description": "Continue development",
        "tempo_type": "gain",
    },
    # === LOSE TEMPO (avoid) ===
    {
        "name": "Avoid Moving Same Piece",
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        "bad_moves": ["f3e5", "f3g1", "f3h4"],  # Moving knight again wastes tempo
        "description": "Don't move same piece twice in opening",
        "tempo_type": "avoid_waste",
    },
    {
        "name": "Don't Waste Tempo on Pawn Moves",
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        "expected_moves": ["f1b5", "f1c4", "d2d4", "b1c3"],  # Piece development
        "bad_moves": ["h2h3", "a2a3"],  # Slow pawn moves
        "description": "Develop pieces, not pawns",
        "tempo_type": "avoid_waste",
    },
    # === TEMPO IN TACTICS ===
    {
        "name": "Zwischenzug Tempo",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4N3/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 1",
        "expected_moves": ["d7d5"],  # Counter in center
        "description": "Counter-attack gains tempo",
        "tempo_type": "gain",
    },
    {
        "name": "Discovered Attack Tempo",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "expected_moves": ["f3g5"],  # Ng5 threatens f7
        "description": "Ng5 attacks with tempo on f7",
        "tempo_type": "gain",
    },
    # === DEVELOPMENT TEMPO ===
    {
        "name": "Rapid Development",
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        "expected_moves": ["g1f3", "b1c3", "f1c4"],  # Fast development
        "description": "Develop pieces quickly",
        "tempo_type": "gain",
    },
    {
        "name": "Castle for Tempo",
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "expected_moves": ["e1g1"],  # Castle quickly
        "description": "Castling develops rook",
        "tempo_type": "gain",
    },
]


def test_tempo(network, results: TestResults):
    """Test if network understands tempo."""
    print(header("TEST: Tempo Understanding"))

    passed = 0
    total = 0

    for test in TEST_POSITIONS:
        board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        policy, value = predict_for_board(board, network)

        top_5 = np.argsort(policy)[::-1][:5]

        print(f"\n  Value: {value:+.4f}")
        print(f"  {'Rank':<6} {'Move':<8} {'Prob':>8} {'Quality':<12}")
        print("  " + "-" * 40)

        if test["tempo_type"] == "gain" or test["tempo_type"] == "avoid_waste":
            total += 1
            expected_set = set(test.get("expected_moves", []))
            bad_set = set(test.get("bad_moves", []))

            good_found_at = None
            bad_found_at = None

            for i, idx in enumerate(top_5):
                move = decode_move(idx, board)
                prob = policy[idx]
                if move:
                    uci = move.uci()
                    is_good = uci in expected_set
                    is_bad = uci in bad_set

                    if is_good and good_found_at is None:
                        good_found_at = i + 1
                    if is_bad and bad_found_at is None:
                        bad_found_at = i + 1

                    quality = "GOOD" if is_good else ("BAD" if is_bad else "")
                    color = Colors.GREEN if is_good else (Colors.RED if is_bad else "")
                    end_color = Colors.ENDC if color else ""
                    print(
                        f"  {color}{i+1:<6} {uci:<8} {prob*100:>7.2f}% {quality}{end_color}"
                    )

            # Scoring
            if expected_set:
                if good_found_at == 1:
                    print(f"\n  {ok('Understands tempo!')}")
                    passed += 1
                elif good_found_at and good_found_at <= 3:
                    print(f"\n  {warn(f'Good move at rank {good_found_at}')}")
                    passed += 0.5
                elif bad_found_at == 1:
                    print(f"\n  {fail('Plays tempo-wasting move')}")
                else:
                    print(f"\n  {warn('Mixed tempo understanding')}")
                    passed += 0.5
            else:
                # Only checking bad moves
                if bad_found_at == 1:
                    print(f"\n  {fail('Wastes tempo')}")
                else:
                    print(f"\n  {ok('Avoids wasting tempo')}")
                    passed += 1

            results.add_diagnostic("tempo", f"{test['name']}_good_rank", good_found_at)
            results.add_diagnostic("tempo", f"{test['name']}_bad_rank", bad_found_at)

    score = passed / total if total > 0 else 0
    results.add_diagnostic("tempo", "total_tested", total)
    results.add_diagnostic("tempo", "correct", passed)
    results.add("Tempo", score >= 0.4, score, 1.0)

    return score
