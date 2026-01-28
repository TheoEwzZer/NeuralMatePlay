"""Test: Opening Development."""

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


# Test positions covering different opening phases
TEST_POSITIONS = [
    # === STARTING POSITION (1) ===
    {
        "name": "Starting Position (White)",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "excellent": {"e2e4", "d2d4"},
        "good": {"g1f3", "c2c4", "b1c3"},
        "okay": {"e2e3", "d2d3", "g2g3", "b2b3"},
        "dubious": {"h2h4", "a2a4", "g1h3", "b1a3", "f2f4"},
    },
    # === AFTER 1.e4 (Black's response) ===
    {
        "name": "After 1.e4 (Black)",
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "excellent": {"e7e5", "c7c5"},  # Open game, Sicilian
        "good": {"e7e6", "c7c6", "d7d6", "g8f6"},  # French, Caro-Kann, Pirc, Alekhine
        "okay": {"d7d5", "g7g6"},  # Scandinavian, Modern
        "dubious": {"h7h6", "a7a6", "b8a6", "g8h6"},
    },
    # === AFTER 1.d4 (Black's response) ===
    {
        "name": "After 1.d4 (Black)",
        "fen": "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
        "excellent": {"d7d5", "g8f6"},  # Queen's Gambit, Indian
        "good": {"e7e6", "c7c5", "f7f5"},  # Queen's Indian, Benoni, Dutch
        "okay": {"d7d6", "g7g6"},  # Pirc, King's Indian setup
        "dubious": {"h7h6", "a7a6", "b8a6", "g8h6"},
    },
    # === AFTER 1.e4 e5 (White's 2nd move) ===
    {
        "name": "After 1.e4 e5 (White)",
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "excellent": {"g1f3"},  # King's Knight
        "good": {"f1c4", "b1c3", "d2d4"},  # Italian setup, Vienna, Center Game
        "okay": {"f1b5", "f2f4"},  # Ruy Lopez, King's Gambit
        "dubious": {"h2h4", "a2a4", "g1h3", "d1h5"},
    },
    # === AFTER 1.d4 d5 (White's 2nd move) ===
    {
        "name": "After 1.d4 d5 (White)",
        "fen": "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",
        "excellent": {"c2c4"},  # Queen's Gambit
        "good": {"g1f3", "e2e3", "c1f4"},  # London, Colle
        "okay": {"b1c3", "c1g5"},  # Veresov
        "dubious": {"h2h4", "a2a4", "g1h3", "c1h6"},
    },
    # === SICILIAN DEFENSE (White's response) ===
    {
        "name": "Sicilian 1...c5 (White)",
        "fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "excellent": {"g1f3"},  # Main line
        "good": {"b1c3", "c2c3", "d2d4"},  # Closed, Alapin, immediate d4
        "okay": {"f1c4", "f2f4"},  # Grand Prix
        "dubious": {"h2h4", "a2a4", "g1h3", "d1h5"},
    },
    # === DEVELOPED POSITION (3-4 moves in) ===
    {
        "name": "Italian Game (White)",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "excellent": {"d2d3", "c2c3"},  # Giuoco Piano, Italian Game
        "good": {"b2b4", "d2d4", "b1c3"},  # Evans Gambit, Center attack
        "okay": {"e1g1", "a2a4"},  # Castle, slow
        "dubious": {"h2h4", "g1h3", "c4f7"},
    },
    {
        "name": "Queen's Gambit Declined",
        "fen": "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
        "excellent": {"c1g5", "g1f3"},  # Main lines
        "good": {"e2e3", "c1f4"},  # Solid development
        "okay": {"f1d3", "d1c2"},
        "dubious": {"h2h4", "a2a4", "g1h3"},
    },
]


def _analyze_position(board, network, test):
    """Analyze opening move quality for a single position."""
    policy, value = predict_for_board(board, network)

    top_5 = np.argsort(policy)[::-1][:5]

    excellent_prob = 0
    good_prob = 0
    okay_prob = 0
    dubious_prob = 0

    # Calculate probabilities for each category
    for idx in range(len(policy)):
        move = decode_move(idx, board)
        if move:
            uci = move.uci()
            prob = policy[idx]
            if uci in test["excellent"]:
                excellent_prob += prob
            elif uci in test["good"]:
                good_prob += prob
            elif uci in test["okay"]:
                okay_prob += prob
            elif uci in test["dubious"]:
                dubious_prob += prob

    # Get top move category
    top_move = decode_move(top_5[0], board)
    top_uci = top_move.uci() if top_move else None

    if top_uci in test["excellent"]:
        category = "excellent"
    elif top_uci in test["good"]:
        category = "good"
    elif top_uci in test["okay"]:
        category = "okay"
    elif top_uci in test["dubious"]:
        category = "dubious"
    else:
        category = "other"

    return {
        "policy": policy,
        "value": value,
        "top_move": top_uci,
        "top_category": category,
        "excellent_prob": excellent_prob,
        "good_prob": good_prob,
        "okay_prob": okay_prob,
        "dubious_prob": dubious_prob,
        "top_5": top_5,
    }


def test_development(network, results: TestResults):
    """Test if network develops pieces reasonably in opening."""
    print(header("TEST: Opening Development"))

    passed_count = 0
    total_excellent_prob = 0
    total_good_prob = 0
    total_dubious_prob = 0

    for test in TEST_POSITIONS:
        board = chess.Board(test["fen"])
        print(subheader(test["name"]))
        print(board)

        stats = _analyze_position(board, network, test)

        # Accumulate stats
        total_excellent_prob += stats["excellent_prob"]
        total_good_prob += stats["good_prob"]
        total_dubious_prob += stats["dubious_prob"]

        # Print top 5 moves
        print(f"\n  Value: {stats['value']:+.4f}")
        print(f"  {'Rank':<6} {'Move':<8} {'Prob':>8} {'Category':<15}")
        print("  " + "-" * 45)

        for i, idx in enumerate(stats["top_5"]):
            move = decode_move(idx, board)
            prob = stats["policy"][idx]
            if move:
                uci = move.uci()
                if uci in test["excellent"]:
                    cat = "EXCELLENT"
                    color = Colors.GREEN
                elif uci in test["good"]:
                    cat = "Good"
                    color = Colors.GREEN
                elif uci in test["okay"]:
                    cat = "Okay"
                    color = Colors.YELLOW
                elif uci in test["dubious"]:
                    cat = "Dubious"
                    color = Colors.RED
                else:
                    cat = ""
                    color = ""

                end_color = Colors.ENDC if color else ""
                print(f"  {color}{i+1:<6} {uci:<8} {prob*100:>7.2f}% {cat}{end_color}")

        # Check if position passed
        top_move = stats["top_move"]
        if stats["top_category"] in ["excellent", "good"]:
            print(f"\n  {ok(f'Good opening move: {top_move}')}")
            passed_count += 1
        elif stats["top_category"] == "okay":
            print(f"\n  {warn(f'Passive but playable: {top_move}')}")
            passed_count += 0.5
        else:
            print(f"\n  {fail(f'Poor opening move: {top_move}')}")

        # Store per-position diagnostics
        results.add_diagnostic("opening", f"{test['name']}_top_move", stats["top_move"])
        results.add_diagnostic(
            "opening", f"{test['name']}_category", stats["top_category"]
        )
        results.add_diagnostic(
            "opening", f"{test['name']}_excellent_prob", stats["excellent_prob"]
        )

    # Summary
    num_positions = len(TEST_POSITIONS)
    avg_excellent = total_excellent_prob / num_positions
    avg_good = total_good_prob / num_positions
    avg_dubious = total_dubious_prob / num_positions

    print(subheader("Summary"))
    print(f"  Positions tested: {num_positions}")
    print(f"  Passed: {passed_count}/{num_positions}")
    print(f"  Avg prob on excellent moves: {avg_excellent*100:.1f}%")
    print(f"  Avg prob on good moves: {avg_good*100:.1f}%")
    print(f"  Avg prob on dubious moves: {avg_dubious*100:.1f}%")

    results.add_diagnostic("opening", "positions_tested", num_positions)
    results.add_diagnostic("opening", "passed_count", passed_count)
    results.add_diagnostic("opening", "avg_excellent_prob", avg_excellent)
    results.add_diagnostic("opening", "avg_good_prob", avg_good)
    results.add_diagnostic("opening", "avg_dubious_prob", avg_dubious)

    # Issues
    if avg_dubious > 0.2:
        results.add_issue(
            "MEDIUM",
            "opening",
            f"High probability ({avg_dubious*100:.0f}%) on dubious opening moves",
            "Network may play h4, a4, etc. frequently",
        )

    score = passed_count / num_positions
    overall_passed = score >= 0.5

    results.add("Opening Development", overall_passed, score, 1.0)
    return score
