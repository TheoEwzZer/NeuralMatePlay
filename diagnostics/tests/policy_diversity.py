"""Test: Policy Head Diversity (Adaptive).

This test measures whether the network has appropriate diversity based on
position complexity:
- Simple positions (clear best move) → should be CONCENTRATED (low entropy)
- Complex positions (multiple good options) → should be DIVERSE (high entropy)

A good network should be confident when there's a clear answer and uncertain
when the position is genuinely unclear.
"""

import numpy as np
import chess

from alphazero.move_encoding import encode_move, decode_move
from ..core import (
    Colors,
    TestResults,
    header,
    subheader,
    ok,
    warn,
    fail,
    encode_for_network,
)


# Test positions categorized by expected diversity level
TEST_POSITIONS = [
    # === SIMPLE POSITIONS - Should be CONCENTRATED (low entropy OK) ===
    {
        "name": "Starting Position",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "description": "Standard opening - few principled moves",
        "category": "simple",
        "expected_diversity": "low_to_medium",  # e4, d4, Nf3, c4 are main choices
    },
    {
        "name": "Obvious Recapture",
        "fen": "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
        "description": "Black should recapture on d5",
        "category": "simple",
        "expected_diversity": "low",  # Qxd5 or exd5 clearly best
    },
    {
        "name": "Forced Mate Response",
        "fen": "rnbqkbnr/ppppp1pp/8/5p1Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2",
        "description": "Must block with g6",
        "category": "simple",
        "expected_diversity": "very_low",  # Only g6 saves the game
    },
    {
        "name": "King and Pawn Endgame",
        "fen": "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
        "description": "Simple pawn push or king move",
        "category": "simple",
        "expected_diversity": "low",  # Ke4 or e4 clearly best
    },
    # === COMPLEX POSITIONS - Should be DIVERSE (high entropy expected) ===
    {
        "name": "Complex Middlegame",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "description": "Italian Game - many strategic options",
        "category": "complex",
        "expected_diversity": "high",  # d3, c3, Nc3, O-O, b4 all playable
    },
    {
        "name": "Tactical Middlegame",
        "fen": "r2qkb1r/ppp2ppp/2n1bn2/4p1B1/4P3/2N2N2/PPP2PPP/R2QKB1R w KQkq - 0 6",
        "description": "Multiple tactical ideas",
        "category": "complex",
        "expected_diversity": "high",  # Many pieces, multiple plans
    },
    {
        "name": "Quiet Position Many Options",
        "fen": "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
        "description": "Many reasonable developing moves",
        "category": "complex",
        "expected_diversity": "high",  # h3, a4, Be3, Bg5, Re1 all good
    },
    {
        "name": "Rook Endgame",
        "fen": "8/5pk1/8/8/8/8/5PK1/R7 w - - 0 1",
        "description": "Multiple rook moves reasonable",
        "category": "complex",
        "expected_diversity": "medium_to_high",  # Many rook moves playable
    },
]


def _analyze_position(board, network):
    """Analyze policy diversity for a single position."""
    state = encode_for_network(board, network)
    policy, _ = network.predict_single(state)

    # Calculate entropy
    policy_clipped = np.clip(policy, 1e-10, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(policy_clipped * np.log(policy_clipped))
    max_entropy = np.log(len(policy))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Get statistics
    top_prob = np.max(policy)
    sorted_policy = np.sort(policy)[::-1]
    top_5_prob = np.sum(sorted_policy[:5])
    top_10_prob = np.sum(sorted_policy[:10])
    num_above_1pct = np.sum(policy > 0.01)
    num_above_5pct = np.sum(policy > 0.05)

    # Legal move analysis
    legal_moves = list(board.legal_moves)
    legal_indices = [encode_move(m) for m in legal_moves]
    legal_indices = [i for i in legal_indices if i is not None]
    legal_prob = sum(policy[i] for i in legal_indices) if legal_indices else 0

    return {
        "policy": policy,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "top_prob": top_prob,
        "top_5_prob": top_5_prob,
        "top_10_prob": top_10_prob,
        "num_above_1pct": num_above_1pct,
        "num_above_5pct": num_above_5pct,
        "legal_prob": legal_prob,
        "legal_moves": legal_moves,
        "num_legal": len(legal_moves),
    }


def _score_diversity_match(stats, expected_diversity):
    """
    Score how well the actual diversity matches expected diversity.

    Returns a score from 0 to 1 based on whether the network's confidence
    level is appropriate for the position complexity.
    """
    norm_entropy = stats["normalized_entropy"]
    top_prob = stats["top_prob"]

    if expected_diversity == "very_low":
        # Should be very concentrated (top move > 70%, entropy < 0.15)
        if top_prob > 0.7 and norm_entropy < 0.15:
            return 1.0
        elif top_prob > 0.5 and norm_entropy < 0.25:
            return 0.8
        elif top_prob > 0.4:
            return 0.5
        else:
            return 0.2  # Too diverse for a forced position

    elif expected_diversity == "low":
        # Should be fairly concentrated (top move > 50%, entropy < 0.25)
        if top_prob > 0.5 and norm_entropy < 0.25:
            return 1.0
        elif top_prob > 0.35 and norm_entropy < 0.35:
            return 0.8
        elif top_prob > 0.25:
            return 0.6
        else:
            return 0.3  # Too diverse for a simple position

    elif expected_diversity == "low_to_medium":
        # Moderate concentration acceptable (top move 30-60%)
        if 0.25 < top_prob < 0.65 and 0.15 < norm_entropy < 0.4:
            return 1.0
        elif 0.2 < top_prob < 0.75:
            return 0.8
        else:
            return 0.5

    elif expected_diversity == "medium_to_high":
        # Should show diversity (top move < 50%, entropy > 0.2)
        if top_prob < 0.5 and norm_entropy > 0.2:
            return 1.0
        elif top_prob < 0.6 and norm_entropy > 0.15:
            return 0.8
        elif top_prob < 0.7:
            return 0.5
        else:
            return 0.2  # Too concentrated for a complex position

    elif expected_diversity == "high":
        # Should be diverse (top move < 40%, entropy > 0.25)
        if top_prob < 0.4 and norm_entropy > 0.25:
            return 1.0
        elif top_prob < 0.5 and norm_entropy > 0.2:
            return 0.8
        elif top_prob < 0.6 and norm_entropy > 0.15:
            return 0.6
        else:
            return 0.3  # Too concentrated for a complex position

    return 0.5  # Default


def test_policy_diversity(network, results: TestResults):
    """Test if the policy head has ADAPTIVE diversity (appropriate to position complexity)."""
    print(header("TEST: Policy Head Diversity (Adaptive)"))

    print(f"\n  {Colors.CYAN}This test checks if diversity matches position complexity:{Colors.ENDC}")
    print(f"  - Simple positions (forced moves) → Should be CONCENTRATED")
    print(f"  - Complex positions (many options) → Should be DIVERSE")
    print()

    total_score = 0.0
    total_legal_prob = 0.0
    simple_scores = []
    complex_scores = []
    position_results = []

    for test in TEST_POSITIONS:
        board = chess.Board(test["fen"])
        category = test["category"]
        expected = test["expected_diversity"]

        print(subheader(f"{test['name']} [{category.upper()}]: {test['description']}"))
        print(f"  Expected diversity: {expected}")

        stats = _analyze_position(board, network)
        position_results.append(stats)

        # Score based on match with expected diversity
        diversity_score = _score_diversity_match(stats, expected)

        # Bonus/penalty for legal probability
        legal_bonus = 0.0
        if stats["legal_prob"] >= 0.95:
            legal_bonus = 0.1
        elif stats["legal_prob"] >= 0.9:
            legal_bonus = 0.05
        elif stats["legal_prob"] < 0.8:
            legal_bonus = -0.1

        position_score = min(1.0, max(0.0, diversity_score + legal_bonus))
        total_score += position_score
        total_legal_prob += stats["legal_prob"]

        if category == "simple":
            simple_scores.append(position_score)
        else:
            complex_scores.append(position_score)

        # Print summary for position
        print(f"  Legal moves: {stats['num_legal']}")
        print(f"  Top move: {stats['top_prob']*100:.1f}%")
        print(f"  Top 5: {stats['top_5_prob']*100:.1f}%")
        print(f"  Norm. entropy: {stats['normalized_entropy']:.3f}")
        print(f"  Legal prob: {stats['legal_prob']*100:.1f}%")
        print(f"  Moves >5%: {stats['num_above_5pct']}")

        # Show top 5 moves
        top_indices = np.argsort(stats["policy"])[::-1][:5]
        print(f"\n  Top 5 moves:")
        for i, idx in enumerate(top_indices):
            move = decode_move(idx, board)
            prob = stats["policy"][idx]
            legal_str = (
                ""
                if move and move in stats["legal_moves"]
                else f" {Colors.RED}(illegal){Colors.ENDC}"
            )
            move_str = move.uci() if move else f"[idx {idx}]"
            print(f"    {i+1}. {move_str:<8} {prob*100:>6.2f}%{legal_str}")

        # Show score with color
        if position_score >= 0.8:
            print(f"\n  {ok(f'Diversity match score: {position_score*100:.0f}%')}")
        elif position_score >= 0.5:
            print(f"\n  {warn(f'Diversity match score: {position_score*100:.0f}%')}")
        else:
            print(f"\n  {fail(f'Diversity match score: {position_score*100:.0f}%')}")

        # Store per-position diagnostics
        results.add_diagnostic(
            "policy_diversity", f"{test['name']}_entropy", float(stats["normalized_entropy"])
        )
        results.add_diagnostic(
            "policy_diversity", f"{test['name']}_top_prob", float(stats["top_prob"])
        )
        results.add_diagnostic(
            "policy_diversity", f"{test['name']}_match_score", float(position_score)
        )

    # Aggregate statistics
    num_positions = len(TEST_POSITIONS)
    avg_score = total_score / num_positions
    avg_legal_prob = total_legal_prob / num_positions
    avg_simple = np.mean(simple_scores) if simple_scores else 0
    avg_complex = np.mean(complex_scores) if complex_scores else 0

    print(subheader("Summary Statistics"))
    print(f"  Positions tested: {num_positions}")
    print(f"  Average legal probability: {avg_legal_prob*100:.1f}%")
    print()
    print(f"  {Colors.CYAN}Adaptive Diversity Scores:{Colors.ENDC}")
    print(f"    Simple positions (should concentrate): {avg_simple*100:.1f}%")
    print(f"    Complex positions (should diversify):  {avg_complex*100:.1f}%")
    print(f"    Overall match score:                   {avg_score*100:.1f}%")

    # Store aggregate diagnostics
    results.add_diagnostic("policy_diversity", "avg_score", float(avg_score))
    results.add_diagnostic("policy_diversity", "avg_legal_prob", float(avg_legal_prob))
    results.add_diagnostic("policy_diversity", "simple_score", float(avg_simple))
    results.add_diagnostic("policy_diversity", "complex_score", float(avg_complex))

    # Determine overall status
    issues = []

    if avg_simple < 0.5:
        issues.append(f"Network too diverse on simple positions ({avg_simple*100:.0f}%)")
        results.add_issue(
            "MEDIUM",
            "policy",
            f"Network not concentrating on simple positions ({avg_simple*100:.0f}%)",
            "May play suboptimal moves when there's a clear best choice",
        )

    if avg_complex < 0.5:
        issues.append(f"Network too concentrated on complex positions ({avg_complex*100:.0f}%)")
        results.add_issue(
            "MEDIUM",
            "policy",
            f"Network not diversifying on complex positions ({avg_complex*100:.0f}%)",
            "May miss good alternatives in unclear positions",
        )

    if avg_legal_prob < 0.85:
        issues.append(f"Low legal move probability ({avg_legal_prob*100:.0f}%)")
        results.add_issue(
            "HIGH",
            "policy",
            f"Average legal probability only {avg_legal_prob*100:.0f}%",
            "Policy head may not be learning move legality properly",
        )

    if issues:
        print(f"\n  {fail('Issues detected:')}")
        for issue in issues:
            print(f"    {issue}")
        test_passed = False
    else:
        print(f"\n  {ok('Policy has appropriate adaptive diversity')}")
        test_passed = True

    print(f"\n  Final score: {avg_score*100:.1f}%")

    results.add("Policy Diversity", test_passed, avg_score, 1.0)
    return avg_score
