"""Test: Policy Head Diversity."""

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


# Test positions covering different game phases and situations
TEST_POSITIONS = [
    # === OPENING POSITIONS (2) ===
    {
        "name": "Starting Position",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "description": "Standard opening - should have diverse first moves",
        "expected_legal": 20,
    },
    {
        "name": "After 1.e4",
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "description": "Black's response to 1.e4",
        "expected_legal": 20,
    },
    # === COMPLEX MIDDLEGAME (2) ===
    {
        "name": "Complex Middlegame",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "description": "Italian Game - many strategic options",
        "expected_legal": 33,
    },
    {
        "name": "Tactical Middlegame",
        "fen": "r2qkb1r/ppp2ppp/2n1bn2/4p1B1/4P3/2N2N2/PPP2PPP/R2QKB1R w KQkq - 0 6",
        "description": "Active pieces with tension",
        "expected_legal": 40,
    },
    # === FORCED POSITIONS (2) ===
    {
        "name": "Few Legal Moves",
        "fen": "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
        "description": "King + pawns ending - limited options",
        "expected_legal": 8,
    },
    {
        "name": "Check Response",
        "fen": "rnbqkbnr/ppppp1pp/8/5p1Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2",
        "description": "Must block or move king",
        "expected_legal": 4,
    },
    # === ENDGAMES (2) ===
    {
        "name": "Rook Endgame",
        "fen": "8/5pk1/8/8/8/8/5PK1/R7 w - - 0 1",
        "description": "Rook vs pawns endgame",
        "expected_legal": 18,
    },
    {
        "name": "Queen Endgame",
        "fen": "8/5pk1/8/8/8/8/5PK1/Q7 w - - 0 1",
        "description": "Queen vs pawns - many queen moves",
        "expected_legal": 27,
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
        "legal_prob": legal_prob,
        "legal_moves": legal_moves,
        "num_legal": len(legal_moves),
    }


def test_policy_diversity(network, results: TestResults):
    """Test if the policy head outputs diverse moves or is collapsed."""
    print(header("TEST: Policy Head Diversity"))

    total_entropy = 0
    total_legal_prob = 0
    collapsed_positions = 0
    low_entropy_positions = 0
    low_legal_positions = 0
    position_results = []
    passed = 0.0  # Progressive score

    for test in TEST_POSITIONS:
        board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))

        stats = _analyze_position(board, network)
        position_results.append(stats)

        total_entropy += stats["normalized_entropy"]
        total_legal_prob += stats["legal_prob"]

        # Progressive scoring for each position
        position_score = 0.0

        # Score based on entropy (higher is better, max 0.4)
        if stats["normalized_entropy"] >= 0.4:
            position_score += 0.4
        elif stats["normalized_entropy"] >= 0.3:
            position_score += 0.3
        elif stats["normalized_entropy"] >= 0.2:
            position_score += 0.2
        elif stats["normalized_entropy"] >= 0.1:
            position_score += 0.1

        # Score based on legal probability (max 0.3)
        if stats["legal_prob"] >= 0.95:
            position_score += 0.3
        elif stats["legal_prob"] >= 0.9:
            position_score += 0.25
        elif stats["legal_prob"] >= 0.8:
            position_score += 0.15
        elif stats["legal_prob"] >= 0.7:
            position_score += 0.1

        # Score based on top move concentration (lower is better, max 0.3)
        if stats["top_prob"] < 0.3:
            position_score += 0.3  # Excellent diversity
        elif stats["top_prob"] < 0.4:
            position_score += 0.25
        elif stats["top_prob"] < 0.5:
            position_score += 0.15
        elif stats["top_prob"] < 0.7:
            position_score += 0.05  # Still some diversity

        passed += position_score

        # Check for issues
        issues = []
        if stats["top_prob"] > 0.5:
            collapsed_positions += 1
            issues.append(f"Collapsed: top at {stats['top_prob']*100:.0f}%")
        if stats["normalized_entropy"] < 0.2:
            low_entropy_positions += 1
            issues.append(f"Low entropy: {stats['normalized_entropy']:.3f}")
        if stats["legal_prob"] < 0.9:
            low_legal_positions += 1
            issues.append(f"Legal prob: {stats['legal_prob']*100:.0f}%")

        # Print summary for position
        print(f"  Legal moves: {stats['num_legal']}")
        print(f"  Top move: {stats['top_prob']*100:.1f}%")
        print(f"  Top 5: {stats['top_5_prob']*100:.1f}%")
        print(f"  Norm. entropy: {stats['normalized_entropy']:.3f}")
        print(f"  Legal prob: {stats['legal_prob']*100:.1f}%")
        print(f"  Moves >1%: {stats['num_above_1pct']}")

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

        if issues:
            print(f"\n  {warn(' | '.join(issues))}")
        else:
            print(f"\n  {ok('Healthy diversity')}")

        # Store per-position diagnostics
        results.add_diagnostic(
            "policy_diversity", f"{test['name']}_entropy", float(stats["normalized_entropy"])
        )
        results.add_diagnostic(
            "policy_diversity", f"{test['name']}_top_prob", float(stats["top_prob"])
        )
        results.add_diagnostic(
            "policy_diversity", f"{test['name']}_legal_prob", float(stats["legal_prob"])
        )

    # Aggregate statistics
    num_positions = len(TEST_POSITIONS)
    avg_entropy = total_entropy / num_positions
    avg_legal_prob = total_legal_prob / num_positions

    print(subheader("Summary Statistics"))
    print(f"  Positions tested: {num_positions}")
    print(f"  Average normalized entropy: {avg_entropy:.3f}")
    print(f"  Average legal probability: {avg_legal_prob*100:.1f}%")
    print(f"  Collapsed positions (top>50%): {collapsed_positions}/{num_positions}")
    print(f"  Low entropy positions (<0.2): {low_entropy_positions}/{num_positions}")
    print(f"  Low legal prob positions (<90%): {low_legal_positions}/{num_positions}")

    # Store aggregate diagnostics
    results.add_diagnostic("policy_diversity", "avg_entropy", float(avg_entropy))
    results.add_diagnostic("policy_diversity", "avg_legal_prob", float(avg_legal_prob))
    results.add_diagnostic("policy_diversity", "collapsed_count", collapsed_positions)
    results.add_diagnostic("policy_diversity", "positions_tested", num_positions)

    # Determine overall status
    issues = []
    if collapsed_positions > num_positions * 0.25:
        issues.append(f"{collapsed_positions} positions have collapsed policy")
        results.add_issue(
            "HIGH",
            "policy",
            f"Policy collapsed in {collapsed_positions}/{num_positions} positions",
            "Network may always play the same moves",
        )
    if avg_entropy < 0.2:
        issues.append(f"Average entropy too low ({avg_entropy:.3f})")
        results.add_issue(
            "HIGH",
            "policy",
            f"Average policy entropy too low ({avg_entropy:.3f})",
            "Network may have overfit or training collapsed",
        )
    if avg_legal_prob < 0.85:
        issues.append(f"Low legal move probability ({avg_legal_prob*100:.0f}%)")
        results.add_issue(
            "MEDIUM",
            "policy",
            f"Average legal probability only {avg_legal_prob*100:.0f}%",
            "Policy head may not be learning move legality properly",
        )

    # Calculate progressive score (max 1.0 per position)
    score = passed / num_positions

    if issues:
        print(f"\n  {fail('Issues detected:')}")
        for issue in issues:
            print(f"    • {issue}")
        test_passed = False
    else:
        print(f"\n  {ok('Policy has healthy diversity across all positions')}")
        test_passed = True

    # Show progressive score breakdown
    print(f"\n  Progressive score: {score*100:.1f}%")

    results.add("Policy Diversity", test_passed, score, 1.0)
    return score
