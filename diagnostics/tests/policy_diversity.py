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
    encode_for_network,
)


def test_policy_diversity(network, results: TestResults):
    """Test if the policy head outputs diverse moves or is collapsed."""
    print(header("TEST: Policy Head Diversity"))

    board = chess.Board()
    state = encode_for_network(board, network)
    policy, _ = network.predict_single(state)

    # Calculate entropy (suppress warnings for edge cases)
    policy_clipped = np.clip(policy, 1e-10, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.nansum(policy_clipped * np.log(policy_clipped))
    max_entropy = np.log(len(policy))
    normalized_entropy = entropy / max_entropy

    # Get statistics
    top_prob = np.max(policy)
    sorted_policy = np.sort(policy)[::-1]
    top_5_prob = np.sum(sorted_policy[:5])
    top_10_prob = np.sum(sorted_policy[:10])
    top_20_prob = np.sum(sorted_policy[:20])
    num_above_1pct = np.sum(policy > 0.01)
    num_above_5pct = np.sum(policy > 0.05)
    num_nonzero = np.sum(policy > 1e-6)

    # Legal move analysis
    legal_moves = list(board.legal_moves)
    legal_indices = [encode_move(m) for m in legal_moves]
    legal_indices = [i for i in legal_indices if i is not None]
    legal_prob = sum(policy[i] for i in legal_indices)

    print(subheader("Policy Distribution Statistics"))
    print(f"  {'Metric':<35} {'Value':>15}")
    print("  " + "-" * 55)
    print(f"  {'Total policy outputs':<35} {len(policy):>15}")
    print(f"  {'Legal moves in position':<35} {len(legal_moves):>15}")
    print(f"  {'Probability on legal moves':<35} {legal_prob*100:>14.1f}%")
    print(f"  {'Non-zero probabilities':<35} {num_nonzero:>15}")
    print(f"  {'Moves with >1% probability':<35} {num_above_1pct:>15}")
    print(f"  {'Moves with >5% probability':<35} {num_above_5pct:>15}")
    print("  " + "-" * 55)
    print(f"  {'Top move probability':<35} {top_prob*100:>14.1f}%")
    print(f"  {'Top 5 moves combined':<35} {top_5_prob*100:>14.1f}%")
    print(f"  {'Top 10 moves combined':<35} {top_10_prob*100:>14.1f}%")
    print(f"  {'Top 20 moves combined':<35} {top_20_prob*100:>14.1f}%")
    print("  " + "-" * 55)
    print(f"  {'Entropy':<35} {entropy:>15.4f}")
    print(f"  {'Max possible entropy':<35} {max_entropy:>15.4f}")
    print(f"  {'Normalized entropy':<35} {normalized_entropy:>15.4f}")

    # Store diagnostics
    results.add_diagnostic("policy_diversity", "top_prob", float(top_prob))
    results.add_diagnostic("policy_diversity", "top_5_prob", float(top_5_prob))
    results.add_diagnostic("policy_diversity", "top_10_prob", float(top_10_prob))
    results.add_diagnostic("policy_diversity", "entropy", float(entropy))
    results.add_diagnostic(
        "policy_diversity", "normalized_entropy", float(normalized_entropy)
    )
    results.add_diagnostic("policy_diversity", "num_above_1pct", int(num_above_1pct))
    results.add_diagnostic("policy_diversity", "legal_prob", float(legal_prob))

    # Show top moves
    print(subheader("Top 10 Moves"))
    top_indices = np.argsort(policy)[::-1][:10]
    for i, idx in enumerate(top_indices):
        move = decode_move(idx, board)
        prob = policy[idx]
        legal_str = (
            ""
            if move and move in legal_moves
            else f" {Colors.RED}(illegal){Colors.ENDC}"
        )
        move_str = move.uci() if move else f"[idx {idx}]"
        print(f"  {i+1:>2}. {move_str:<8} {prob*100:>6.2f}%{legal_str}")

    # Determine status
    issues = []
    if top_prob > 0.5:
        issues.append(f"Policy collapsed - one move at {top_prob*100:.0f}%")
        results.add_issue(
            "HIGH",
            "policy",
            f"Policy is collapsed - top move at {top_prob*100:.0f}%",
            "Network has very low diversity, may always play same move",
        )
    if top_5_prob > 0.9:
        issues.append(f"Policy narrow - top 5 at {top_5_prob*100:.0f}%")
    if num_above_1pct < 10:
        issues.append(f"Only {num_above_1pct} moves above 1%")
    if normalized_entropy < 0.2:
        issues.append(f"Very low entropy ({normalized_entropy:.3f})")
        results.add_issue(
            "HIGH",
            "policy",
            f"Policy entropy too low ({normalized_entropy:.3f})",
            "Network may have overfit or training collapsed",
        )
    if legal_prob < 0.9:
        issues.append(f"Only {legal_prob*100:.0f}% on legal moves")
        results.add_issue(
            "MEDIUM",
            "policy",
            f"Significant probability ({(1-legal_prob)*100:.0f}%) on illegal moves",
            "Policy head may not be learning move legality properly",
        )

    if issues:
        print(f"\n  {warn('Issues detected:')}")
        for issue in issues:
            print(f"    • {issue}")
        passed = False
    else:
        print(f"\n  {ok('Policy has healthy diversity')}")
        passed = True

    results.add("Policy Diversity", passed, 1.0 if passed else 0.5, 1.0)
    return 1.0 if passed else 0.0
