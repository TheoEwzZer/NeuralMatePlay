"""Test: WDL Head Material Evaluation (Nuanced).

This test evaluates if the WDL head correctly understands material imbalances
while allowing for nuanced evaluations. A sophisticated network should:
- Recognize clear material advantages
- Show appropriate confidence levels (not overconfident)
- Understand that material isn't everything (compensation exists)
"""

import numpy as np
import chess

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


def test_wdl_head(network, results: TestResults):
    """Test if the WDL head correctly evaluates material imbalances with nuance."""
    print(header("TEST: WDL Head Material Evaluation"))

    # Test positions with BOTH White to move (WTM) and Black to move (BTM)
    # Value is always from CURRENT PLAYER's perspective
    # "tolerance" indicates how much nuance is acceptable (higher = more flexible)
    test_positions = [
        # === Starting positions (should be neutral) ===
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "desc": "Start (WTM)",
            "material": 0,
            "tolerance": "strict",  # Should be very close to 0
        },
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            "desc": "Start (BTM)",
            "material": 0,
            "tolerance": "strict",
        },
        # === Clear material advantages (should show clear evaluation) ===
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1",
            "desc": "W -N (WTM)",
            "material": -3,
            "tolerance": "normal",
        },
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R b KQkq - 0 1",
            "desc": "W -N (BTM)",
            "material": +3,
            "tolerance": "normal",
        },
        {
            "fen": "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "desc": "B -N (WTM)",
            "material": +3,
            "tolerance": "normal",
        },
        {
            "fen": "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            "desc": "B -N (BTM)",
            "material": -3,
            "tolerance": "normal",
        },
        # === Large material differences (should be very clear) ===
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
            "desc": "W -Q (WTM)",
            "material": -9,
            "tolerance": "normal",
        },
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1",
            "desc": "W -Q (BTM)",
            "material": +9,
            "tolerance": "normal",
        },
        {
            "fen": "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "desc": "B -Q (WTM)",
            "material": +9,
            "tolerance": "normal",
        },
        {
            "fen": "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            "desc": "B -Q (BTM)",
            "material": -9,
            "tolerance": "normal",
        },
        # === Small material difference (nuance acceptable) ===
        {
            "fen": "rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "desc": "B -P (WTM)",
            "material": +1,
            "tolerance": "flexible",  # One pawn is small, nuance OK
        },
        {
            "fen": "rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            "desc": "B -P (BTM)",
            "material": -1,
            "tolerance": "flexible",
        },
    ]

    print(
        f"\n  {'Position':<20} {'Material':>10} {'Value':>10} {'Status':>12} {'Score':>8}"
    )
    print("  " + "-" * 65)

    passed = 0.0  # Progressive score
    correct = 0  # Full correct count
    total_error = 0
    wrong_sign = 0
    values = []

    # Track WTM vs BTM separately for bias detection
    wtm_correct = 0
    wtm_total = 0
    wtm_wrong_sign = 0
    btm_correct = 0
    btm_total = 0
    btm_wrong_sign = 0

    for test in test_positions:
        fen = test["fen"]
        desc = test["desc"]
        material = test["material"]
        tolerance = test["tolerance"]

        is_wtm = "(WTM)" in desc
        is_btm = "(BTM)" in desc
        board = chess.Board(fen)
        state = encode_for_network(board, network)
        _, value = network.predict_single(state)
        values.append(value)

        # Determine score based on tolerance level
        position_score, status, is_correct, has_wrong_sign = _evaluate_position(
            value, material, tolerance
        )

        passed += position_score

        # Update global counters
        if is_correct:
            correct += 1
        if has_wrong_sign:
            wrong_sign += 1

        # Update WTM/BTM counters
        if is_wtm:
            wtm_total += 1
            if is_correct:
                wtm_correct += 1
            if has_wrong_sign:
                wtm_wrong_sign += 1
        elif is_btm:
            btm_total += 1
            if is_correct:
                btm_correct += 1
            if has_wrong_sign:
                btm_wrong_sign += 1

        error = abs(value - material / 10)
        total_error += error

        # Color the status
        if position_score >= 0.85:
            status_colored = ok(status)
        elif position_score >= 0.5:
            status_colored = warn(status)
        else:
            status_colored = fail(status)

        print(
            f"  {desc:<20} {material:>+10} {value:>+10.4f} {status_colored:>12} {position_score*100:>7.0f}%"
        )

        results.add_diagnostic("material_eval", f"{desc}_value", float(value))
        results.add_diagnostic("material_eval", f"{desc}_material", material)
        results.add_diagnostic("material_eval", f"{desc}_score", position_score)

    print("  " + "-" * 65)

    # Statistical analysis
    avg_error = total_error / len(test_positions)
    value_std = np.std(values)
    value_range = max(values) - min(values)

    print(subheader("WDL Head Statistics"))
    print(f"  Correct evaluations:   {correct}/{len(test_positions)}")
    print(f"  Wrong sign errors:     {wrong_sign}")
    print(f"  Average error:         {avg_error:.4f}")
    print(f"  Value std deviation:   {value_std:.4f}")
    print(f"  Value range:           {value_range:.4f}")

    # WTM vs BTM breakdown
    print(subheader("Perspective Bias Analysis"))
    if wtm_total > 0:
        wtm_pct = wtm_correct * 100 / wtm_total
        print(
            f"  White to Move (WTM):   {wtm_correct}/{wtm_total} ({wtm_pct:.0f}%) correct, {wtm_wrong_sign} wrong sign"
        )
    if btm_total > 0:
        btm_pct = btm_correct * 100 / btm_total
        print(
            f"  Black to Move (BTM):   {btm_correct}/{btm_total} ({btm_pct:.0f}%) correct, {btm_wrong_sign} wrong sign"
        )

    # Detect perspective bias
    if wtm_total > 0 and btm_total > 0:
        wtm_accuracy = wtm_correct / wtm_total
        btm_accuracy = btm_correct / btm_total
        bias = wtm_accuracy - btm_accuracy

        if abs(bias) > 0.3:
            if bias > 0:
                print(
                    f"\n  {Colors.RED}[!] PERSPECTIVE BIAS DETECTED: WTM {bias*100:+.0f}% better than BTM{Colors.ENDC}"
                )
                results.add_issue(
                    "HIGH",
                    "wdl_head",
                    f"Perspective bias: WTM accuracy {wtm_pct:.0f}% vs BTM {btm_pct:.0f}%",
                    "Network evaluates differently depending on who is to move",
                )
            else:
                print(
                    f"\n  {Colors.RED}[!] PERSPECTIVE BIAS DETECTED: BTM {-bias*100:+.0f}% better than WTM{Colors.ENDC}"
                )
                results.add_issue(
                    "HIGH",
                    "wdl_head",
                    f"Perspective bias: BTM accuracy {btm_pct:.0f}% vs WTM {wtm_pct:.0f}%",
                    "Network evaluates differently depending on who is to move",
                )
        else:
            print(
                f"\n  {Colors.GREEN}[OK] No significant perspective bias detected{Colors.ENDC}"
            )

    results.add_diagnostic("material_eval", "correct_count", correct)
    results.add_diagnostic("material_eval", "wrong_sign_count", wrong_sign)
    results.add_diagnostic("material_eval", "avg_error", float(avg_error))
    results.add_diagnostic("material_eval", "value_std", float(value_std))
    results.add_diagnostic("material_eval", "value_range", float(value_range))
    results.add_diagnostic("material_eval", "all_values", [float(v) for v in values])

    # Identify issues
    if value_std < 0.1:
        results.add_issue(
            "CRITICAL",
            "wdl_head",
            f"WDL head has very low variance (std={value_std:.4f})",
            "Output is nearly constant regardless of position - training may have collapsed",
        )

    if wrong_sign > 2:  # Allow 1-2 wrong signs (nuance)
        results.add_issue(
            "HIGH",
            "wdl_head",
            f"WDL head gives wrong sign for {wrong_sign} positions",
            "Network may not understand basic material advantage",
        )

    if value_range < 0.2:
        results.add_issue(
            "HIGH",
            "wdl_head",
            f"WDL head has very narrow range ({value_range:.4f})",
            "Cannot distinguish between winning and losing positions",
        )

    score = passed / len(test_positions)  # Progressive score
    test_passed = score >= 0.6  # More lenient threshold

    print(subheader("Scoring Philosophy"))
    print(f"  {Colors.CYAN}This test accepts nuanced evaluations:{Colors.ENDC}")
    print(f"  - Small advantages (1 pawn) can have flexible evaluation")
    print(f"  - Large advantages should show clear direction")
    print(f"  - Neutral positions should be close to 0")

    if not test_passed:
        results.add_recommendation(
            1,
            "Review WDL head training",
            f"Score {score*100:.0f}% ({correct}/{len(test_positions)} fully correct)",
        )

    print(f"\n  Progressive score: {score*100:.1f}%")

    results.add("Material Evaluation", test_passed, score, 1.0)
    return score


def _evaluate_position(value, material, tolerance):
    """
    Evaluate a single position with tolerance for nuance.

    Returns: (score, status_string, is_correct, has_wrong_sign)
    """
    is_correct = False
    has_wrong_sign = False

    if tolerance == "strict":
        # Equal position - should be close to 0
        if abs(value) < 0.08:
            return 1.0, "Perfect", True, False
        elif abs(value) < 0.15:
            return 0.85, "Good", True, False
        elif abs(value) < 0.25:
            return 0.6, "Acceptable", False, False
        elif abs(value) < 0.4:
            return 0.3, "Off", False, False
        else:
            return 0.1, "Way off", False, True

    elif tolerance == "flexible":
        # Small material difference - more nuance accepted
        if material > 0:
            if value > 0.05:
                return 1.0, "Correct", True, False
            elif value > -0.05:
                return 0.7, "Neutral OK", True, False  # Small edge, neutral acceptable
            elif value > -0.15:
                return 0.4, "Weak", False, False
            else:
                return 0.1, "Wrong", False, True
        elif material < 0:
            if value < -0.05:
                return 1.0, "Correct", True, False
            elif value < 0.05:
                return 0.7, "Neutral OK", True, False
            elif value < 0.15:
                return 0.4, "Weak", False, False
            else:
                return 0.1, "Wrong", False, True

    else:  # "normal" tolerance
        # Clear material difference - should show direction but nuance OK
        if material > 0:
            # Expected positive value
            if value > 0.15:
                return 1.0, "Clear", True, False
            elif value > 0.05:
                return 0.85, "Good", True, False
            elif value > -0.05:
                return 0.5, "Weak", False, False  # Near neutral
            elif value > -0.15:
                return 0.25, "Wrong dir", False, False
            else:
                return 0.0, "Inverted", False, True
        elif material < 0:
            # Expected negative value
            if value < -0.15:
                return 1.0, "Clear", True, False
            elif value < -0.05:
                return 0.85, "Good", True, False
            elif value < 0.05:
                return 0.5, "Weak", False, False
            elif value < 0.15:
                return 0.25, "Wrong dir", False, False
            else:
                return 0.0, "Inverted", False, True

    return 0.5, "Unknown", False, False
