"""Test: WDL Head Material Evaluation."""

import numpy as np
import chess

from ..core import (
    Colors,
    TestResults,
    header,
    subheader,
    ok,
    fail,
    encode_for_network,
)


def test_wdl_head(network, results: TestResults):
    """Test if the WDL head correctly evaluates material imbalances."""
    print(header("TEST: WDL Head Material Evaluation"))

    # Test positions with BOTH White to move (WTM) and Black to move (BTM)
    # Value is always from CURRENT PLAYER's perspective:
    #   - WTM + White ahead = positive
    #   - BTM + White ahead = negative (Black's POV: losing)
    test_positions = [
        # === Starting positions ===
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Start (WTM)", 0),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "Start (BTM)", 0),
        # === White missing Knight (-3 material for White) ===
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1", "W -N (WTM)", -3),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R b KQkq - 0 1", "W -N (BTM)", +3),
        # === Black missing Knight (+3 material for White) ===
        ("rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "B -N (WTM)", +3),
        ("rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "B -N (BTM)", -3),
        # === White missing Queen (-9 material for White) ===
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1", "W -Q (WTM)", -9),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1", "W -Q (BTM)", +9),
        # === Black missing Queen (+9 material for White) ===
        ("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "B -Q (WTM)", +9),
        ("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "B -Q (BTM)", -9),
        # === Black missing Pawn (+1 material for White) ===
        ("rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "B -P (WTM)", +1),
        ("rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "B -P (BTM)", -1),
    ]

    print(
        f"\n  {'Position':<20} {'Material':>10} {'Value':>10} {'Correct?':>10} {'Error':>10}"
    )
    print("  " + "-" * 65)

    correct = 0
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

    for fen, desc, material in test_positions:
        is_wtm = "(WTM)" in desc
        is_btm = "(BTM)" in desc
        board = chess.Board(fen)
        state = encode_for_network(board, network)
        _, value = network.predict_single(state)
        values.append(value)

        # Determine if sign is correct
        is_correct = False
        has_wrong_sign = False

        if material > 0 and value > 0.05:
            status = ok("")
            is_correct = True
            error = abs(value - material / 10)  # Normalize material to [-1,1] range
        elif material < 0 and value < -0.05:
            status = ok("")
            is_correct = True
            error = abs(value - material / 10)
        elif material == 0 and abs(value) < 0.15:
            status = ok("")
            is_correct = True
            error = abs(value)
        else:
            status = fail("")
            error = abs(value - material / 10)
            if (material > 0 and value <= 0) or (material < 0 and value >= 0):
                has_wrong_sign = True

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

        total_error += error
        print(
            f"  {desc:<20} {material:>+10} {value:>+10.4f} {status:>10} {error:>10.4f}"
        )

        results.add_diagnostic("material_eval", f"{desc}_value", float(value))
        results.add_diagnostic("material_eval", f"{desc}_material", material)

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

    # WTM vs BTM breakdown - KEY for detecting perspective bias
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
                print(
                    f"  {Colors.YELLOW}   -> Network may not generalize well to Black's perspective{Colors.ENDC}"
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

    # WTM/BTM diagnostics
    results.add_diagnostic("material_eval", "wtm_correct", wtm_correct)
    results.add_diagnostic("material_eval", "wtm_total", wtm_total)
    results.add_diagnostic("material_eval", "wtm_wrong_sign", wtm_wrong_sign)
    results.add_diagnostic("material_eval", "btm_correct", btm_correct)
    results.add_diagnostic("material_eval", "btm_total", btm_total)
    results.add_diagnostic("material_eval", "btm_wrong_sign", btm_wrong_sign)

    # Identify issues
    if value_std < 0.1:
        results.add_issue(
            "CRITICAL",
            "wdl_head",
            f"WDL head has very low variance (std={value_std:.4f})",
            "Output is nearly constant regardless of position - training may have collapsed",
        )

    if wrong_sign > 0:
        results.add_issue(
            "CRITICAL",
            "wdl_head",
            f"WDL head gives wrong sign for {wrong_sign} positions",
            "Network doesn't understand basic material advantage",
        )

    if value_range < 0.2:
        results.add_issue(
            "HIGH",
            "wdl_head",
            f"WDL head has very narrow range ({value_range:.4f})",
            "Cannot distinguish between winning and losing positions",
        )

    score = correct / len(test_positions)
    passed = correct >= len(test_positions) * 0.7

    if not passed:
        results.add_recommendation(
            1,
            "URGENT: Fix WDL head training",
            f"Only {correct}/{len(test_positions)} correct, value std={value_std:.4f}",
        )

    results.add("Material Evaluation", passed, score, 1.0)
    return score
