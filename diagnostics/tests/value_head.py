"""Test: Value Head Material Evaluation."""

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


def test_value_head(network, results: TestResults):
    """Test if the value head correctly evaluates material imbalances."""
    print(header("TEST: Value Head Material Evaluation"))

    test_positions = [
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "Starting position",
            0,
        ),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1", "White -N", -3),
        ("rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Black -N", +3),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1", "White -Q", -9),
        ("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Black -Q", +9),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1", "White -B", -3),
        ("rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Black -B", +3),
        ("rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Black -P", +1),
    ]

    print(
        f"\n  {'Position':<20} {'Material':>10} {'Value':>10} {'Correct?':>10} {'Error':>10}"
    )
    print("  " + "-" * 65)

    correct = 0
    total_error = 0
    wrong_sign = 0
    values = []

    for fen, desc, material in test_positions:
        board = chess.Board(fen)
        state = encode_for_network(board, network)
        _, value = network.predict_single(state)
        values.append(value)

        # Determine if sign is correct
        if material > 0 and value > 0.05:
            status = ok("")
            correct += 1
            error = abs(value - material / 10)  # Normalize material to [-1,1] range
        elif material < 0 and value < -0.05:
            status = ok("")
            correct += 1
            error = abs(value - material / 10)
        elif material == 0 and abs(value) < 0.15:
            status = ok("")
            correct += 1
            error = abs(value)
        else:
            status = fail("")
            error = abs(value - material / 10)
            if (material > 0 and value <= 0) or (material < 0 and value >= 0):
                wrong_sign += 1

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

    print(subheader("Value Head Statistics"))
    print(f"  Correct evaluations:   {correct}/{len(test_positions)}")
    print(f"  Wrong sign errors:     {wrong_sign}")
    print(f"  Average error:         {avg_error:.4f}")
    print(f"  Value std deviation:   {value_std:.4f}")
    print(f"  Value range:           {value_range:.4f}")
    print(f"  Values: {[f'{v:.3f}' for v in values]}")

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
            "value_head",
            f"Value head has very low variance (std={value_std:.4f})",
            "Output is nearly constant regardless of position - training may have collapsed",
        )

    if wrong_sign > 0:
        results.add_issue(
            "CRITICAL",
            "value_head",
            f"Value head gives wrong sign for {wrong_sign} positions",
            "Network doesn't understand basic material advantage",
        )

    if value_range < 0.2:
        results.add_issue(
            "HIGH",
            "value_head",
            f"Value head has very narrow range ({value_range:.4f})",
            "Cannot distinguish between winning and losing positions",
        )

    score = correct / len(test_positions)
    passed = correct >= len(test_positions) * 0.7

    if not passed:
        results.add_recommendation(
            1,
            "URGENT: Fix value head training",
            f"Only {correct}/{len(test_positions)} correct, value std={value_std:.4f}",
        )

    results.add("Material Evaluation", passed, score, 1.0)
    return score
