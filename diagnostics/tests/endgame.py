"""Test: Basic Endgame Understanding."""

import chess

from ..core import (
    TestResults,
    header,
    subheader,
    ok,
    fail,
    predict_for_board,
)


def test_endgame(network, results: TestResults):
    """Test basic endgame understanding."""
    print(header("TEST: Endgame Understanding"))

    test_positions = [
        {
            "name": "K+R vs K (White winning)",
            "fen": "8/8/8/4k3/8/8/8/R3K3 w Q - 0 1",
            "expected_eval": "positive",
            "material_diff": 5,
        },
        {
            "name": "K+Q vs K (White winning)",
            "fen": "8/8/8/4k3/8/8/8/Q3K3 w - - 0 1",
            "expected_eval": "positive",
            "material_diff": 9,
        },
        {
            "name": "K+R vs K (Black winning)",
            "fen": "8/8/8/4K3/8/8/8/r3k3 w - - 0 1",
            "expected_eval": "negative",
            "material_diff": -5,
        },
        {
            "name": "K+P vs K (White should win)",
            "fen": "8/4P3/8/4k3/8/8/8/4K3 w - - 0 1",
            "expected_eval": "positive",
            "material_diff": 1,
        },
    ]

    passed = 0
    value_errors = []

    print(
        f"\n  {'Position':<30} {'Material':>10} {'Expected':>10} {'Actual':>10} {'Status':>10}"
    )
    print("  " + "-" * 75)

    for test in test_positions:
        board = chess.Board(test["fen"])
        policy, value = predict_for_board(board, network)

        expected = test["expected_eval"]
        material = test["material_diff"]

        # Determine if evaluation is correct
        if expected == "positive" and value > 0.1:
            status = ok("")
            passed += 1
        elif expected == "negative" and value < -0.1:
            status = ok("")
            passed += 1
        elif expected == "neutral" and abs(value) < 0.2:
            status = ok("")
            passed += 1
        else:
            status = fail("")
            value_errors.append(
                {
                    "position": test["name"],
                    "expected": expected,
                    "actual": value,
                    "material": material,
                }
            )

        exp_str = (
            "+" if expected == "positive" else ("-" if expected == "negative" else "=")
        )
        print(
            f"  {test['name']:<30} {material:>+10} {exp_str:>10} {value:>+10.4f} {status}"
        )

        results.add_diagnostic("endgame", f"{test['name']}_value", float(value))
        results.add_diagnostic("endgame", f"{test['name']}_expected", expected)
        results.add_diagnostic("endgame", f"{test['name']}_material", material)

    print("  " + "-" * 75)

    # Analysis of errors
    if value_errors:
        print(subheader("Value Head Error Analysis"))
        for err in value_errors:
            print(f"  • {err['position']}")
            print(f"    Expected: {err['expected']} (material: {err['material']:+d})")
            print(f"    Actual:   {err['actual']:+.4f}")

            if err["expected"] == "positive" and err["actual"] <= 0:
                results.add_issue(
                    "CRITICAL",
                    "value_head",
                    f"Value head gives wrong sign for {err['position']}",
                    f"Expected positive (material {err['material']:+d}), got {err['actual']:+.4f}",
                )
            elif err["expected"] == "negative" and err["actual"] >= 0:
                results.add_issue(
                    "CRITICAL",
                    "value_head",
                    f"Value head gives wrong sign for {err['position']}",
                    f"Expected negative (material {err['material']:+d}), got {err['actual']:+.4f}",
                )

    score = passed / len(test_positions)
    results.add_diagnostic("endgame", "total_tested", len(test_positions))
    results.add_diagnostic("endgame", "correct_evals", passed)
    results.add(
        "Endgame Understanding", passed >= len(test_positions) * 0.75, score, 1.0
    )

    if score < 0.75:
        results.add_recommendation(
            5,
            "Review value head training - material evaluation broken",
            f"Only {passed}/{len(test_positions)} endgame positions correctly evaluated",
        )

    return score
