"""Test: Basic Endgame Understanding."""

import chess

from ..core import (
    TestResults,
    header,
    subheader,
    ok,
    fail,
    warn,
    predict_for_board,
)


def test_endgame(network, results: TestResults):
    """Test basic endgame understanding."""
    print(header("TEST: Endgame Understanding"))

    test_positions = [
        # === BASIC PIECE ENDGAMES (4) ===
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
            "name": "K+Q vs K (Black winning)",
            "fen": "8/8/8/4K3/8/8/8/q3k3 w - - 0 1",
            "expected_eval": "negative",
            "material_diff": -9,
        },
        # === PAWN ENDGAMES (4) ===
        {
            "name": "K+P vs K (advanced pawn)",
            "fen": "8/4P3/8/4k3/8/8/8/4K3 w - - 0 1",
            "expected_eval": "positive",
            "material_diff": 1,
        },
        {
            "name": "K+P vs K (center pawn)",
            "fen": "8/8/8/4k3/4P3/8/8/4K3 w - - 0 1",
            "expected_eval": "positive",
            "material_diff": 1,
        },
        {
            "name": "K+PP vs K (two pawns)",
            "fen": "8/8/8/4k3/3PP3/8/8/4K3 w - - 0 1",
            "expected_eval": "positive",
            "material_diff": 2,
        },
        {
            "name": "K vs K+P (Black pawn)",
            "fen": "8/8/8/4K3/8/4p3/8/4k3 w - - 0 1",
            "expected_eval": "negative",
            "material_diff": -1,
        },
        # === ROOK ENDGAMES (2) ===
        {
            "name": "R+P vs R (Lucena-like)",
            "fen": "1K6/1P6/8/8/8/8/r7/2R1k3 w - - 0 1",
            "expected_eval": "positive",
            "material_diff": 1,
        },
        {
            "name": "R vs R (equal)",
            "fen": "8/8/8/4k3/8/8/R7/4K2r w - - 0 1",
            "expected_eval": "neutral",
            "material_diff": 0,
        },
        # === MINOR PIECE ENDGAMES (2) ===
        {
            "name": "K+B+N vs K (White winning)",
            "fen": "8/8/8/4k3/8/8/8/B3KN2 w - - 0 1",
            "expected_eval": "positive",
            "material_diff": 6,
        },
        {
            "name": "K+B vs K+B (opposite colors)",
            "fen": "8/8/8/4k3/8/8/2b5/B3K3 w - - 0 1",
            "expected_eval": "neutral",
            "material_diff": 0,
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

        # Progressive scoring based on how close the evaluation is
        if expected == "positive":
            if value > 0.2:
                status = ok("")
                passed += 1.0
            elif value > 0.05:
                status = warn("")
                passed += 0.7
            elif value > -0.1:
                status = warn("")
                passed += 0.3
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
        elif expected == "negative":
            if value < -0.2:
                status = ok("")
                passed += 1.0
            elif value < -0.05:
                status = warn("")
                passed += 0.7
            elif value < 0.1:
                status = warn("")
                passed += 0.3
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
        else:  # neutral
            if abs(value) < 0.15:
                status = ok("")
                passed += 1.0
            elif abs(value) < 0.3:
                status = warn("")
                passed += 0.7
            elif abs(value) < 0.5:
                status = warn("")
                passed += 0.3
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
                    "wdl_head",
                    f"WDL head gives wrong sign for {err['position']}",
                    f"Expected positive (material {err['material']:+d}), got {err['actual']:+.4f}",
                )
            elif err["expected"] == "negative" and err["actual"] >= 0:
                results.add_issue(
                    "CRITICAL",
                    "wdl_head",
                    f"WDL head gives wrong sign for {err['position']}",
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
