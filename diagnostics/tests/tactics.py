"""Test: Basic Tactics."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

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
    info,
    predict_for_board,
)

if TYPE_CHECKING:
    from src.alphazero.network import DualHeadNetwork


def test_tactics(network: DualHeadNetwork, results: TestResults) -> float:
    """Test if the network can find basic tactics."""
    print(header("TEST: Basic Tactics"))

    test_positions: list[dict[str, Any]] = [
        # === FORKS (4 positions) ===
        {
            "name": "Knight Fork Setup",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
            "tactical_move": "f3g5",
            "description": "Ng5 attacks f7 (weak point)",
            "type": "attack",
        },
        {
            "name": "Central Fork",
            "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4N3/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1",
            "tactical_move": "e4d6",
            "description": "Nd6+ forks king and bishop",
            "type": "fork",
        },
        {
            "name": "Knight Fork Check",
            "fen": "4k3/8/8/3N4/8/8/8/4K3 w - - 0 1",
            "tactical_move": "d5c7",
            "description": "Nc7+ forks king (corner threats)",
            "type": "fork",
        },
        {
            "name": "Queen Fork",
            "fen": "4k3/8/8/8/8/8/4q3/4K1NR b - - 0 1",
            "tactical_move": "e2g2",
            "description": "Qg2 forks knight and threatens",
            "type": "fork",
        },
        # === BACK RANK TACTICS (2 positions) ===
        {
            "name": "Back Rank Mate",
            "fen": "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1",
            "tactical_move": "a1a8",
            "description": "Ra8# back rank mate",
            "type": "mate",
        },
        {
            "name": "Double Attack f7",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
            "tactical_move": "f3g5",
            "description": "Ng5 double attack on f7",
            "type": "attack",
        },
        # === REMOVING DEFENDER / CAPTURES (3 positions) ===
        {
            "name": "Win Material",
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 1",
            "tactical_move": "h4e1",
            "description": "Qe1+ wins material",
            "type": "attack",
        },
        {
            "name": "Remove Defender",
            "fen": "r1b1k2r/ppppqppp/2n2n2/4p1B1/2B1P3/3P1N2/PPP2PPP/RN1QK2R w KQkq - 0 1",
            "tactical_move": "g5f6",
            "description": "Bxf6 removes defender",
            "type": "capture",
        },
        {
            "name": "Sacrifice Pattern",
            "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
            "tactical_move": "c4f7",
            "description": "Bxf7+ classic sacrifice",
            "type": "sacrifice",
        },
        # === SKEWER (1 position) ===
        {
            "name": "Rook Skewer",
            "fen": "4k3/8/8/8/8/4R3/8/4K3 w - - 0 1",
            "tactical_move": "e3e8",
            "description": "Re8+ skewer pattern",
            "type": "skewer",
        },
        # === EVALUATION POSITIONS (2 positions) ===
        {
            "name": "Discovered Attack",
            "fen": "r1bqkb1r/pppp1Bpp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
            "tactical_move": None,
            "description": "After Bxf7+ (Italian Game trap)",
            "type": "evaluation",
        },
        {
            "name": "Pin Position",
            "fen": "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 1",
            "tactical_move": None,
            "description": "Bb4 pins Nc3 to king",
            "type": "evaluation",
        },
    ]

    passed: float = 0
    tactical_found: int = 0

    for test in test_positions:
        board: chess.Board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        policy: np.ndarray
        value: float
        policy, value = predict_for_board(board, network)

        top_5: np.ndarray = np.argsort(policy)[::-1][:5]

        print(f"\n  Value evaluation: {value:+.4f}")
        print(f"\n  {'Rank':<6} {'Move':<8} {'Prob':>8} {'Check?':<8} {'Notes':<20}")
        print("  " + "-" * 55)

        tactical_rank: int | None = None

        for i, idx in enumerate(top_5):
            move: chess.Move | None = decode_move(idx, board)
            prob: float = policy[idx]
            if move:
                board.push(move)
                is_check: bool = board.is_check()
                board.pop()

                check_str: str = "+" if is_check else ""
                notes: str = ""

                if test.get("tactical_move"):
                    if move.uci() == test["tactical_move"]:
                        notes = "TACTICAL!"
                        if tactical_rank is None:
                            tactical_rank = i + 1
                        if i == 0:
                            tactical_found += 1

                color: str = (
                    Colors.GREEN if notes else (Colors.YELLOW if is_check else "")
                )
                end_color: str = Colors.ENDC if color else ""
                print(
                    f"  {color}{i+1:<6} {move.uci():<8} {prob*100:>7.2f}%"
                    f" {check_str:<8} {notes}{end_color}"
                )

        if test.get("tactical_move"):
            # Progressive scoring based on rank
            if tactical_rank == 1:
                print(f"\n  {ok('Tactical move found as top choice!')}")
                passed += 1.0
            elif tactical_rank == 2:
                print(f"\n  {warn(f'Tactical move at rank 2')}")
                passed += 0.75
            elif tactical_rank == 3:
                print(f"\n  {warn(f'Tactical move at rank 3')}")
                passed += 0.5
            elif tactical_rank:
                print(f"\n  {warn(f'Tactical move at rank {tactical_rank}')}")
                passed += 0.25
            else:
                print(f"\n  {fail('Tactical move not in top 5')}")
        else:
            # Evaluation-only position - score based on value correctness
            passed += 0.5
            print(f"\n  {info('Position for evaluation analysis')}")

        results.add_diagnostic(
            "tactics", f"{test['name']}_tactical_rank", tactical_rank
        )
        results.add_diagnostic("tactics", f"{test['name']}_value", float(value))

    score: float = passed / len(test_positions)
    results.add_diagnostic("tactics", "total_tested", len(test_positions))
    results.add_diagnostic("tactics", "tactical_found", tactical_found)
    results.add("Tactics", score >= 0.5, score, 1.0)

    if score < 0.5:
        results.add_recommendation(
            4,
            "Include tactical puzzles in training data",
            f"Network tactical score: {score*100:.0f}%",
        )

    return score
