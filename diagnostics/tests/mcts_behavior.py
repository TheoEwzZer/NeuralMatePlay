"""Test: MCTS Behavior."""

from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

import numpy as np
import chess

from alphazero.move_encoding import encode_move, decode_move
from alphazero.mcts import MCTS
from ..core import (
    Colors,
    TestResults,
    header,
    subheader,
    ok,
    fail,
    warn,
    encode_for_network,
    get_history_length,
)

if TYPE_CHECKING:
    from src.alphazero.network import DualHeadNetwork


# Test positions for MCTS behavior
TEST_POSITIONS: list[dict[str, Any]] = [
    # === OBVIOUS CAPTURES (2) ===
    {
        "name": "Hanging Queen",
        "fen": "rnb1kbnr/pppppppp/8/8/3q4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
        "description": "Black queen hanging on d4",
        "expected_move": "e2d4",  # or any queen capture
        "test_type": "capture_queen",
        "simulations": 100,
    },
    {
        "name": "Hanging Rook",
        "fen": "rnbqkbnr/pppppppp/8/8/3r4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
        "description": "Black rook hanging on d4",
        "expected_move": "e2d4",
        "test_type": "capture_rook",
        "simulations": 100,
    },
    # === MATE IN 1 (2) ===
    {
        "name": "Back Rank Mate",
        "fen": "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1",
        "description": "Ra8# is mate",
        "expected_move": "a1a8",
        "test_type": "mate_in_1",
        "simulations": 100,
    },
    {
        "name": "Queen Mate",
        "fen": "k7/8/1K6/8/8/8/8/Q7 w - - 0 1",
        "description": "Qa7# or Qa8# is mate",
        "expected_move": "a1a7",  # or a1a8
        "test_type": "mate_in_1",
        "simulations": 100,
    },
    # === TACTICAL POSITIONS (2) ===
    {
        "name": "Fork Setup",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "description": "Ng5 attacks f7",
        "expected_move": "f3g5",
        "test_type": "tactic",
        "simulations": 200,
    },
    {
        "name": "Discovered Attack",
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
        "description": "Bxf7+ sacrifice",
        "expected_move": "c4f7",
        "test_type": "tactic",
        "simulations": 200,
    },
]


def _find_best_capture(board: chess.Board, piece_type: int) -> chess.Move | None:
    """Find move that captures a specific piece type."""
    for move in board.legal_moves:
        if board.is_capture(move):
            captured: chess.Piece | None = board.piece_at(move.to_square)
            if captured and captured.piece_type == piece_type:
                return move
    return None


def _find_mating_move(board: chess.Board) -> chess.Move | None:
    """Find a move that delivers checkmate."""
    for move in board.legal_moves:
        board.push(move)
        is_mate: bool = board.is_checkmate()
        board.pop()
        if is_mate:
            return move
    return None


def _test_single_position(
    board: chess.Board,
    network: DualHeadNetwork,
    mcts: MCTS,
    test: dict[str, Any],
    results: TestResults,
) -> tuple[float, float]:
    """Test MCTS on a single position. Returns (score, details) with progressive scoring."""
    simulations: int = test.get("simulations", 100)

    # Get raw network output first
    state: np.ndarray = encode_for_network(board, network)
    raw_policy: np.ndarray
    raw_value: float
    raw_policy, raw_value = network.predict_single(state)
    raw_top_idx: int = int(np.argmax(raw_policy))
    raw_top_move: chess.Move | None = decode_move(raw_top_idx, board)

    print(
        f"  Raw policy top: {raw_top_move.uci() if raw_top_move else 'None'} ({raw_policy[raw_top_idx]*100:.1f}%)"
    )
    print(f"  Raw value: {raw_value:+.4f}")

    # Run MCTS
    start: float = time.time()
    mcts.num_simulations = simulations
    policy: np.ndarray = mcts.search(board, add_noise=False)
    elapsed: float = time.time() - start

    top_indices: np.ndarray = np.argsort(policy)[::-1][:5]
    mcts_top_move: chess.Move | None = decode_move(top_indices[0], board)

    print(f"\n  MCTS ({simulations} sims, {elapsed*1000:.0f}ms):")
    print(f"  {'Rank':<6} {'Move':<8} {'Visits':>10}")
    print("  " + "-" * 30)

    for i, idx in enumerate(top_indices[:5]):
        move: chess.Move | None = decode_move(idx, board)
        visits: float = policy[idx]
        if move:
            print(f"  {i+1:<6} {move.uci():<8} {visits*100:>9.1f}%")

    # Check if MCTS found the expected move with progressive scoring
    position_score: float = 0.0
    test_type: str = test["test_type"]

    if test_type == "capture_queen":
        expected: chess.Move | None = _find_best_capture(board, chess.QUEEN)
        found_rank: int | None = (
            _find_move_rank(expected, top_indices, board) if expected else None
        )
        if found_rank == 1:
            position_score = 1.0
            print(f"\n  {ok(f'MCTS captures queen: {mcts_top_move.uci()}')}")
        elif found_rank == 2:
            position_score = 0.7
            print(f"\n  {warn(f'Queen capture at rank 2')}")
        elif found_rank == 3:
            position_score = 0.5
            print(f"\n  {warn(f'Queen capture at rank 3')}")
        elif found_rank:
            position_score = 0.3
            print(f"\n  {warn(f'Queen capture at rank {found_rank}')}")
        else:
            print(
                f"\n  {fail(f'MCTS plays {mcts_top_move.uci()} instead of capturing queen')}"
            )

    elif test_type == "capture_rook":
        expected = _find_best_capture(board, chess.ROOK)
        found_rank = _find_move_rank(expected, top_indices, board) if expected else None
        if found_rank == 1:
            position_score = 1.0
            print(f"\n  {ok(f'MCTS captures rook: {mcts_top_move.uci()}')}")
        elif found_rank == 2:
            position_score = 0.7
            print(f"\n  {warn(f'Rook capture at rank 2')}")
        elif found_rank == 3:
            position_score = 0.5
            print(f"\n  {warn(f'Rook capture at rank 3')}")
        elif found_rank:
            position_score = 0.3
            print(f"\n  {warn(f'Rook capture at rank {found_rank}')}")
        else:
            print(
                f"\n  {fail(f'MCTS plays {mcts_top_move.uci()} instead of capturing rook')}"
            )

    elif test_type == "mate_in_1":
        # Check all top moves for mate
        mate_rank: int | None = None
        for i, idx in enumerate(top_indices[:5]):
            move = decode_move(idx, board)
            if move:
                board.push(move)
                if board.is_checkmate():
                    mate_rank = i + 1
                board.pop()
                if mate_rank:
                    break

        if mate_rank == 1:
            position_score = 1.0
            print(f"\n  {ok(f'MCTS finds mate: {mcts_top_move.uci()}')}")
        elif mate_rank == 2:
            position_score = 0.7
            print(f"\n  {warn(f'Mate found at rank 2')}")
        elif mate_rank == 3:
            position_score = 0.5
            print(f"\n  {warn(f'Mate found at rank 3')}")
        elif mate_rank:
            position_score = 0.3
            print(f"\n  {warn(f'Mate found at rank {mate_rank}')}")
        else:
            print(f"\n  {fail(f'MCTS plays {mcts_top_move.uci()} instead of mating')}")

    elif test_type == "tactic":
        expected_uci: str = test["expected_move"]
        # Find rank of expected move
        found_rank = None
        for i, idx in enumerate(top_indices[:5]):
            move = decode_move(idx, board)
            if move and move.uci() == expected_uci:
                found_rank = i + 1
                break

        if found_rank == 1:
            position_score = 1.0
            print(f"\n  {ok(f'MCTS finds tactic: {mcts_top_move.uci()}')}")
        elif found_rank == 2:
            position_score = 0.75
            print(f"\n  {warn(f'Tactic at rank 2 (top is {mcts_top_move.uci()})')}")
        elif found_rank == 3:
            position_score = 0.5
            print(f"\n  {warn(f'Tactic at rank 3 (top is {mcts_top_move.uci()})')}")
        elif found_rank:
            position_score = 0.25
            print(
                f"\n  {warn(f'Tactic at rank {found_rank} (top is {mcts_top_move.uci()})')}"
            )
        else:
            print(
                f"\n  {fail(f'MCTS plays {mcts_top_move.uci()} instead of {expected_uci}')}"
            )

    # Store diagnostics
    results.add_diagnostic("mcts", f"{test['name']}_score", position_score)
    results.add_diagnostic("mcts", f"{test['name']}_time_ms", elapsed * 1000)
    results.add_diagnostic(
        "mcts", f"{test['name']}_top_visits", float(policy[top_indices[0]])
    )

    return position_score, elapsed


def _find_move_rank(
    expected_move: chess.Move | None, top_indices: np.ndarray, board: chess.Board
) -> int | None:
    """Find the rank of expected_move in top_indices."""
    if not expected_move:
        return None
    for i, idx in enumerate(top_indices[:5]):
        move: chess.Move | None = decode_move(idx, board)
        if move == expected_move:
            return i + 1
    return None


def test_mcts_behavior(network: DualHeadNetwork, results: TestResults) -> float:
    """Test MCTS behavior on multiple positions."""
    print(header("TEST: MCTS Behavior"))

    history_length: int = get_history_length(network)
    mcts: MCTS = MCTS(
        network=network, c_puct=1.0, num_simulations=100, history_length=history_length
    )
    mcts.temperature = 0.1

    total_score: float = 0.0
    total_time: float = 0
    full_passes: int = 0

    for test in TEST_POSITIONS:
        board: chess.Board = chess.Board(test["fen"])
        print(subheader(f"{test['name']}: {test['description']}"))
        print(board)

        position_score: float
        elapsed: float
        position_score, elapsed = _test_single_position(
            board, network, mcts, test, results
        )
        total_score += position_score
        if position_score >= 1.0:
            full_passes += 1
        total_time += elapsed

    # Summary
    print(subheader("Summary"))
    score: float = total_score / len(TEST_POSITIONS)
    print(f"  Positions tested: {len(TEST_POSITIONS)}")
    print(f"  Full passes: {full_passes}/{len(TEST_POSITIONS)}")
    print(f"  Progressive score: {score*100:.1f}%")
    print(f"  Total search time: {total_time*1000:.0f}ms")
    print(f"  Avg time per position: {total_time/len(TEST_POSITIONS)*1000:.0f}ms")

    results.add_diagnostic("mcts", "positions_tested", len(TEST_POSITIONS))
    results.add_diagnostic("mcts", "full_passes", full_passes)
    results.add_diagnostic("mcts", "progressive_score", score)
    results.add_diagnostic("mcts", "total_time_ms", total_time * 1000)

    if score < 0.5:
        results.add_issue(
            "CRITICAL",
            "mcts",
            f"MCTS score only {score*100:.0f}% ({full_passes}/{len(TEST_POSITIONS)} full passes)",
            "Value head may not guide search correctly",
        )

    overall_passed: bool = score >= 0.5
    results.add("MCTS Behavior", overall_passed, score, 1.0)
    results.add_timing("MCTS Total", total_time)

    return score
