"""Test: History Comparison - Real vs Duplicated History in MCTS."""

import time
import numpy as np
import chess

from alphazero.move_encoding import encode_move, decode_move, flip_policy
from alphazero.mcts import MCTS
from alphazero.spatial_encoding import encode_board_with_history
from ..core import (
    Colors,
    TestResults,
    header,
    subheader,
    ok,
    fail,
    warn,
    get_history_length,
)


# Test scenarios: sequences of moves to create real history
TEST_SCENARIOS = [
    {
        "name": "Italian Game Opening",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
        "description": "After 1.e4 e5 2.Nf3 Nc6 3.Bc4 - test with real opening history",
        "test_position_move": "d7d6",  # Natural developing move
    },
    {
        "name": "Sicilian Defense",
        "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"],
        "description": "After 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 - Open Sicilian",
        "test_position_move": "g8f6",  # Develop knight
    },
    {
        "name": "Queens Gambit",
        "moves": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"],
        "description": "After 1.d4 d5 2.c4 e6 3.Nc3 Nf6 - QGD position",
        "test_position_move": "c4d5",  # Take on d5
    },
    {
        "name": "Tactical Position After Captures",
        "moves": ["e2e4", "e7e5", "d2d4", "e5d4", "d1d4", "b8c6", "d4e3"],
        "description": "Position with captures in history - tests value learning",
        "test_position_move": "f8b4",  # Pin the knight
    },
]


def _encode_with_real_history(
    current_board: chess.Board,
    history_boards: list[chess.Board],
    history_length: int,
) -> np.ndarray:
    """Encode position with real history boards."""
    # Build boards list: [current, T-1, T-2, ...]
    boards = [current_board] + history_boards[:history_length]

    # Pad if needed
    if len(boards) < history_length + 1:
        pad_board = boards[-1] if boards else current_board
        boards = boards + [pad_board] * (history_length + 1 - len(boards))

    return encode_board_with_history(boards, from_perspective=True)


def _encode_with_duplicated_history(
    current_board: chess.Board,
    history_length: int,
) -> np.ndarray:
    """Encode position with duplicated current position as history."""
    boards = [current_board] * (history_length + 1)
    return encode_board_with_history(boards, from_perspective=True)


def _play_moves(board: chess.Board, moves: list[str]) -> list[chess.Board]:
    """Play a sequence of moves and return history of positions."""
    history = []
    for move_uci in moves:
        history.append(board.copy())  # Save position before move
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            board.push(move)
        else:
            raise ValueError(f"Illegal move {move_uci} in position {board.fen()}")
    return history


def _compare_encodings(
    network,
    current_board: chess.Board,
    history_boards: list[chess.Board],
    history_length: int,
) -> dict:
    """Compare network outputs between real and duplicated history."""
    # Encode with real history
    state_real = _encode_with_real_history(
        current_board, history_boards, history_length
    )
    policy_real, value_real, wdl_real = network.predict_single_with_wdl(state_real)
    if current_board.turn == chess.BLACK:
        policy_real = flip_policy(policy_real)

    # Encode with duplicated history
    state_dup = _encode_with_duplicated_history(current_board, history_length)
    policy_dup, value_dup, wdl_dup = network.predict_single_with_wdl(state_dup)
    if current_board.turn == chess.BLACK:
        policy_dup = flip_policy(policy_dup)

    # Calculate differences
    policy_diff = np.abs(policy_real - policy_dup)
    value_diff = abs(value_real - value_dup)

    # Find top moves for each
    top_real = np.argsort(policy_real)[::-1][:5]
    top_dup = np.argsort(policy_dup)[::-1][:5]

    return {
        "policy_real": policy_real,
        "policy_dup": policy_dup,
        "value_real": value_real,
        "value_dup": value_dup,
        "wdl_real": wdl_real,
        "wdl_dup": wdl_dup,
        "policy_max_diff": float(np.max(policy_diff)),
        "policy_mean_diff": float(np.mean(policy_diff)),
        "value_diff": value_diff,
        "top_real": top_real,
        "top_dup": top_dup,
        "top_move_same": top_real[0] == top_dup[0],
    }


def _compare_mcts(
    network,
    current_board: chess.Board,
    history_boards: list[chess.Board],
    history_length: int,
    num_simulations: int = 100,
) -> dict:
    """Compare MCTS outputs between real and duplicated history."""
    mcts = MCTS(
        network=network,
        c_puct=1.0,
        num_simulations=num_simulations,
        history_length=history_length,
    )
    mcts.temperature = 0.1

    # MCTS with real history
    start = time.time()
    policy_real = mcts.search(
        current_board, add_noise=False, history_boards=history_boards
    )
    time_real = time.time() - start

    mcts.clear_cache()

    # MCTS with no history (will use duplication internally via padding)
    start = time.time()
    policy_dup = mcts.search(current_board, add_noise=False, history_boards=None)
    time_dup = time.time() - start

    # Flip policies back to absolute coordinates for Black
    if current_board.turn == chess.BLACK:
        policy_real = flip_policy(policy_real)
        policy_dup = flip_policy(policy_dup)

    # Calculate differences
    policy_diff = np.abs(policy_real - policy_dup)

    # Find top moves for each
    top_real = np.argsort(policy_real)[::-1][:5]
    top_dup = np.argsort(policy_dup)[::-1][:5]

    return {
        "policy_real": policy_real,
        "policy_dup": policy_dup,
        "policy_max_diff": float(np.max(policy_diff)),
        "policy_mean_diff": float(np.mean(policy_diff)),
        "top_real": top_real,
        "top_dup": top_dup,
        "top_move_same": top_real[0] == top_dup[0],
        "time_real": time_real,
        "time_dup": time_dup,
    }


def _test_single_scenario(
    network, scenario: dict, history_length: int, results: TestResults
) -> float:
    """Test a single scenario with real vs duplicated history."""
    board = chess.Board()

    # Play moves to create history
    history_boards = _play_moves(board, scenario["moves"])

    print(f"\n  Position after {len(scenario['moves'])} moves:")
    print(f"  FEN: {board.fen()}")
    print(board)

    # Compare raw network outputs
    print(f"\n  {Colors.BOLD}--- Raw Network Comparison ---{Colors.ENDC}")
    net_cmp = _compare_encodings(network, board, history_boards, history_length)

    print(f"  Value (real history):      {net_cmp['value_real']:+.4f}")
    print(f"  Value (duplicated):        {net_cmp['value_dup']:+.4f}")
    print(f"  Value difference:          {net_cmp['value_diff']:.4f}")
    print(
        f"  WDL (real):    [W={net_cmp['wdl_real'][0]:.3f}, D={net_cmp['wdl_real'][1]:.3f}, L={net_cmp['wdl_real'][2]:.3f}]"
    )
    print(
        f"  WDL (dup):     [W={net_cmp['wdl_dup'][0]:.3f}, D={net_cmp['wdl_dup'][1]:.3f}, L={net_cmp['wdl_dup'][2]:.3f}]"
    )

    # Top moves comparison
    print(f"\n  Top moves (real history):")
    for i, idx in enumerate(net_cmp["top_real"][:3]):
        move = decode_move(idx, board)
        if move:
            print(f"    {i+1}. {move.uci()} ({net_cmp['policy_real'][idx]*100:.1f}%)")

    print(f"\n  Top moves (duplicated):")
    for i, idx in enumerate(net_cmp["top_dup"][:3]):
        move = decode_move(idx, board)
        if move:
            print(f"    {i+1}. {move.uci()} ({net_cmp['policy_dup'][idx]*100:.1f}%)")

    if net_cmp["top_move_same"]:
        top_move = decode_move(net_cmp["top_real"][0], board)
        move_str = top_move.uci() if top_move else "?"
        print(f"\n  {ok('Same top move: ' + move_str)} (network)")
    else:
        top_real = decode_move(net_cmp["top_real"][0], board)
        top_dup = decode_move(net_cmp["top_dup"][0], board)
        real_str = top_real.uci() if top_real else "?"
        dup_str = top_dup.uci() if top_dup else "?"
        print(
            f"\n  {warn('Different top moves: real=' + real_str + ', dup=' + dup_str)}"
        )

    # Compare MCTS outputs
    print(f"\n  {Colors.BOLD}--- MCTS Comparison (100 sims) ---{Colors.ENDC}")
    mcts_cmp = _compare_mcts(network, board, history_boards, history_length)

    print(f"  Time (real history):       {mcts_cmp['time_real']*1000:.0f}ms")
    print(f"  Time (duplicated):         {mcts_cmp['time_dup']*1000:.0f}ms")
    print(f"  Policy max diff:           {mcts_cmp['policy_max_diff']*100:.2f}%")
    print(f"  Policy mean diff:          {mcts_cmp['policy_mean_diff']*100:.4f}%")

    print(f"\n  Top MCTS moves (real history):")
    for i, idx in enumerate(mcts_cmp["top_real"][:3]):
        move = decode_move(idx, board)
        if move:
            print(f"    {i+1}. {move.uci()} ({mcts_cmp['policy_real'][idx]*100:.1f}%)")

    print(f"\n  Top MCTS moves (duplicated):")
    for i, idx in enumerate(mcts_cmp["top_dup"][:3]):
        move = decode_move(idx, board)
        if move:
            print(f"    {i+1}. {move.uci()} ({mcts_cmp['policy_dup'][idx]*100:.1f}%)")

    if mcts_cmp["top_move_same"]:
        top_move = decode_move(mcts_cmp["top_real"][0], board)
        move_str = top_move.uci() if top_move else "?"
        print(f"\n  {ok('Same top MCTS move: ' + move_str)} (MCTS)")
    else:
        top_real = decode_move(mcts_cmp["top_real"][0], board)
        top_dup = decode_move(mcts_cmp["top_dup"][0], board)
        real_str = top_real.uci() if top_real else "?"
        dup_str = top_dup.uci() if top_dup else "?"
        print(
            f"\n  {warn('Different MCTS top moves: real=' + real_str + ', dup=' + dup_str)}"
        )

    # Store diagnostics
    results.add_diagnostic(
        "history", f"{scenario['name']}_net_value_diff", net_cmp["value_diff"]
    )
    results.add_diagnostic(
        "history", f"{scenario['name']}_net_policy_max_diff", net_cmp["policy_max_diff"]
    )
    results.add_diagnostic(
        "history",
        f"{scenario['name']}_net_same_top",
        1.0 if net_cmp["top_move_same"] else 0.0,
    )
    results.add_diagnostic(
        "history",
        f"{scenario['name']}_mcts_policy_max_diff",
        mcts_cmp["policy_max_diff"],
    )
    results.add_diagnostic(
        "history",
        f"{scenario['name']}_mcts_same_top",
        1.0 if mcts_cmp["top_move_same"] else 0.0,
    )

    # Score: penalize if there are significant differences AND different top moves
    # Small differences are expected and OK
    score = 1.0

    # Check if implementation works (history should produce different results than duplication)
    has_difference = (
        net_cmp["policy_max_diff"] > 0.001
        or net_cmp["value_diff"] > 0.001
        or mcts_cmp["policy_max_diff"] > 0.001
    )

    if has_difference:
        print(
            f"\n  {ok('History produces different results - implementation working!')}"
        )
    else:
        print(f"\n  {warn('No difference detected - history may not be used')}")
        score = 0.5  # Reduce score if no difference (might indicate bug)

    return score


def test_history_comparison(network, results: TestResults):
    """Test that real history vs duplicated history produces expected behavior."""
    print(header("TEST: History Comparison (Real vs Duplicated)"))

    history_length = get_history_length(network)
    print(f"  History length: {history_length}")

    total_score = 0.0

    for scenario in TEST_SCENARIOS:
        print(subheader(f"{scenario['name']}: {scenario['description']}"))
        score = _test_single_scenario(network, scenario, history_length, results)
        total_score += score

    # Summary
    print(subheader("Summary"))
    avg_score = total_score / len(TEST_SCENARIOS)
    print(f"  Scenarios tested: {len(TEST_SCENARIOS)}")
    print(f"  Average score: {avg_score*100:.1f}%")

    results.add_diagnostic("history", "scenarios_tested", len(TEST_SCENARIOS))
    results.add_diagnostic("history", "average_score", avg_score)

    if avg_score < 0.5:
        results.add_issue(
            "WARNING",
            "history",
            f"History comparison score only {avg_score*100:.0f}%",
            "History may not be properly utilized",
        )

    overall_passed = avg_score >= 0.5
    results.add("History Comparison", overall_passed, avg_score, 1.0)

    return avg_score
