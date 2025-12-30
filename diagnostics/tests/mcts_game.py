"""Test: Game vs Pure MCTS Player."""

import time
import chess

from alphazero.arena import NetworkPlayer, PureMCTSPlayer
from chess_encoding.board_utils import get_raw_material_diff
from ..core import (
    TestResults,
    header,
    subheader,
    ok,
    fail,
    warn,
    get_history_length,
)


def test_mcts_game(network, results: TestResults, mcts_simulations: int = 200):
    """
    Play a game vs pure MCTS to test network strength.

    Pure MCTS uses random rollouts without neural network guidance.
    A trained network should beat this opponent consistently.
    """
    print(header("TEST: Game vs Pure MCTS"))

    history_length = get_history_length(network)
    network_player = NetworkPlayer(
        network, num_simulations=100, name="Network", history_length=history_length
    )
    mcts_player = PureMCTSPlayer(
        num_simulations=mcts_simulations,
        max_rollout_depth=30,
        name=f"PureMCTS({mcts_simulations})"
    )

    board = chess.Board()
    move_count = 0
    move_log = []

    print(subheader(f"Playing as White (100 sims) vs MCTS ({mcts_simulations} sims)"))

    start_time = time.time()

    while not board.is_game_over() and move_count < 80:
        if board.turn == chess.WHITE:
            move = network_player.select_move(board)
            player = "Network"
        else:
            move = mcts_player.select_move(board)
            player = "MCTS"

        if move is None:
            break

        # Log capture
        is_capture = board.is_capture(move)
        captured = ""
        if is_capture and board.piece_at(move.to_square):
            piece_names = {1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}
            captured = (
                f"x{piece_names.get(board.piece_at(move.to_square).piece_type, '?')}"
            )

        move_log.append(
            {
                "num": move_count + 1,
                "player": player,
                "move": move.uci(),
                "capture": captured,
            }
        )

        board.push(move)
        move_count += 1

    game_time = time.time() - start_time

    # Print move log (compact)
    print(f"\n  Move log (first 20 of {move_count}):")
    print("  " + "-" * 50)
    for entry in move_log[:20]:
        side = "W" if entry["player"] == "Network" else "B"
        cap = entry["capture"]
        print(f"  {entry['num']:>3}. [{side}] {entry['move']:<6} {cap}")

    if move_count > 20:
        print(f"  ... ({move_count - 20} more moves)")

    print("  " + "-" * 50)
    print(f"\n  Final position after {move_count} moves:")
    print(board)

    # Analysis
    material = get_raw_material_diff(board)

    print(subheader("Game Analysis"))
    print(f"  Total moves: {move_count}")
    print(f"  Game time: {game_time:.1f}s")
    print(f"  Avg time per move: {game_time/max(1, move_count)*1000:.0f}ms")
    print(f"  Material balance: {material:+d}")

    results.add_diagnostic("mcts_game", "total_moves", move_count)
    results.add_diagnostic("mcts_game", "material_diff", material)
    results.add_diagnostic("mcts_game", "game_time_s", game_time)

    # Determine result
    if board.is_game_over():
        result = board.result()
        print(f"  Game result: {result}")
        results.add_diagnostic("mcts_game", "result", result)

        if result == "1-0":
            print(f"\n  {ok('Network won vs Pure MCTS!')}")
            passed = True
            score = 1.0
        elif result == "0-1":
            print(f"\n  {warn('Network lost to Pure MCTS')}")
            # Losing to MCTS is not as bad as losing to random
            passed = material >= -3
            score = 0.3
            if not passed:
                results.add_issue(
                    "MEDIUM",
                    "gameplay",
                    "Network loses to Pure MCTS",
                    "Network should generally beat MCTS without neural guidance",
                )
        else:
            print(f"\n  {warn('Draw vs Pure MCTS')}")
            passed = True
            score = 0.6
    else:
        results.add_diagnostic("mcts_game", "result", "incomplete")
        if material < -5:
            print(f"\n  {warn('Network losing significant material')}")
            passed = False
            score = 0.2
            results.add_issue(
                "MEDIUM",
                "gameplay",
                f"Network losing material ({material:+d}) vs MCTS",
                "Struggling against pure MCTS opponent",
            )
        elif material > 5:
            print(f"\n  {ok('Network winning with material advantage')}")
            passed = True
            score = 0.9
        else:
            print(f"\n  {ok('Material roughly equal - competitive game')}")
            passed = True
            score = 0.7

    results.add("Game vs Pure MCTS", passed, score, 0.8)
    return score
