"""Test: Game vs Random Player."""

import time
import chess

from alphazero.arena import NetworkPlayer, RandomPlayer
from src.chess_encoding.board_utils import get_raw_material_diff
from ..core import (
    TestResults,
    header,
    subheader,
    ok,
    fail,
    warn,
    get_history_length,
)


def test_random_game(network, results: TestResults):
    """Play a quick game vs random to see what happens."""
    print(header("TEST: Game vs Random Player"))

    history_length = get_history_length(network)
    network_player = NetworkPlayer(
        network, num_simulations=50, name="Network", history_length=history_length
    )
    random_player = RandomPlayer()

    board = chess.Board()
    move_count = 0
    move_log = []

    print(subheader("Playing as White (50 MCTS simulations per move)"))

    start_time = time.time()

    while not board.is_game_over() and move_count < 60:
        if board.turn == chess.WHITE:
            move = network_player.select_move(board)
            player = "Network"
        else:
            move = random_player.select_move(board)
            player = "Random"

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
    print(f"  Avg time per move: {game_time/move_count*1000:.0f}ms")
    print(f"  Material balance: {material:+d}")

    results.add_diagnostic("game", "total_moves", move_count)
    results.add_diagnostic("game", "material_diff", material)
    results.add_diagnostic("game", "game_time_s", game_time)

    # Determine result
    if board.is_game_over():
        result = board.result()
        print(f"  Game result: {result}")
        results.add_diagnostic("game", "result", result)

        if result == "1-0":
            print(f"\n  {ok('Network won!')}")
            passed = True
            score = 1.0
        elif result == "0-1":
            print(f"\n  {fail('Network lost to random!')}")
            passed = False
            score = 0.0
            results.add_issue(
                "CRITICAL",
                "gameplay",
                "Network loses to random player",
                "This is a serious problem - random play should be easily beaten",
            )
        else:
            print(f"\n  {warn('Draw')}")
            passed = material >= 0
            score = 0.5
    else:
        results.add_diagnostic("game", "result", "incomplete")
        if material < -5:
            print(f"\n  {fail('Network losing significant material')}")
            passed = False
            score = 0.0
            results.add_issue(
                "HIGH",
                "gameplay",
                f"Network losing material ({material:+d}) vs random",
                "Cannot hold material against random play",
            )
        elif material > 5:
            print(f"\n  {ok('Network winning with material advantage')}")
            passed = True
            score = 1.0
        else:
            print(f"\n  {warn('Material roughly equal')}")
            passed = True
            score = 0.7

    results.add("Game vs Random", passed, score, 1.0)
    return score
