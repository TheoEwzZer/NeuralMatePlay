"""Test: Find Equilibrium Point (Network vs Pure MCTS)."""

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
    info,
    get_history_length,
)


def play_quick_game(network_player, mcts_player, max_moves: int = 60):
    """
    Play a quick game and return the result.

    Returns:
        Tuple of (winner, material_diff, move_count)
        winner: 'network', 'mcts', or 'draw'
    """
    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        if board.turn == chess.WHITE:
            move = network_player.select_move(board)
        else:
            move = mcts_player.select_move(board)

        if move is None:
            break

        board.push(move)
        move_count += 1

    material = get_raw_material_diff(board)

    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return "network", material, move_count
        elif result == "0-1":
            return "mcts", material, move_count
        else:
            return "draw", material, move_count
    else:
        # Game not finished - use material to determine winner
        if material > 3:
            return "network", material, move_count
        elif material < -3:
            return "mcts", material, move_count
        else:
            return "draw", material, move_count


def test_equilibrium_point(network, results: TestResults):
    """
    Find the equilibrium point where Network and Pure MCTS perform equally.

    Tests the network against Pure MCTS at different simulation counts
    to find the point where they have ~50% win rate.

    This helps understand the "strength" of the trained network:
    - Low equilibrium (e.g., 100 sims): weak network
    - High equilibrium (e.g., 2000+ sims): strong network
    """
    print(header("TEST: Equilibrium Point (Network vs MCTS)"))

    history_length = get_history_length(network)

    # Network always uses 100 simulations (fixed)
    network_sims = 100

    # Test against different MCTS strengths
    mcts_sim_levels = [50, 100, 200, 400]
    games_per_level = 3

    print(subheader(f"Network ({network_sims} sims) vs Pure MCTS (varying sims)"))
    print(f"  Games per level: {games_per_level}")
    print()

    level_results = []

    for mcts_sims in mcts_sim_levels:
        network_player = NetworkPlayer(
            network,
            num_simulations=network_sims,
            name="Network",
            history_length=history_length,
        )
        mcts_player = PureMCTSPlayer(
            num_simulations=mcts_sims,
            max_rollout_depth=30,
            name=f"MCTS({mcts_sims})",
        )

        wins = 0
        losses = 0
        draws = 0
        total_material = 0

        start_time = time.time()

        for game_idx in range(games_per_level):
            winner, material, moves = play_quick_game(network_player, mcts_player)

            if winner == "network":
                wins += 1
            elif winner == "mcts":
                losses += 1
            else:
                draws += 1

            total_material += material

            # Reset players for next game
            network_player.reset()
            mcts_player.reset()

        elapsed = time.time() - start_time

        # Calculate win rate (wins + 0.5*draws) / total
        win_rate = (wins + 0.5 * draws) / games_per_level
        avg_material = total_material / games_per_level

        level_results.append(
            {
                "mcts_sims": mcts_sims,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": win_rate,
                "avg_material": avg_material,
                "time": elapsed,
            }
        )

        # Display result
        status = (
            ok("WIN")
            if win_rate > 0.6
            else (fail("LOSS") if win_rate < 0.4 else warn("EVEN"))
        )
        print(
            f"  vs MCTS({mcts_sims:>4} sims): {wins}W-{draws}D-{losses}L "
            f"({win_rate*100:5.1f}%) mat={avg_material:+.1f} {status}"
        )

    print()

    # Find equilibrium point (interpolate where win_rate crosses 0.5)
    equilibrium_sims = None

    for i in range(len(level_results) - 1):
        curr = level_results[i]
        next_r = level_results[i + 1]

        # Check if win rate crosses 0.5 between these levels
        if curr["win_rate"] >= 0.5 and next_r["win_rate"] < 0.5:
            # Linear interpolation
            ratio = (curr["win_rate"] - 0.5) / (curr["win_rate"] - next_r["win_rate"])
            equilibrium_sims = curr["mcts_sims"] + ratio * (
                next_r["mcts_sims"] - curr["mcts_sims"]
            )
            break

    # If network wins all, equilibrium is above our range
    if equilibrium_sims is None:
        if level_results[-1]["win_rate"] >= 0.5:
            equilibrium_sims = mcts_sim_levels[-1] * 2  # Estimate: above our range
            equilibrium_status = "above_range"
        else:
            equilibrium_sims = mcts_sim_levels[0] // 2  # Estimate: below our range
            equilibrium_status = "below_range"
    else:
        equilibrium_status = "found"

    print(subheader("Equilibrium Analysis"))

    if equilibrium_status == "above_range":
        print(f"  {ok('Strong network!')}")
        print(f"  Equilibrium point: >{mcts_sim_levels[-1]} MCTS simulations")
        print(f"  Network beats MCTS even at {mcts_sim_levels[-1]} sims")
        passed = True
        score = 1.0
    elif equilibrium_status == "below_range":
        print(f"  {fail('Weak network')}")
        print(f"  Equilibrium point: <{mcts_sim_levels[0]} MCTS simulations")
        print(f"  Network loses to MCTS even at {mcts_sim_levels[0]} sims")
        passed = False
        score = 0.2
    else:
        print(f"  Equilibrium point: ~{equilibrium_sims:.0f} MCTS simulations")

        if equilibrium_sims >= 400:
            print(
                f"  {ok('Good network!')} Beats MCTS until ~{equilibrium_sims:.0f} sims"
            )
            passed = True
            score = 0.8
        elif equilibrium_sims >= 200:
            print(
                f"  {warn('Moderate network')} Equilibrium at ~{equilibrium_sims:.0f} sims"
            )
            passed = True
            score = 0.6
        else:
            print(
                f"  {warn('Weak network')} Loses to MCTS above {equilibrium_sims:.0f} sims"
            )
            passed = True
            score = 0.4

    # Network strength interpretation
    print()
    print(subheader("Network Strength Rating"))

    strength_table = [
        (50, "Very Weak", "Loses to minimal MCTS"),
        (100, "Weak", "Barely trained"),
        (200, "Developing", "Learning basics"),
        (400, "Moderate", "Decent tactical awareness"),
        (800, "Good", "Strong pattern recognition"),
        (1600, "Strong", "Well-trained network"),
        (3200, "Very Strong", "Expert-level patterns"),
    ]

    network_rating = "Unknown"
    for threshold, rating, desc in strength_table:
        if equilibrium_sims >= threshold:
            network_rating = f"{rating} ({desc})"

    print(f"  Network rating: {network_rating}")
    print(f"  Equilibrium: ~{equilibrium_sims:.0f} MCTS simulations")

    # Add diagnostics
    results.add_diagnostic("equilibrium", "equilibrium_sims", equilibrium_sims)
    results.add_diagnostic("equilibrium", "equilibrium_status", equilibrium_status)
    results.add_diagnostic("equilibrium", "network_rating", network_rating)

    for i, lr in enumerate(level_results):
        results.add_diagnostic(
            "equilibrium", f"vs_mcts_{lr['mcts_sims']}_winrate", lr["win_rate"]
        )

    # Add issues if network is weak
    if equilibrium_sims < 200:
        results.add_issue(
            "HIGH",
            "network_strength",
            f"Network equilibrium point is low ({equilibrium_sims:.0f} sims)",
            "Network is weaker than expected. Consider more training.",
        )
        results.add_recommendation(
            1,
            "Continue training for more iterations/epochs",
            f"Network equilibrium at {equilibrium_sims:.0f} sims indicates undertrained model",
        )

    results.add("Equilibrium Point", passed, score, 1.0)
    results.add_timing("equilibrium_test", sum(lr["time"] for lr in level_results))

    return score
