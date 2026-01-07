"""
Test script to find optimal simulation count.

Automatically tests multiple simulation thresholds to determine
where diminishing returns begin.
"""

import chess
import torch
from src.alphazero.network import DualHeadNetwork
from src.alphazero.mcts import MCTS

# Configuration
MODEL_PATH = "models_save/saved_pretrained_best_7_network.pt"
NUM_GAMES_PER_MATCH = 6  # Games per matchup (should be even for color balance)
C_PUCT = 1.5

# Simulation counts to test (each level plays against the next)
SIMULATION_LEVELS = [
    100,
    200,
    400,
    600,
    800,
    1000,
    1200,
    1400,
    1600,
    1800,
    2000,
    2200,
    2400,
    2600,
    2800,
    3000,
]


def load_network(path: str) -> DualHeadNetwork:
    """Load network from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "config" in checkpoint:
        config = checkpoint["config"]
        network = DualHeadNetwork(
            num_filters=config.get("num_filters", 192),
            num_residual_blocks=config.get("num_residual_blocks", 12),
            num_input_planes=config.get("num_input_planes", 60),
        )
    else:
        network = DualHeadNetwork()

    if "model_state_dict" in checkpoint:
        network.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        network.load_state_dict(checkpoint["state_dict"])
    else:
        network.load_state_dict(checkpoint)

    network.to(device)
    network.eval()
    return network


def play_game(
    mcts_low: MCTS,
    mcts_high: MCTS,
    low_is_white: bool,
) -> tuple[str, int]:
    """Play a single game. Returns: ("low", "high", or "draw", move_count)"""
    board = chess.Board()
    move_count = 0

    while not board.is_game_over(claim_draw=True):
        if board.fullmove_number > 150:
            return "draw", move_count

        if board.turn == chess.WHITE:
            mcts = mcts_low if low_is_white else mcts_high
        else:
            mcts = mcts_high if low_is_white else mcts_low

        move = mcts.get_best_move(board)
        board.push(move)
        move_count += 1

    result = board.result(claim_draw=True)

    if result == "1-0":
        return ("low" if low_is_white else "high"), move_count
    elif result == "0-1":
        return ("high" if low_is_white else "low"), move_count
    else:
        return "draw", move_count


def play_match(
    network: DualHeadNetwork,
    sims_low: int,
    sims_high: int,
    num_games: int,
) -> dict:
    """Play a match between two simulation counts."""
    mcts_low = MCTS(network=network, num_simulations=sims_low, c_puct=C_PUCT)
    mcts_high = MCTS(network=network, num_simulations=sims_high, c_puct=C_PUCT)

    results = {"low": 0, "high": 0, "draw": 0}

    for game_num in range(1, num_games + 1):
        low_is_white = game_num % 2 == 1
        color_str = "W" if low_is_white else "B"

        print(
            f"    Game {game_num}/{num_games} ({sims_low}={color_str})...",
            end=" ",
            flush=True,
        )

        result, moves = play_game(mcts_low, mcts_high, low_is_white)
        results[result] += 1

        winner = (
            f"{sims_low}"
            if result == "low"
            else f"{sims_high}" if result == "high" else "Draw"
        )
        print(f"{winner} ({moves} moves)")

    return results


def main():
    print(f"Loading network from {MODEL_PATH}...")
    network = load_network(MODEL_PATH)
    print(f"Network loaded on {next(network.parameters()).device}")

    print(f"\nSimulation levels to test: {SIMULATION_LEVELS}")
    print(f"Games per matchup: {NUM_GAMES_PER_MATCH}")
    print(f"c_puct: {C_PUCT}")

    # Store results for each matchup
    matchup_results = []

    # Test each pair of consecutive simulation levels
    for i in range(len(SIMULATION_LEVELS) - 1):
        sims_low = SIMULATION_LEVELS[i]
        sims_high = SIMULATION_LEVELS[i + 1]

        print(f"\n{'='*50}")
        print(f"MATCHUP: {sims_low} vs {sims_high} simulations")
        print("=" * 50)

        results = play_match(network, sims_low, sims_high, NUM_GAMES_PER_MATCH)

        low_score = results["low"] + 0.5 * results["draw"]
        high_score = results["high"] + 0.5 * results["draw"]
        high_winrate = high_score / NUM_GAMES_PER_MATCH * 100

        matchup_results.append(
            {
                "sims_low": sims_low,
                "sims_high": sims_high,
                "low_wins": results["low"],
                "high_wins": results["high"],
                "draws": results["draw"],
                "low_score": low_score,
                "high_score": high_score,
                "high_winrate": high_winrate,
                "advantage": high_score - low_score,
            }
        )

        print(
            f"\n  Result: {sims_low} sims: {low_score} | {sims_high} sims: {high_score}"
        )
        print(f"  Higher sim winrate: {high_winrate:.1f}%")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"\n{'Matchup':<20} {'Low Score':<12} {'High Score':<12} {'Advantage':<12}")
    print("-" * 56)

    for m in matchup_results:
        matchup_str = f"{m['sims_low']} vs {m['sims_high']}"
        print(
            f"{matchup_str:<20} {m['low_score']:<12.1f} {m['high_score']:<12.1f} {m['advantage']:+.1f}"
        )

    # Determine optimal simulation count
    print("\n" + "=" * 60)
    print("ANALYSIS & RECOMMENDATION")
    print("=" * 60)

    # Find where diminishing returns begin (advantage < 1 point)
    optimal_sims = SIMULATION_LEVELS[0]
    diminishing_threshold = 1.0  # Less than 1 point advantage = diminishing returns

    for i, m in enumerate(matchup_results):
        if m["advantage"] >= diminishing_threshold:
            # Higher sims still provide significant advantage
            optimal_sims = m["sims_high"]
            print(
                f"\n  {m['sims_low']} -> {m['sims_high']}: +{m['advantage']:.1f} advantage (significant)"
            )
        else:
            print(
                f"\n  {m['sims_low']} -> {m['sims_high']}: +{m['advantage']:.1f} advantage (diminishing returns)"
            )
            # Don't update optimal_sims, stay at previous level
            if i == 0:
                optimal_sims = m["sims_low"]
            break
    else:
        # All levels showed significant improvement
        optimal_sims = SIMULATION_LEVELS[-1]
        print(f"\n  All levels show improvement - consider testing higher values")

    print(f"\n{'='*60}")
    print(f"OPTIMAL SIMULATION COUNT: {optimal_sims}")
    print("=" * 60)

    # Additional insights
    if optimal_sims == SIMULATION_LEVELS[-1]:
        print("\nNote: The network may still benefit from more simulations.")
        print("Consider testing higher values (e.g., 3200, 6400).")
    elif optimal_sims == SIMULATION_LEVELS[0]:
        print("\nNote: Even the lowest simulation count performs similarly.")
        print("The network quality is likely the bottleneck, not search depth.")
    else:
        idx = SIMULATION_LEVELS.index(optimal_sims)
        next_level = (
            SIMULATION_LEVELS[idx + 1] if idx + 1 < len(SIMULATION_LEVELS) else None
        )
        if next_level:
            print(
                f"\nNote: Going from {optimal_sims} to {next_level} provides minimal benefit."
            )
            print(f"Use {optimal_sims} simulations for best performance/cost ratio.")


if __name__ == "__main__":
    main()
