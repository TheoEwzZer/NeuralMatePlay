"""
Test script to compare different c_puct values.

Plays AI vs AI with:
- Player 1: c_puct = 0.1 (exploitation-focused)
- Player 2: c_puct = 2.0 (exploration-focused)
"""

import chess
import torch
from src.alphazero.network import DualHeadNetwork
from src.alphazero.mcts import MCTS

# Configuration
MODEL_PATH = "models/pretrained_best_7_network.pt"
NUM_GAMES = 10
NUM_SIMULATIONS = 1600
C_PUCT_LOW = 0.1
C_PUCT_HIGH = 2


def load_network(path: str) -> DualHeadNetwork:
    """Load network from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Get network config from checkpoint or use defaults
    if "config" in checkpoint:
        config = checkpoint["config"]
        network = DualHeadNetwork(
            num_filters=config.get("num_filters", 192),
            num_residual_blocks=config.get("num_residual_blocks", 12),
            num_input_planes=config.get("num_input_planes", 60),
        )
    else:
        network = DualHeadNetwork()

    # Load weights
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
    network: DualHeadNetwork,
    mcts_low: MCTS,
    mcts_high: MCTS,
    low_is_white: bool,
) -> str:
    """
    Play a single game.

    Returns: "low", "high", or "draw"
    """
    board = chess.Board()

    while not board.is_game_over(claim_draw=True):
        if board.fullmove_number > 150:
            return "draw"

        # Select MCTS based on whose turn it is
        if board.turn == chess.WHITE:
            mcts = mcts_low if low_is_white else mcts_high
        else:
            mcts = mcts_high if low_is_white else mcts_low

        # Get move (temperature=0 set in constructor for deterministic play)
        move = mcts.get_best_move(board)
        board.push(move)

    # Determine result
    result = board.result(claim_draw=True)

    if result == "1-0":
        return "low" if low_is_white else "high"
    elif result == "0-1":
        return "high" if low_is_white else "low"
    else:
        return "draw"


def main():
    print(f"Loading network from {MODEL_PATH}...")
    network = load_network(MODEL_PATH)
    print(f"Network loaded on {next(network.parameters()).device}")

    print(f"\nCreating MCTS instances:")
    print(f"  - Low c_puct:  {C_PUCT_LOW}")
    print(f"  - High c_puct: {C_PUCT_HIGH}")
    print(f"  - Simulations: {NUM_SIMULATIONS}")

    mcts_low = MCTS(
        network=network,
        num_simulations=NUM_SIMULATIONS,
        c_puct=C_PUCT_LOW,
    )

    mcts_high = MCTS(
        network=network,
        num_simulations=NUM_SIMULATIONS,
        c_puct=C_PUCT_HIGH,
    )

    print(f"\nPlaying {NUM_GAMES} games...\n")

    results = {"low": 0, "high": 0, "draw": 0}

    for game_num in range(1, NUM_GAMES + 1):
        # Alternate colors
        low_is_white = game_num % 2 == 1
        color_str = "White" if low_is_white else "Black"

        print(
            f"Game {game_num}/{NUM_GAMES}: c_puct={C_PUCT_LOW} plays {color_str}...",
            end=" ",
            flush=True,
        )

        result = play_game(network, mcts_low, mcts_high, low_is_white)
        results[result] += 1

        if result == "low":
            print(f"c_puct={C_PUCT_LOW} wins!")
        elif result == "high":
            print(f"c_puct={C_PUCT_HIGH} wins!")
        else:
            print("Draw")

    # Print summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"c_puct={C_PUCT_LOW} (exploitation): {results['low']} wins")
    print(f"c_puct={C_PUCT_HIGH} (exploration):  {results['high']} wins")
    print(f"Draws: {results['draw']}")
    print()

    total = NUM_GAMES
    low_score = results["low"] + 0.5 * results["draw"]
    high_score = results["high"] + 0.5 * results["draw"]

    print(f"Score: {low_score}/{total} vs {high_score}/{total}")
    print(f"Win rate c_puct={C_PUCT_LOW}: {low_score/total*100:.1f}%")
    print(f"Win rate c_puct={C_PUCT_HIGH}: {high_score/total*100:.1f}%")


if __name__ == "__main__":
    main()
