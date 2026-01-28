#!/usr/bin/env python3
"""Debug script to analyze MCTS value head behavior across checkpoints."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess
import numpy as np
import torch

from alphazero.network import DualHeadNetwork
from alphazero.mcts import MCTS
from alphazero.move_encoding import decode_move, flip_policy
from alphazero.spatial_encoding import encode_board_with_history

# Test positions
TEST_POSITIONS = [
    {
        "name": "Hanging Queen",
        "fen": "rnb1kbnr/pppppppp/8/8/3q4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
        "expected": "d1d4",
        "description": "Black queen hanging on d4 - White can capture with Qxd4",
    },
    {
        "name": "Back Rank Mate",
        "fen": "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1",
        "expected": "a1a8",
        "description": "Ra8# is mate in 1",
    },
    {
        "name": "Queen Mate",
        "fen": "k7/8/1K6/8/8/8/8/Q7 w - - 0 1",
        "expected": "a1a7",
        "description": "Qa7# or Qa8# is mate in 1",
    },
]


def load_network(path: str) -> DualHeadNetwork:
    """Load network from checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})

    network = DualHeadNetwork(
        num_residual_blocks=config.get(
            "num_residual_blocks", config.get("num_blocks", 8)
        ),
        num_filters=config.get("num_filters", 128),
        num_input_planes=config.get("num_input_planes", 72),
    )
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    return network


def encode_position(board: chess.Board, network: DualHeadNetwork) -> np.ndarray:
    """Encode board position for network input (same as diagnostic tests)."""
    expected_planes = network.num_input_planes
    # 72 planes = (history_length + 1) * 12 + 24 (metadata + semantic + tactical)
    history_length = (expected_planes - 24) // 12 - 1
    boards = [board] * (history_length + 1)  # Current + history (all same)
    return encode_board_with_history(boards, from_perspective=True)


def predict_for_board(board: chess.Board, network: DualHeadNetwork):
    """Get network prediction with correct perspective handling."""
    state = encode_position(board, network)
    policy, value, wdl = network.predict_single_with_wdl(state)
    # Flip policy back to absolute coordinates for Black
    if board.turn == chess.BLACK:
        policy = flip_policy(policy)
    return policy, value, wdl


def analyze_position(network: DualHeadNetwork, fen: str, name: str, description: str):
    """Analyze a position with raw network and after each move."""
    board = chess.Board(fen)

    print(f"\n{'='*70}")
    print(f"Position: {name}")
    print(f"Description: {description}")
    print(f"FEN: {fen}")
    print(f"Side to move: {'White' if board.turn else 'Black'}")
    print(f"{'='*70}")
    print(board)

    # Get raw network output (using same method as diagnostic tests)
    policy, value, wdl = predict_for_board(board, network)

    print(f"\n--- RAW NETWORK OUTPUT (before any move) ---")
    print(
        f"Value (from {'White' if board.turn else 'Black'}'s perspective): {value:+.4f}"
    )
    print(f"WDL: [Win={wdl[0]:.3f}, Draw={wdl[1]:.3f}, Loss={wdl[2]:.3f}]")

    # Top 5 moves by policy
    top_indices = np.argsort(policy)[::-1][:10]

    print(f"\n--- TOP POLICY MOVES (what network thinks) ---")
    print(
        f"{'Rank':<6} {'Move':<8} {'Prior':>8} {'Value After':>12} {'WDL After':<30} {'Note':<15}"
    )
    print("-" * 90)

    found_moves = 0
    for idx in top_indices:
        if found_moves >= 5:
            break
        move = decode_move(idx, board)
        if move is None or move not in board.legal_moves:
            continue

        prior = policy[idx]

        # Make move and evaluate resulting position
        board.push(move)
        _, child_value, child_wdl = predict_for_board(board, network)

        # Check if it's checkmate
        is_mate = board.is_checkmate()
        board.pop()

        # Determine note
        note = ""
        if is_mate:
            note = "CHECKMATE!"
        elif board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                piece_names = {
                    1: "pawn",
                    2: "knight",
                    3: "bishop",
                    4: "rook",
                    5: "queen",
                    6: "king",
                }
                note = f"captures {piece_names.get(captured.piece_type, '?')}"

        wdl_str = f"[W={child_wdl[0]:.2f}, D={child_wdl[1]:.2f}, L={child_wdl[2]:.2f}]"

        # Value after is from opponent's perspective
        # If White played, child_value is from Black's view
        # MCTS will negate this: q = -child_value
        mcts_q = -child_value

        print(
            f"{found_moves+1:<6} {move.uci():<8} {prior*100:>7.2f}% {child_value:>+11.4f} {wdl_str:<30} {note:<15}"
        )
        print(f"       {'':8} {'':>8} MCTS Q={mcts_q:+.4f} (negated for parent)")
        found_moves += 1

    # Run MCTS
    print(f"\n--- MCTS SEARCH (100 sims) ---")
    history_length = (network.num_input_planes - 24) // 12 - 1
    mcts = MCTS(
        network=network,
        c_puct=1.5,
        num_simulations=100,
        history_length=history_length,
    )
    mcts.temperature = 0.0

    # Run search but also get access to internal state
    mcts_policy = mcts.search(board, add_noise=False)

    # Get root node to see actual visit counts
    root = mcts._get_or_create_node(board)

    print(
        f"{'Rank':<6} {'Move':<8} {'Visits':>8} {'Q-Value':>10} {'Prior':>8} {'PUCT Score':>12}"
    )
    print("-" * 70)

    # Sort children by visit count
    sorted_children = sorted(
        root.children.items(),
        key=lambda x: x[1].visit_count,
        reverse=True,
    )

    total_visits = sum(c.visit_count for _, c in sorted_children)
    sqrt_total = np.sqrt(total_visits) if total_visits > 0 else 1.0

    for i, (move, child) in enumerate(sorted_children[:10]):
        visits = child.visit_count
        q = -child.q_value if child.visit_count > 0 else 0.0
        prior = child.prior
        # PUCT formula
        u = mcts.c_puct * prior * sqrt_total / (1 + child.visit_count)
        puct_score = q + u

        pct = visits / total_visits * 100 if total_visits > 0 else 0
        marker = " <-- BEST" if i == 0 else ""

        # Check if this is a capture
        note = ""
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                piece_names = {1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}
                note = f"x{piece_names.get(captured.piece_type, '?')}"

        print(
            f"{i+1:<6} {move.uci():<8} {visits:>7} ({pct:4.1f}%) {q:>+9.4f} {prior*100:>7.2f}% {puct_score:>+11.4f} {note}{marker}"
        )


def main():
    checkpoints = [
        "models/pretrained_best_1_network.pt",
        "models/pretrained_best_2_network.pt",
        "models/pretrained_best_3_network.pt",
    ]

    for cp_path in checkpoints:
        print(f"\n{'#'*70}")
        print(f"# CHECKPOINT: {cp_path}")
        print(f"{'#'*70}")

        if not os.path.exists(cp_path):
            print(f"  File not found: {cp_path}")
            continue

        try:
            network = load_network(cp_path)
        except Exception as e:
            print(f"Error loading {cp_path}: {e}")
            continue

        for test in TEST_POSITIONS:
            analyze_position(network, test["fen"], test["name"], test["description"])

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(
        """
Key things to look for:
1. For 'Hanging Queen': After d1d4 (capture), Value After should be NEGATIVE
   (because it's from Black's perspective, and Black just lost their queen)
   MCTS Q should be POSITIVE (good for White)

2. If Value After is POSITIVE after capturing the queen, the value head is wrong!
   This would explain why MCTS rejects the capture.

3. Compare across checkpoints - does checkpoint 1 have different behavior?
"""
    )


if __name__ == "__main__":
    main()
