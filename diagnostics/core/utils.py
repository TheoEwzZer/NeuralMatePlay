"""Utility functions for network diagnostics."""

import numpy as np
import chess

from alphazero.spatial_encoding import encode_board_with_history
from alphazero.move_encoding import flip_policy
from alphazero import DualHeadNetwork


def encode_for_network(board: chess.Board, network: DualHeadNetwork) -> np.ndarray:
    """
    Encode a board position for the given network.
    """
    expected_planes = network.num_input_planes
    # 72 planes = (history_length + 1) * 12 + 24 (metadata + semantic + tactical)
    history_length = (expected_planes - 24) // 12 - 1
    boards = [board] * (history_length + 1)  # Current + history (all same)
    return encode_board_with_history(boards, from_perspective=True)


def predict_for_board(
    board: chess.Board, network: DualHeadNetwork
) -> tuple[np.ndarray, float]:
    """
    Get network prediction for a board position.
    Handles perspective flipping correctly: encodes from current player's view,
    then flips policy back to absolute coordinates for move decoding.
    """
    state = encode_for_network(board, network)
    policy, value = network.predict_single(state)
    # Flip policy back to absolute coordinates for Black
    if board.turn == chess.BLACK:
        policy = flip_policy(policy)
    return policy, value


def get_history_length(network) -> int:
    """Calculate history_length from network input planes."""
    # 72 planes = (history_length + 1) * 12 + 24 (metadata + semantic + tactical)
    return (network.num_input_planes - 24) // 12 - 1
