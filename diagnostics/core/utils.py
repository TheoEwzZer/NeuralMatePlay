"""Utility functions for network diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import chess

from alphazero.spatial_encoding import encode_board_with_history
from alphazero.move_encoding import flip_policy

if TYPE_CHECKING:
    from alphazero import DualHeadNetwork


def encode_for_network(board: chess.Board, network: DualHeadNetwork) -> np.ndarray:
    """
    Encode a board position for the given network.
    """
    expected_planes: int = network.num_input_planes
    history_length: int = (expected_planes - 24) // 12 - 1
    boards: list[chess.Board] = [board] * (history_length + 1)
    return encode_board_with_history(boards, from_perspective=True)


def predict_for_board(
    board: chess.Board, network: DualHeadNetwork
) -> tuple[np.ndarray, float]:
    """
    Get network prediction for a board position.
    Handles perspective flipping correctly: encodes from current player's view,
    then flips policy back to absolute coordinates for move decoding.
    """
    state: np.ndarray = encode_for_network(board, network)
    policy: np.ndarray
    value: float
    policy, value = network.predict_single(state)
    if board.turn == chess.BLACK:
        policy = flip_policy(policy)
    return policy, value


def get_history_length(network: DualHeadNetwork) -> int:
    """Calculate history_length from network input planes."""
    return (network.num_input_planes - 24) // 12 - 1
