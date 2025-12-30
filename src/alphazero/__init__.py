"""
AlphaZero-style chess engine.

This module provides:
- DualHeadNetwork: SE-ResNet with attention for policy and value prediction
- MCTS: Monte Carlo Tree Search with neural network guidance
- AlphaZeroTrainer: Self-play training loop
- Arena: Match playing and ELO estimation
"""

from .network import DualHeadNetwork, DualHeadResNet
from .mcts import MCTS
from .trainer import AlphaZeroTrainer, TrainingConfig
from .arena import Arena, NetworkPlayer, RandomPlayer, Player, ArenaStats, MatchResult
from .replay_buffer import ReplayBuffer
from .checkpoint_manager import CheckpointManager
from .device import get_device, set_device, supports_mixed_precision, get_device_info
from .move_encoding import (
    encode_move,
    decode_move,
    encode_move_from_perspective,
    decode_move_from_perspective,
    flip_policy,
    get_legal_move_mask,
    policy_to_moves,
    MOVE_ENCODING_SIZE,
)
from .spatial_encoding import (
    encode_board_spatial,
    encode_board_with_history,
    encode_single_position,
    PositionHistory,
    get_num_planes,
    history_length_from_planes,
    DEFAULT_HISTORY_LENGTH,
)

__all__ = [
    # Network
    "DualHeadNetwork",
    "DualHeadResNet",
    # MCTS
    "MCTS",
    # Training
    "AlphaZeroTrainer",
    "TrainingConfig",
    "ReplayBuffer",
    "CheckpointManager",
    # Arena
    "Arena",
    "NetworkPlayer",
    "RandomPlayer",
    "Player",
    "ArenaStats",
    "MatchResult",
    # Device
    "get_device",
    "set_device",
    "supports_mixed_precision",
    "get_device_info",
    # Move encoding
    "encode_move",
    "decode_move",
    "encode_move_from_perspective",
    "decode_move_from_perspective",
    "flip_policy",
    "get_legal_move_mask",
    "policy_to_moves",
    "MOVE_ENCODING_SIZE",
    # Spatial encoding
    "encode_board_spatial",
    "encode_board_with_history",
    "encode_single_position",
    "PositionHistory",
    "get_num_planes",
    "history_length_from_planes",
    "DEFAULT_HISTORY_LENGTH",
]

__version__ = "1.0.0"
