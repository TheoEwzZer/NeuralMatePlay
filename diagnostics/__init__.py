"""
Neural Network Diagnostic Suite for Chess Engine.
Provides comprehensive testing and comparison tools.
"""

from .core.colors import Colors, ok, fail, warn, info, header, subheader, dim
from .core.results import TestResults
from .core.utils import encode_for_network, predict_for_board
from .core.network_loader import (
    load_network_from_path,
    load_latest_network,
    get_available_checkpoints,
)

__all__ = [
    "Colors",
    "ok",
    "fail",
    "warn",
    "info",
    "header",
    "subheader",
    "dim",
    "TestResults",
    "encode_for_network",
    "predict_for_board",
    "load_network_from_path",
    "load_latest_network",
    "get_available_checkpoints",
]
