"""Core diagnostic utilities."""

import sys
import os

# Ensure src is in path for alphazero imports
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_root_dir, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_root_dir, "src"))

from .colors import Colors, ok, fail, warn, info, header, subheader, dim
from .results import TestResults
from .utils import encode_for_network, predict_for_board, get_history_length
from .network_loader import (
    load_network_from_path,
    load_latest_network,
    get_available_checkpoints,
    analyze_network_architecture,
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
    "get_history_length",
    "load_network_from_path",
    "load_latest_network",
    "get_available_checkpoints",
    "analyze_network_architecture",
]
