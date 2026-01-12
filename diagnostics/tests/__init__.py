"""Individual diagnostic tests for chess neural networks."""

import sys
import os

# Ensure proper imports
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
if os.path.join(_root_dir, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_root_dir, "src"))

from .free_capture import test_free_capture
from .mate_in_one import test_mate_in_one
from .defense import test_defense
from .check_response import test_check_response
from .tactics import test_tactics
from .endgame import test_endgame
from .development import test_development
from .wdl_head import test_wdl_head
from .wdl_head_health import test_wdl_head_health
from .policy_diversity import test_policy_diversity
from .mcts_behavior import test_mcts_behavior
from .random_game import test_random_game

# All tests in order
ALL_TESTS = [
    (test_free_capture, "Free Capture", 1.0),
    (test_mate_in_one, "Mate in 1", 1.0),
    (test_defense, "Defense", 1.0),
    (test_check_response, "Check Response", 1.0),
    (test_tactics, "Tactics", 1.0),
    (test_endgame, "Endgame Understanding", 1.0),
    (test_development, "Opening Development", 1.0),
    (test_wdl_head, "Material Evaluation", 1.0),
    (test_wdl_head_health, "WDL Head Health", 3.0),  # CRITICAL for MCTS
    (test_policy_diversity, "Policy Diversity", 1.0),
    (test_mcts_behavior, "MCTS Behavior", 1.0),
    (test_random_game, "Game vs Random", 1.0),
]

__all__ = [
    "test_free_capture",
    "test_mate_in_one",
    "test_defense",
    "test_check_response",
    "test_tactics",
    "test_endgame",
    "test_development",
    "test_wdl_head",
    "test_wdl_head_health",
    "test_policy_diversity",
    "test_mcts_behavior",
    "test_random_game",
    "ALL_TESTS",
]
