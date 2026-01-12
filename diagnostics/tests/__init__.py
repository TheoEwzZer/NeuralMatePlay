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
from .mate_in_two import test_mate_in_two
from .defense import test_defense
from .check_response import test_check_response
from .tactics import test_tactics
from .endgame import test_endgame
from .development import test_development
from .wdl_head import test_wdl_head
from .wdl_head_health import test_wdl_head_health
from .policy_diversity import test_policy_diversity
from .mcts_behavior import test_mcts_behavior
from .passed_pawn import test_passed_pawn
from .king_safety import test_king_safety
from .trapped_pieces import test_trapped_pieces
from .pin_handling import test_pin_handling
from .zugzwang import test_zugzwang
from .fortress import test_fortress
from .exchange_eval import test_exchange_eval
from .tempo import test_tempo
from .repetition import test_repetition
from .pawn_structure import test_pawn_structure
from .outposts import test_outposts
from .bishop_quality import test_bishop_quality
from .prophylaxis import test_prophylaxis

# All tests in order
ALL_TESTS = [
    # === CRITICAL TESTS ===
    (test_free_capture, "Free Capture", 1.0),
    (test_mate_in_one, "Mate in 1", 1.0),
    (test_mate_in_two, "Mate in 2", 1.0),
    (test_defense, "Defense", 1.0),
    (test_check_response, "Check Response", 1.0),
    (test_tactics, "Tactics", 1.0),
    # === POSITIONAL TESTS ===
    (test_endgame, "Endgame Understanding", 1.0),
    (test_development, "Opening Development", 1.0),
    (test_passed_pawn, "Passed Pawn", 1.0),
    (test_king_safety, "King Safety", 1.0),
    (test_pawn_structure, "Pawn Structure", 1.0),
    (test_outposts, "Outposts", 1.0),
    (test_bishop_quality, "Bishop Quality", 1.0),
    # === TACTICAL AWARENESS ===
    (test_trapped_pieces, "Trapped Pieces", 1.0),
    (test_pin_handling, "Pin Handling", 1.0),
    (test_tempo, "Tempo", 1.0),
    (test_prophylaxis, "Prophylaxis", 1.0),
    # === ADVANCED CONCEPTS ===
    (test_zugzwang, "Zugzwang", 1.0),
    (test_fortress, "Fortress", 1.0),
    (test_exchange_eval, "Exchange Evaluation", 1.0),
    (test_repetition, "Repetition Awareness", 1.0),
    # === HEALTH CHECKS ===
    (test_wdl_head, "Material Evaluation", 1.0),
    (test_wdl_head_health, "WDL Head Health", 3.0),  # CRITICAL for MCTS
    (test_policy_diversity, "Policy Diversity", 1.0),
    (test_mcts_behavior, "MCTS Behavior", 1.0),
]

__all__ = [
    "test_free_capture",
    "test_mate_in_one",
    "test_mate_in_two",
    "test_defense",
    "test_check_response",
    "test_tactics",
    "test_endgame",
    "test_development",
    "test_passed_pawn",
    "test_king_safety",
    "test_pawn_structure",
    "test_outposts",
    "test_bishop_quality",
    "test_trapped_pieces",
    "test_pin_handling",
    "test_tempo",
    "test_prophylaxis",
    "test_zugzwang",
    "test_fortress",
    "test_exchange_eval",
    "test_repetition",
    "test_wdl_head",
    "test_wdl_head_health",
    "test_policy_diversity",
    "test_mcts_behavior",
    "ALL_TESTS",
]
