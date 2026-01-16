"""
Tactical weighting for pretraining positions.

This module provides functions to calculate weights for training positions
based on their tactical complexity. Positions with captures, checks, and
hanging pieces receive higher weights to emphasize tactical learning.
"""

import chess


def calculate_tactical_weight(
    board: chess.Board,
    move: chess.Move,
    capture_weight: float = 1.5,
    check_weight: float = 1.3,
    hanging_weight: float = 1.2,
) -> float:
    """
    Calculate a weight for a position based on tactical complexity.

    Higher weights are assigned to positions with tactical elements:
    - Captures: multiply by capture_weight
    - Checks: multiply by check_weight
    - Hanging pieces: multiply by hanging_weight

    Args:
        board: Current chess position (before move is made)
        move: The move being played
        capture_weight: Weight multiplier for captures
        check_weight: Weight multiplier for moves giving check
        hanging_weight: Weight multiplier for positions with hanging pieces

    Returns:
        Weight in range [1.0, 3.0]
    """
    weight = 1.0

    # Factor 1: Capture
    if board.is_capture(move):
        weight *= capture_weight

    # Factor 2: Check (need to apply move temporarily)
    board.push(move)
    gives_check = board.is_check()
    board.pop()

    if gives_check:
        weight *= check_weight

    # Factor 3: Hanging pieces in current position
    if _has_hanging_pieces(board):
        weight *= hanging_weight

    return min(weight, 3.0)  # Cap at 3x


def _has_hanging_pieces(board: chess.Board) -> bool:
    """
    Check if the position has hanging pieces (attacked but not defended).

    A piece is considered hanging if it's attacked by the opponent
    and has no defenders.

    Args:
        board: Chess position to analyze

    Returns:
        True if there are hanging pieces for the side to move
    """
    my_color = board.turn
    opp_color = not board.turn

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        # Skip empty squares and opponent pieces
        if piece is None or piece.color != my_color:
            continue

        # Skip the king (can't be captured)
        if piece.piece_type == chess.KING:
            continue

        # Check if attacked by opponent
        attackers = board.attackers(opp_color, square)
        if not attackers:
            continue

        # Check if defended
        defenders = board.attackers(my_color, square)
        if not defenders:
            return True

    return False
