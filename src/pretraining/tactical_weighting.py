"""
Tactical weighting for pretraining positions.

This module provides functions to calculate weights for training positions
based on their tactical complexity. Positions with captures, checks, and
hanging pieces receive higher weights to emphasize tactical learning.

Extended to detect rare/complex positions:
- Zugzwang (few legal moves + quiet position)
- Repetition potential
- King safety (multiple attackers)
- Endgames
"""

import chess


def calculate_tactical_weight(
    board: chess.Board,
    move: chess.Move,
    capture_weight: float = 1.5,
    check_weight: float = 1.3,
    hanging_weight: float = 1.2,
    zugzwang_weight: float = 2.0,
    repetition_weight: float = 1.8,
    king_danger_weight: float = 1.5,
    endgame_weight: float = 1.3,
) -> float:
    """
    Calculate a weight for a position based on tactical complexity.

    Higher weights are assigned to positions with tactical elements:
    - Captures: multiply by capture_weight
    - Checks: multiply by check_weight
    - Hanging pieces: multiply by hanging_weight
    - Zugzwang potential: multiply by zugzwang_weight (few legal moves + quiet)
    - Repetition potential: multiply by repetition_weight
    - King in danger: multiply by king_danger_weight (multiple attackers)
    - Endgames: multiply by endgame_weight

    Args:
        board: Current chess position (before move is made)
        move: The move being played
        capture_weight: Weight multiplier for captures
        check_weight: Weight multiplier for moves giving check
        hanging_weight: Weight multiplier for positions with hanging pieces
        zugzwang_weight: Weight multiplier for zugzwang-like positions
        repetition_weight: Weight multiplier for repetition-prone positions
        king_danger_weight: Weight multiplier for king safety issues
        endgame_weight: Weight multiplier for endgame positions

    Returns:
        Weight in range [1.0, 5.0]
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

    # Factor 4: Zugzwang potential (few legal moves + not in check)
    # Critical positions where the player has limited options
    legal_move_count = board.legal_moves.count()
    if legal_move_count <= 5 and not board.is_check():
        weight *= zugzwang_weight

    # Factor 5: Repetition potential (2nd occurrence)
    # Important for understanding when to avoid/seek draws
    if board.is_repetition(2):
        weight *= repetition_weight

    # Factor 6: King in danger (multiple attackers on king)
    # Critical for king safety evaluation
    king_sq = board.king(board.turn)
    if king_sq is not None:
        attackers = board.attackers(not board.turn, king_sq)
        if len(attackers) >= 2:
            weight *= king_danger_weight

    # Factor 7: Endgame positions
    # More nuanced play, often zugzwang-prone
    if _is_endgame(board):
        weight *= endgame_weight

    return min(weight, 5.0)  # Cap at 5x (increased from 3x)


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


def _is_endgame(board: chess.Board) -> bool:
    """
    Determine if the position is an endgame.

    Uses a simple material-based heuristic:
    - No queens on the board, OR
    - Each side has at most 1 minor piece + rooks + pawns

    Args:
        board: Chess position to analyze

    Returns:
        True if the position is an endgame
    """
    # Count material for both sides
    white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))

    # No queens = endgame
    if white_queens == 0 and black_queens == 0:
        return True

    # Count minor pieces (bishops + knights)
    white_minors = (
        len(board.pieces(chess.BISHOP, chess.WHITE))
        + len(board.pieces(chess.KNIGHT, chess.WHITE))
    )
    black_minors = (
        len(board.pieces(chess.BISHOP, chess.BLACK))
        + len(board.pieces(chess.KNIGHT, chess.BLACK))
    )

    # If one side has queen but very limited material, still endgame
    # Queen vs minor piece endings are tricky endgames
    if white_queens <= 1 and black_queens <= 1:
        # Both sides have at most queen + 1 minor = likely endgame
        if white_minors <= 1 and black_minors <= 1:
            return True

    # Count total non-pawn pieces (excluding kings)
    total_pieces = 0
    for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        total_pieces += len(board.pieces(piece_type, chess.WHITE))
        total_pieces += len(board.pieces(piece_type, chess.BLACK))

    # Very few pieces = endgame
    if total_pieces <= 4:
        return True

    return False


def _has_king_danger(board: chess.Board) -> bool:
    """
    Check if the king of the side to move is under significant threat.

    A king is in danger if:
    - Multiple pieces are attacking squares around the king
    - The king has few escape squares

    Args:
        board: Chess position to analyze

    Returns:
        True if the king is in significant danger
    """
    my_color = board.turn
    opp_color = not board.turn
    king_sq = board.king(my_color)

    if king_sq is None:
        return False

    # Count attackers on king
    attackers = board.attackers(opp_color, king_sq)
    if len(attackers) >= 2:
        return True

    # Check escape squares (king's adjacent squares)
    escape_squares = 0
    for direction in [-9, -8, -7, -1, 1, 7, 8, 9]:
        target_sq = king_sq + direction
        if 0 <= target_sq < 64:
            # Check if square is on the board and not attacked
            if not board.is_attacked_by(opp_color, target_sq):
                # Check if square is empty or has opponent piece
                piece = board.piece_at(target_sq)
                if piece is None or piece.color == opp_color:
                    escape_squares += 1

    # King with few escapes is in danger
    if escape_squares <= 1 and len(attackers) >= 1:
        return True

    return False
