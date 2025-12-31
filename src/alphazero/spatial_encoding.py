"""
Spatial encoding of chess positions for neural network input.

Default: 54 planes with 3 positions of history.

Structure:
- Planes 0-11:  Current position pieces (6 piece types × 2 colors)
- Planes 12-23: Position T-1 (one move ago)
- Planes 24-35: Position T-2 (two moves ago)
- Planes 36-53: Metadata (turn, castling, en passant, repetition, etc.)
"""

import chess
import numpy as np
from typing import Optional


# Piece type to plane offset mapping
PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

# Number of planes per position (6 piece types × 2 colors)
PLANES_PER_POSITION = 12

# Number of metadata planes
METADATA_PLANES = 18

# Default history length
DEFAULT_HISTORY_LENGTH = 3


def encode_board_spatial(board: chess.Board) -> np.ndarray:
    """
    Encode a single board position to 18 planes (no history).

    Args:
        board: Chess board to encode.

    Returns:
        numpy array of shape (18, 8, 8).
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    # Encode pieces (planes 0-11)
    _encode_pieces(board, planes, 0)

    # Encode metadata (planes 12-17)
    _encode_metadata_simple(board, planes, 12)

    return planes


def encode_board_with_history(
    boards: list[chess.Board],
    from_perspective: bool = True,
) -> np.ndarray:
    """
    Encode board positions with history to 60 planes.

    Args:
        boards: List of boards [current, T-1, T-2, ...]. Length should be
                history_length + 1. Missing history is zero-filled.
        from_perspective: If True, encode from current player's perspective
                          (flip board for black).

    Returns:
        numpy array of shape (60, 8, 8).
    """
    history_length = DEFAULT_HISTORY_LENGTH
    num_position_planes = (history_length + 1) * PLANES_PER_POSITION  # 48
    total_planes = num_position_planes + 12  # 60 (metadata + attack maps)

    planes = np.zeros((total_planes, 8, 8), dtype=np.float32)

    current_board = boards[0] if boards else chess.Board()
    flip = from_perspective and current_board.turn == chess.BLACK

    # Encode each position in history
    for i, board in enumerate(boards[: history_length + 1]):
        plane_offset = i * PLANES_PER_POSITION
        _encode_pieces(board, planes, plane_offset, flip=flip)

    # Zero-fill missing history
    for i in range(len(boards), history_length + 1):
        pass  # Already zeros

    # Encode metadata (simplified for 54 planes)
    metadata_offset = num_position_planes  # 48
    _encode_metadata_54(current_board, planes, metadata_offset, flip=flip)

    return planes


def _encode_pieces(
    board: chess.Board,
    planes: np.ndarray,
    offset: int,
    flip: bool = False,
) -> None:
    """
    Encode piece positions into planes using bitboard iteration.

    Optimized: Iterates only over existing pieces (~16-32) instead of
    all 64 squares. Uses board.pieces() which leverages internal bitboards.

    Args:
        board: Chess board.
        planes: Target array to fill.
        offset: Starting plane index.
        flip: Whether to flip perspective (for black).
    """
    # Iterate by piece type and color (faster than checking all 64 squares)
    for piece_type in range(1, 7):  # PAWN=1 to KING=6
        piece_plane = PIECE_TO_PLANE[piece_type]

        for color in [chess.WHITE, chess.BLACK]:
            # Get all squares with this piece type and color (bitboard operation)
            squares = board.pieces(piece_type, color)
            if not squares:
                continue

            # Determine plane index based on color and flip
            if flip:
                # When flipped, swap colors for plane assignment
                display_color = not color
            else:
                display_color = color

            if display_color == chess.WHITE:
                plane_idx = offset + piece_plane
            else:
                plane_idx = offset + 6 + piece_plane

            # Set bits for all pieces of this type/color
            for square in squares:
                file_idx = chess.square_file(square)
                rank_idx = chess.square_rank(square)

                if flip:
                    rank_idx = 7 - rank_idx

                # Set the bit (note: rank 0 is bottom, but array index 0 is top)
                planes[plane_idx, 7 - rank_idx, file_idx] = 1.0


def _encode_metadata_simple(
    board: chess.Board,
    planes: np.ndarray,
    offset: int,
) -> None:
    """
    Encode metadata for 18-plane encoding.

    Args:
        board: Chess board.
        planes: Target array.
        offset: Starting plane index.
    """
    # Plane 0: Side to move (1 if white, 0 if black)
    if board.turn == chess.WHITE:
        planes[offset] = 1.0

    # Plane 1: White kingside castling
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[offset + 1] = 1.0

    # Plane 2: White queenside castling
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[offset + 2] = 1.0

    # Plane 3: Black kingside castling
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[offset + 3] = 1.0

    # Plane 4: Black queenside castling
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[offset + 4] = 1.0

    # Plane 5: En passant square
    if board.ep_square is not None:
        file_idx = chess.square_file(board.ep_square)
        rank_idx = chess.square_rank(board.ep_square)
        planes[offset + 5, 7 - rank_idx, file_idx] = 1.0


def _encode_attack_maps(
    board: chess.Board,
    planes: np.ndarray,
    offset: int,
    flip: bool = False,
) -> None:
    """
    Encode attack maps for both players.

    Planes (from offset):
        0: My attacks (squares attacked by player to move)
        1: Opponent attacks (squares attacked by opponent)

    Args:
        board: Chess board.
        planes: Target array.
        offset: Starting plane index.
        flip: Whether perspective is flipped.
    """
    # Determine colors from perspective
    if flip:
        my_color = chess.BLACK
        opp_color = chess.WHITE
    else:
        my_color = chess.WHITE if board.turn == chess.WHITE else chess.BLACK
        opp_color = chess.BLACK if board.turn == chess.WHITE else chess.WHITE

    # My attacks (plane 0)
    for square in chess.SQUARES:
        attackers = board.attackers(my_color, square)
        if attackers:
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            if flip:
                rank_idx = 7 - rank_idx
            planes[offset, 7 - rank_idx, file_idx] = 1.0

    # Opponent attacks (plane 1)
    for square in chess.SQUARES:
        attackers = board.attackers(opp_color, square)
        if attackers:
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            if flip:
                rank_idx = 7 - rank_idx
            planes[offset + 1, 7 - rank_idx, file_idx] = 1.0


def _encode_metadata_54(
    board: chess.Board,
    planes: np.ndarray,
    offset: int,
    flip: bool = False,
) -> None:
    """
    Encode metadata for 60-plane encoding.

    Planes (from offset):
        0: Side to move (always 1 from perspective)
        1: Move count (normalized /200)
        2-5: Castling rights (my kingside, my queenside, opp kingside, opp queenside)
        6: En passant square
        7: Halfmove clock (50-move rule, normalized /100)
        8: Repetition count (0.5 if 1x, 1.0 if 2x+)
        9: Is in check
        10: My attacks (squares attacked by player to move)
        11: Opponent attacks (squares attacked by opponent)

    Args:
        board: Chess board.
        planes: Target array.
        offset: Starting plane index (48 for 60-plane encoding with history_length=3).
        flip: Whether perspective is flipped.
    """
    # Plane 0: Side to move (always 1 from current player's perspective)
    planes[offset] = 1.0  # Always "my turn" from perspective

    # Plane 1: Total move count (normalized)
    move_count = min(board.fullmove_number, 200) / 200.0
    planes[offset + 1] = move_count

    # Planes 2-5: Castling rights (from perspective)
    if flip:
        # From black's perspective: black's rights are "my" rights
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[offset + 2] = 1.0  # My kingside
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[offset + 3] = 1.0  # My queenside
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[offset + 4] = 1.0  # Opponent kingside
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[offset + 5] = 1.0  # Opponent queenside
    else:
        # From white's perspective
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[offset + 2] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[offset + 3] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[offset + 4] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[offset + 5] = 1.0

    # Plane 6: En passant square
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        ep_rank = chess.square_rank(board.ep_square)
        if flip:
            ep_rank = 7 - ep_rank
        planes[offset + 6, 7 - ep_rank, ep_file] = 1.0

    # Plane 7: Halfmove clock (50-move rule, normalized)
    halfmove_clock = min(board.halfmove_clock, 100) / 100.0
    planes[offset + 7] = halfmove_clock

    # Plane 8: Repetition count
    # Note: is_repetition() requires move stack, may return False for standalone boards
    try:
        if board.is_repetition(2):
            planes[offset + 8] = 1.0  # 2+ repetitions (draw imminent)
        elif board.is_repetition(1):
            planes[offset + 8] = 0.5  # 1 repetition
    except Exception:
        pass  # No move stack available, leave as 0

    # Plane 9: Is in check
    if board.is_check():
        planes[offset + 9] = 1.0

    # Planes 10-11: Attack maps
    _encode_attack_maps(board, planes, offset + 10, flip)


def encode_single_position(
    board: chess.Board,
    from_perspective: bool = True,
) -> np.ndarray:
    """
    Encode a single position (convenience wrapper for 54 planes with no history).

    Args:
        board: Chess board.
        from_perspective: Encode from current player's perspective.

    Returns:
        numpy array of shape (54, 8, 8).
    """
    return encode_board_with_history([board], from_perspective)


def get_num_planes(history_length: int = DEFAULT_HISTORY_LENGTH) -> int:
    """
    Calculate the number of input planes for a given history length.

    Args:
        history_length: Number of historical positions (0 = current only).

    Returns:
        Total number of planes.
    """
    if history_length == 0:
        return 18  # Simple encoding
    else:
        # (history_length + 1) positions × 12 planes + 12 metadata (includes attack maps)
        return (history_length + 1) * PLANES_PER_POSITION + 12


def history_length_from_planes(num_planes: int) -> int:
    """
    Infer history length from number of input planes.

    Args:
        num_planes: Number of input planes.

    Returns:
        History length (0 for 18 planes, 3 for 60 planes, etc.).
    """
    if num_planes == 18:
        return 0
    else:
        # num_planes = (history_length + 1) * 12 + 12
        # num_planes - 12 = (history_length + 1) * 12
        # (num_planes - 12) / 12 = history_length + 1
        return (num_planes - 12) // 12 - 1


class PositionHistory:
    """
    Helper class to maintain position history during a game.
    """

    def __init__(self, history_length: int = DEFAULT_HISTORY_LENGTH):
        """
        Initialize position history tracker.

        Args:
            history_length: Number of past positions to remember.
        """
        self.history_length = history_length
        self._boards: list[chess.Board] = []

    def push(self, board: chess.Board) -> None:
        """
        Add a new position to history.

        Args:
            board: Current board position (will be copied).
        """
        self._boards.insert(0, board.copy())

        # Keep only the required history
        if len(self._boards) > self.history_length + 1:
            self._boards = self._boards[: self.history_length + 1]

    def get_boards(self) -> list[chess.Board]:
        """
        Get the list of boards for encoding.

        Returns:
            List of boards [current, T-1, T-2, ...].
        """
        return self._boards.copy()

    def encode(self, from_perspective: bool = True) -> np.ndarray:
        """
        Encode the current position with history.

        Args:
            from_perspective: Encode from current player's perspective.

        Returns:
            numpy array of shape (54, 8, 8) for default history length.
        """
        if not self._boards:
            # Return empty encoding for empty history
            return encode_board_with_history([chess.Board()], from_perspective)

        return encode_board_with_history(self._boards, from_perspective)

    def clear(self) -> None:
        """Clear all history."""
        self._boards.clear()

    def reset(self, board: Optional[chess.Board] = None) -> None:
        """
        Reset history with optional starting position.

        Args:
            board: Starting position (uses standard start if None).
        """
        self.clear()
        if board is not None:
            self.push(board)
        else:
            self.push(chess.Board())
