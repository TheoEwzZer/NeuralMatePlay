"""
Spatial encoding of chess positions for neural network input.

Default: 72 planes with 3 positions of history.

Structure:
- Planes 0-11:  Current position pieces (6 piece types × 2 colors)
- Planes 12-23: Position T-1 (one move ago)
- Planes 24-35: Position T-2 (two moves ago)
- Planes 36-47: Position T-3 (three moves ago)
- Planes 48-59: Metadata (turn, castling, en passant, repetition, attack maps)
- Planes 60-67: Semantic features (king safety, mobility, pawn structure, center)
- Planes 68-71: Tactical features (pins, hanging, attacking, trapped pieces)
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

# Number of metadata planes (12 basic + used in calculation)
METADATA_PLANES = 12

# Number of extra semantic planes (NNUE-inspired features)
SEMANTIC_PLANES = 8

# Number of tactical planes (pins, hanging, attacks, trapped)
TACTICAL_PLANES = 4

# Default history length
DEFAULT_HISTORY_LENGTH = 3

# Center squares for control evaluation
CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]
EXTENDED_CENTER = [
    chess.C3,
    chess.C4,
    chess.C5,
    chess.C6,
    chess.D3,
    chess.D4,
    chess.D5,
    chess.D6,
    chess.E3,
    chess.E4,
    chess.E5,
    chess.E6,
    chess.F3,
    chess.F4,
    chess.F5,
    chess.F6,
]


def encode_board_with_history(
    boards: list[chess.Board],
    from_perspective: bool = True,
) -> np.ndarray:
    """
    Encode board positions with history to 72 planes.

    Args:
        boards: List of boards [current, T-1, T-2, ...]. Length should be
                history_length + 1. Missing history is zero-filled.
        from_perspective: If True, encode from current player's perspective
                          (flip board for black).

    Returns:
        numpy array of shape (72, 8, 8).
    """
    history_length = DEFAULT_HISTORY_LENGTH
    num_position_planes = (history_length + 1) * PLANES_PER_POSITION  # 48
    metadata_planes = 12  # Basic metadata + attack maps
    total_planes = (
        num_position_planes + metadata_planes + SEMANTIC_PLANES + TACTICAL_PLANES
    )  # 72

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

    # Encode metadata (planes 48-59)
    metadata_offset = num_position_planes  # 48
    _encode_metadata_54(current_board, planes, metadata_offset, flip=flip)

    # Encode semantic features (planes 60-67)
    semantic_offset = num_position_planes + metadata_planes  # 60
    _encode_semantic_features(current_board, planes, semantic_offset, flip=flip)

    # Encode tactical features (planes 68-71)
    tactical_offset = semantic_offset + SEMANTIC_PLANES  # 68
    _encode_tactical_features(current_board, planes, tactical_offset, flip=flip)

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


def _encode_semantic_features(
    board: chess.Board,
    planes: np.ndarray,
    offset: int,
    flip: bool = False,
) -> None:
    """
    Encode NNUE-inspired semantic features.

    Planes (from offset):
        0: King attackers - pieces attacking opponent's king
        1: King defenders - pieces defending my king
        2: Knight mobility - squares accessible by my knights
        3: Bishop mobility - squares accessible by my bishops
        4: Passed pawns - my passed pawns
        5: Isolated pawns - my isolated pawns
        6: Weak squares - undefended squares in my territory
        7: Center control - control of central squares

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
        my_color = board.turn
        opp_color = not board.turn

    # Helper to set a square in a plane
    def set_square(plane_idx: int, square: int) -> None:
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        if flip:
            rank_idx = 7 - rank_idx
        planes[plane_idx, 7 - rank_idx, file_idx] = 1.0

    # Plane 0: King attackers - my pieces that attack opponent's king
    opp_king_square = board.king(opp_color)
    if opp_king_square is not None:
        # Mark squares around opponent king that my pieces attack
        for adj_square in _get_king_zone(opp_king_square):
            attackers = board.attackers(my_color, adj_square)
            for attacker_sq in attackers:
                set_square(offset, attacker_sq)

    # Plane 1: King defenders - my pieces that defend squares around my king
    my_king_square = board.king(my_color)
    if my_king_square is not None:
        for adj_square in _get_king_zone(my_king_square):
            defenders = board.attackers(my_color, adj_square)
            for defender_sq in defenders:
                set_square(offset + 1, defender_sq)

    # Plane 2: Knight mobility - squares my knights can move to
    for knight_sq in board.pieces(chess.KNIGHT, my_color):
        for move in board.legal_moves:
            if move.from_square == knight_sq:
                set_square(offset + 2, move.to_square)

    # Plane 3: Bishop mobility - squares my bishops can move to
    for bishop_sq in board.pieces(chess.BISHOP, my_color):
        for move in board.legal_moves:
            if move.from_square == bishop_sq:
                set_square(offset + 3, move.to_square)

    # Plane 4: Passed pawns - my pawns with no opposing pawns ahead
    my_pawns = board.pieces(chess.PAWN, my_color)
    opp_pawns = board.pieces(chess.PAWN, opp_color)
    for pawn_sq in my_pawns:
        if _is_passed_pawn(pawn_sq, my_color, opp_pawns):
            set_square(offset + 4, pawn_sq)

    # Plane 5: Isolated pawns - my pawns with no friendly pawns on adjacent files
    for pawn_sq in my_pawns:
        if _is_isolated_pawn(pawn_sq, my_pawns):
            set_square(offset + 5, pawn_sq)

    # Plane 6: Weak squares - squares in my half not defended by my pawns
    my_half_ranks = range(0, 4) if my_color == chess.WHITE else range(4, 8)
    if flip:
        my_half_ranks = range(4, 8) if my_color == chess.WHITE else range(0, 4)
    for rank in my_half_ranks:
        for file in range(8):
            square = chess.square(file, rank)
            # Check if defended by my pawns
            pawn_defenders = board.attackers(my_color, square) & my_pawns
            if not pawn_defenders:
                set_square(offset + 6, square)

    # Plane 7: Center control - central squares I attack
    for center_sq in CENTER_SQUARES:
        if board.attackers(my_color, center_sq):
            set_square(offset + 7, center_sq)
    # Also mark extended center with lower intensity (0.5)
    for ext_sq in EXTENDED_CENTER:
        if ext_sq not in CENTER_SQUARES and board.attackers(my_color, ext_sq):
            file_idx = chess.square_file(ext_sq)
            rank_idx = chess.square_rank(ext_sq)
            if flip:
                rank_idx = 7 - rank_idx
            planes[offset + 7, 7 - rank_idx, file_idx] = 0.5


def _get_king_zone(king_square: int) -> list[int]:
    """Get squares in the king's zone (king + adjacent squares)."""
    zone = [king_square]
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            if df == 0 and dr == 0:
                continue
            f, r = king_file + df, king_rank + dr
            if 0 <= f < 8 and 0 <= r < 8:
                zone.append(chess.square(f, r))
    return zone


def _is_passed_pawn(
    pawn_sq: int, color: chess.Color, opp_pawns: chess.SquareSet
) -> bool:
    """Check if a pawn is passed (no opposing pawns ahead on same or adjacent files)."""
    pawn_file = chess.square_file(pawn_sq)
    pawn_rank = chess.square_rank(pawn_sq)

    # Direction of advance
    if color == chess.WHITE:
        ahead_ranks = range(pawn_rank + 1, 8)
    else:
        ahead_ranks = range(pawn_rank - 1, -1, -1)

    # Check adjacent files
    for adj_file in [pawn_file - 1, pawn_file, pawn_file + 1]:
        if 0 <= adj_file < 8:
            for rank in ahead_ranks:
                sq = chess.square(adj_file, rank)
                if sq in opp_pawns:
                    return False
    return True


def _is_isolated_pawn(pawn_sq: int, friendly_pawns: chess.SquareSet) -> bool:
    """Check if a pawn is isolated (no friendly pawns on adjacent files)."""
    pawn_file = chess.square_file(pawn_sq)

    for adj_file in [pawn_file - 1, pawn_file + 1]:
        if 0 <= adj_file < 8:
            for rank in range(8):
                sq = chess.square(adj_file, rank)
                if sq in friendly_pawns:
                    return False
    return True


def _encode_tactical_features(
    board: chess.Board,
    planes: np.ndarray,
    offset: int,
    flip: bool = False,
) -> None:
    """
    Encode tactical features for improved pattern recognition.

    Planes (from offset):
        0: Pinned pieces - my pieces pinned to king/queen
        1: Hanging pieces - my undefended pieces under attack
        2: Attacking pieces - my pieces attacking higher-value targets
        3: Trapped pieces - my pieces with very limited mobility

    Args:
        board: Chess board.
        planes: Target array.
        offset: Starting plane index (68 for 72-plane encoding).
        flip: Whether perspective is flipped.
    """
    # Determine colors from perspective
    if flip:
        my_color = chess.BLACK
        opp_color = chess.WHITE
    else:
        my_color = board.turn
        opp_color = not board.turn

    # Helper to set a square in a plane
    def set_square(plane_idx: int, square: int) -> None:
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        if flip:
            rank_idx = 7 - rank_idx
        planes[plane_idx, 7 - rank_idx, file_idx] = 1.0

    # Piece values for comparison
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    # Plane 0: Pinned pieces - my pieces pinned to king
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == my_color:
            if board.is_pinned(my_color, square):
                set_square(offset, square)

    # Plane 1: Hanging pieces - my pieces attacked but not defended
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == my_color and piece.piece_type != chess.KING:
            attackers = board.attackers(opp_color, square)
            if attackers:
                defenders = board.attackers(my_color, square)
                if not defenders:
                    set_square(offset + 1, square)

    # Plane 2: Attacking pieces - my pieces attacking higher-value targets
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == my_color:
            my_value = piece_values.get(piece.piece_type, 0)
            attacks = board.attacks(square)
            for target_sq in attacks:
                target = board.piece_at(target_sq)
                if target and target.color == opp_color:
                    target_value = piece_values.get(target.piece_type, 0)
                    if target_value > my_value:
                        set_square(offset + 2, square)
                        break

    # Plane 3: Trapped pieces - my pieces with very limited mobility (< 2 moves)
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for square in board.pieces(piece_type, my_color):
            mobility = sum(1 for m in board.legal_moves if m.from_square == square)
            if mobility < 2:
                set_square(offset + 3, square)


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
    # is_repetition(n) = position has occurred n times total (including current)
    # Note: requires move stack, may return False for standalone boards
    try:
        if board.is_repetition(3):
            planes[offset + 8] = 1.0  # Threefold repetition (draw claimable)
        elif board.is_repetition(2):
            planes[offset + 8] = 0.5  # Position seen once before
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
    Encode a single position (convenience wrapper for 72 planes with no history).

    Args:
        board: Chess board.
        from_perspective: Encode from current player's perspective.

    Returns:
        numpy array of shape (72, 8, 8).
    """
    return encode_board_with_history([board], from_perspective)


def get_num_planes(history_length: int = DEFAULT_HISTORY_LENGTH) -> int:
    """
    Calculate the number of input planes for a given history length.

    Args:
        history_length: Number of historical positions (0 = current only).

    Returns:
        Total number of planes (72 for default history_length=3).
    """
    # (history_length + 1) positions × 12 planes + 12 metadata + 8 semantic + 4 tactical
    return (
        (history_length + 1) * PLANES_PER_POSITION
        + 12
        + SEMANTIC_PLANES
        + TACTICAL_PLANES
    )


def history_length_from_planes(num_planes: int) -> int:
    """
    Infer history length from number of input planes (72 planes only).

    Args:
        num_planes: Number of input planes.

    Returns:
        History length (3 for 72 planes).
    """
    # 72 planes = (history_length + 1) * 12 + 12 + 8 + 4
    # num_planes - 24 = (history_length + 1) * 12
    return (num_planes - 24) // 12 - 1


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
            numpy array of shape (72, 8, 8) for default history length.
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
