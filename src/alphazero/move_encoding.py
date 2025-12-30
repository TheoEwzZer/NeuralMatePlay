"""
Move encoding/decoding for AlphaZero-style chess.

Uses a "queen-like moves" encoding scheme:
- 64 from-squares × 73 move types = 4672 total indices

Move types (73 per square):
- 56 queen-like moves: 8 directions × 7 distances
- 8 knight moves
- 9 underpromotions: 3 piece types × 3 directions
"""

import chess
import numpy as np
from typing import Optional


# Directions for queen-like moves (dx, dy)
QUEEN_DIRECTIONS = [
    (0, 1),  # North
    (1, 1),  # NE
    (1, 0),  # East
    (1, -1),  # SE
    (0, -1),  # South
    (-1, -1),  # SW
    (-1, 0),  # West
    (-1, 1),  # NW
]

# Knight move offsets (dx, dy)
KNIGHT_MOVES = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]

# Underpromotion pieces (queen promotion is encoded as regular move)
UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

# Underpromotion directions: -1 (left capture), 0 (straight), 1 (right capture)
UNDERPROMOTION_DIRECTIONS = [-1, 0, 1]

# Move encoding size: 64 squares × 73 move types
NUM_MOVE_TYPES = 56 + 8 + 9  # queen moves + knight moves + underpromotions
MOVE_ENCODING_SIZE = 64 * NUM_MOVE_TYPES  # 4672

# Precomputed lookup tables (initialized lazily)
_move_to_index: dict[tuple, int] = {}
_index_to_move: dict[int, tuple] = {}
_initialized = False


def _init_lookup_tables():
    """Initialize the move encoding lookup tables."""
    global _move_to_index, _index_to_move, _initialized

    if _initialized:
        return

    for from_sq in range(64):
        from_file = from_sq % 8
        from_rank = from_sq // 8

        base_idx = from_sq * NUM_MOVE_TYPES
        move_type = 0

        # Queen-like moves: 8 directions × 7 distances
        for direction_idx, (dx, dy) in enumerate(QUEEN_DIRECTIONS):
            for distance in range(1, 8):
                to_file = from_file + dx * distance
                to_rank = from_rank + dy * distance

                if 0 <= to_file < 8 and 0 <= to_rank < 8:
                    to_sq = to_rank * 8 + to_file
                    idx = base_idx + direction_idx * 7 + (distance - 1)

                    # Store for both directions
                    key = (from_sq, to_sq, None)  # None = no underpromotion
                    _move_to_index[key] = idx
                    _index_to_move[idx] = key

                move_type += 1

        # Knight moves
        for knight_idx, (dx, dy) in enumerate(KNIGHT_MOVES):
            to_file = from_file + dx
            to_rank = from_rank + dy

            if 0 <= to_file < 8 and 0 <= to_rank < 8:
                to_sq = to_rank * 8 + to_file
                idx = base_idx + 56 + knight_idx

                key = (from_sq, to_sq, None)
                _move_to_index[key] = idx
                _index_to_move[idx] = key

        # Underpromotions (only from 7th rank for white / 2nd rank for black)
        if from_rank == 6:  # White pawn on 7th rank
            for promo_idx, promo_piece in enumerate(UNDERPROMOTION_PIECES):
                for dir_idx, dx in enumerate(UNDERPROMOTION_DIRECTIONS):
                    to_file = from_file + dx
                    to_rank = 7  # Promotion rank for white

                    if 0 <= to_file < 8:
                        to_sq = to_rank * 8 + to_file
                        idx = base_idx + 64 + promo_idx * 3 + dir_idx

                        key = (from_sq, to_sq, promo_piece)
                        _move_to_index[key] = idx
                        _index_to_move[idx] = key

        if from_rank == 1:  # Black pawn on 2nd rank
            for promo_idx, promo_piece in enumerate(UNDERPROMOTION_PIECES):
                for dir_idx, dx in enumerate(UNDERPROMOTION_DIRECTIONS):
                    to_file = from_file + dx
                    to_rank = 0  # Promotion rank for black

                    if 0 <= to_file < 8:
                        to_sq = to_rank * 8 + to_file
                        idx = base_idx + 64 + promo_idx * 3 + dir_idx

                        key = (from_sq, to_sq, promo_piece)
                        _move_to_index[key] = idx
                        _index_to_move[idx] = key

    _initialized = True


def encode_move(move: chess.Move) -> Optional[int]:
    """
    Encode a chess move to a policy index.

    Args:
        move: A chess.Move object.

    Returns:
        Integer index in [0, MOVE_ENCODING_SIZE) or None if encoding fails.
    """
    _init_lookup_tables()

    from_sq = move.from_square
    to_sq = move.to_square

    # Handle promotion
    promo = None
    if move.promotion is not None and move.promotion != chess.QUEEN:
        promo = move.promotion

    key = (from_sq, to_sq, promo)

    # Direct lookup
    if key in _move_to_index:
        return _move_to_index[key]

    # For queen promotion, treat as regular move
    if move.promotion == chess.QUEEN:
        key = (from_sq, to_sq, None)
        if key in _move_to_index:
            return _move_to_index[key]

    # Fallback: compute directly for edge cases
    from_file = from_sq % 8
    from_rank = from_sq // 8
    to_file = to_sq % 8
    to_rank = to_sq // 8

    dx = to_file - from_file
    dy = to_rank - from_rank

    base_idx = from_sq * NUM_MOVE_TYPES

    # Check if knight move
    if (abs(dx), abs(dy)) in [(1, 2), (2, 1)]:
        for knight_idx, (kdx, kdy) in enumerate(KNIGHT_MOVES):
            if dx == kdx and dy == kdy:
                return base_idx + 56 + knight_idx

    # Queen-like move
    if dx == 0 or dy == 0 or abs(dx) == abs(dy):
        # Normalize direction
        if dx != 0:
            norm_dx = dx // abs(dx)
        else:
            norm_dx = 0
        if dy != 0:
            norm_dy = dy // abs(dy)
        else:
            norm_dy = 0

        distance = max(abs(dx), abs(dy))

        for dir_idx, (ddx, ddy) in enumerate(QUEEN_DIRECTIONS):
            if norm_dx == ddx and norm_dy == ddy:
                return base_idx + dir_idx * 7 + (distance - 1)

    return None


def encode_move_from_perspective(move: chess.Move, flip: bool = False) -> Optional[int]:
    """
    Encode a move, optionally flipping for black's perspective.

    Args:
        move: A chess.Move object.
        flip: If True, flip the move vertically (for black's perspective).

    Returns:
        Integer index or None.
    """
    if not flip:
        return encode_move(move)

    # Flip squares vertically
    from_sq = chess.square_mirror(move.from_square)
    to_sq = chess.square_mirror(move.to_square)

    flipped_move = chess.Move(from_sq, to_sq, promotion=move.promotion)
    return encode_move(flipped_move)


def decode_move(index: int, board: chess.Board) -> Optional[chess.Move]:
    """
    Decode a policy index to a chess move.

    Args:
        index: Policy index in [0, MOVE_ENCODING_SIZE).
        board: Current board state (for validation).

    Returns:
        chess.Move if valid and legal, None otherwise.
    """
    _init_lookup_tables()

    if index < 0 or index >= MOVE_ENCODING_SIZE:
        return None

    if index not in _index_to_move:
        return None

    from_sq, to_sq, promo = _index_to_move[index]

    # Create move
    if promo is not None:
        move = chess.Move(from_sq, to_sq, promotion=promo)
    else:
        # Check if this is actually a pawn promotion (needs queen)
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = to_sq // 8
            if to_rank == 7 or to_rank == 0:
                move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
            else:
                move = chess.Move(from_sq, to_sq)
        else:
            move = chess.Move(from_sq, to_sq)

    # Validate legality
    if move in board.legal_moves:
        return move

    return None


def decode_move_from_perspective(
    index: int, board: chess.Board, flip: bool = False
) -> Optional[chess.Move]:
    """
    Decode a policy index, optionally flipping from black's perspective.

    Args:
        index: Policy index.
        board: Current board state.
        flip: If True, flip the decoded move.

    Returns:
        chess.Move if valid, None otherwise.
    """
    _init_lookup_tables()

    if index < 0 or index >= MOVE_ENCODING_SIZE:
        return None

    if index not in _index_to_move:
        return None

    from_sq, to_sq, promo = _index_to_move[index]

    if flip:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    # Create move
    piece = board.piece_at(from_sq)
    if promo is not None:
        move = chess.Move(from_sq, to_sq, promotion=promo)
    elif piece and piece.piece_type == chess.PAWN:
        to_rank = to_sq // 8
        if to_rank == 7 or to_rank == 0:
            move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        else:
            move = chess.Move(from_sq, to_sq)
    else:
        move = chess.Move(from_sq, to_sq)

    if move in board.legal_moves:
        return move

    return None


def flip_policy(policy: np.ndarray) -> np.ndarray:
    """
    Flip a policy array from one perspective to another.

    This mirrors the policy vertically, converting moves from
    white's perspective to black's or vice versa.

    Optimized: Uses np.nonzero() to only iterate over non-zero elements
    (~50-100 typically) instead of all 4672 elements.

    Args:
        policy: Policy array of shape (MOVE_ENCODING_SIZE,).

    Returns:
        Flipped policy array of same shape.
    """
    _init_lookup_tables()

    flipped = np.zeros_like(policy)

    # Only iterate over non-zero elements (much faster than enumerate over all 4672)
    nonzero_indices = np.nonzero(policy)[0]

    for idx in nonzero_indices:
        if idx not in _index_to_move:
            continue

        from_sq, to_sq, promo = _index_to_move[idx]

        # Flip squares
        flipped_from = chess.square_mirror(from_sq)
        flipped_to = chess.square_mirror(to_sq)

        key = (flipped_from, flipped_to, promo)
        if key in _move_to_index:
            flipped_idx = _move_to_index[key]
            flipped[flipped_idx] = policy[idx]

    return flipped


def get_legal_move_mask(
    board: chess.Board, from_perspective: bool = True
) -> np.ndarray:
    """
    Get a mask of legal moves for the current position.

    Optimized: Collects all indices first, then uses batch numpy indexing
    for a single vectorized assignment instead of ~35 individual assignments.

    Args:
        board: Current board state.
        from_perspective: If True, encode from current player's perspective.

    Returns:
        Boolean mask of shape (MOVE_ENCODING_SIZE,).
    """
    mask = np.zeros(MOVE_ENCODING_SIZE, dtype=np.float32)
    flip = from_perspective and board.turn == chess.BLACK

    # Collect all valid indices first
    indices = []
    for move in board.legal_moves:
        idx = encode_move_from_perspective(move, flip)
        if idx is not None:
            indices.append(idx)

    # Batch assignment (faster than individual assignments)
    if indices:
        mask[indices] = 1.0

    return mask


def policy_to_moves(
    policy: np.ndarray,
    board: chess.Board,
    from_perspective: bool = True,
    top_k: int = 10,
) -> list[tuple[chess.Move, float]]:
    """
    Convert a policy array to a list of (move, probability) tuples.

    Args:
        policy: Policy array of shape (MOVE_ENCODING_SIZE,).
        board: Current board state.
        from_perspective: If True, decode from current player's perspective.
        top_k: Number of top moves to return.

    Returns:
        List of (move, probability) tuples, sorted by probability.
    """
    flip = from_perspective and board.turn == chess.BLACK

    moves_probs = []
    for move in board.legal_moves:
        idx = encode_move_from_perspective(move, flip)
        if idx is not None and idx < len(policy):
            prob = policy[idx]
            moves_probs.append((move, float(prob)))

    # Sort by probability descending
    moves_probs.sort(key=lambda x: x[1], reverse=True)

    return moves_probs[:top_k]
