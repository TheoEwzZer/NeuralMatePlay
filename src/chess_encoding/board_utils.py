"""Board utility functions for chess analysis."""

import chess


PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    # King has no material value
    chess.KING: 0,
}


def get_piece_value(piece_type: chess.PieceType) -> int:
    """Get the material value of a piece type."""
    return PIECE_VALUES.get(piece_type, 0)


def get_raw_material_diff(board: chess.Board) -> int:
    """
    Calculate material difference from White's perspective.

    Returns:
        Positive if White has more material, negative if Black has more.
    """
    material = 0
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        white_count: int = len(board.pieces(piece_type, chess.WHITE))
        black_count: int = len(board.pieces(piece_type, chess.BLACK))
        material += (white_count - black_count) * PIECE_VALUES[piece_type]
    return material


def get_material_count(board: chess.Board, color: chess.Color) -> int:
    """Get total material for a color (excluding king)."""
    total = 0
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        total += len(board.pieces(piece_type, color)) * PIECE_VALUES[piece_type]
    return total


def is_endgame(board: chess.Board) -> bool:
    """
    Determine if the position is an endgame.

    Endgame is defined as:
    - Both sides have no queens, or
    - Every side which has a queen has at most one minor piece
    """
    white_queens: int = len(board.pieces(chess.QUEEN, chess.WHITE))
    black_queens: int = len(board.pieces(chess.QUEEN, chess.BLACK))

    if white_queens == 0 and black_queens == 0:
        return True

    white_material: int = get_material_count(board, chess.WHITE)
    black_material: int = get_material_count(board, chess.BLACK)

    return white_material <= 13 and black_material <= 13


def is_quiet_position(board: chess.Board) -> bool:
    """Check if position is quiet (no captures or checks available)."""
    for move in board.legal_moves:
        if board.is_capture(move):
            return False
        board.push(move)
        in_check: bool = board.is_check()
        board.pop()
        if in_check:
            return False
    return True


def count_pieces(board: chess.Board) -> int:
    """Count total pieces on the board."""
    count = 0
    for piece_type in chess.PIECE_TYPES:
        count += len(board.pieces(piece_type, chess.WHITE))
        count += len(board.pieces(piece_type, chess.BLACK))
    return count


def get_game_phase(board: chess.Board) -> str:
    """
    Determine game phase: opening, middlegame, or endgame.
    """
    piece_count: int = count_pieces(board)

    # Most pieces still on board
    if piece_count >= 28:
        return "opening"
    elif is_endgame(board):
        return "endgame"
    else:
        return "middlegame"
