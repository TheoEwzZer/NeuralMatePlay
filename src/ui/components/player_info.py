"""
Player information widget.

Displays player name, avatar, captured pieces, and material advantage.
"""

import tkinter as tk
from typing import Optional, List

import chess

try:
    from ..styles import COLORS, FONTS, PIECE_UNICODE
except ImportError:
    from src.ui.styles import COLORS, FONTS, PIECE_UNICODE


# Piece values for material calculation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

# Unicode pieces for display (smaller versions)
CAPTURED_PIECE_CHARS = {
    chess.PAWN: "",
    chess.KNIGHT: "",
    chess.BISHOP: "",
    chess.ROOK: "",
    chess.QUEEN: "",
}


class PlayerInfo(tk.Frame):
    """
    Widget displaying player information.

    Shows avatar, name, captured pieces, and material advantage.
    Chess.com-inspired design.
    """

    def __init__(
        self,
        parent: tk.Widget,
        color: chess.Color,
        name: str = "Player",
        width: int = 200,
        **kwargs,
    ):
        bg = kwargs.pop("bg", COLORS["bg_secondary"])
        super().__init__(parent, bg=bg, width=width, **kwargs)

        self.configure(
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )

        self._color = color
        self._name = name
        self._is_active = False
        self._captured: list[chess.PieceType] = []
        self._material_diff = 0

        self._create_widgets(bg)

    def _create_widgets(self, bg: str) -> None:
        """Create the player info widgets."""
        # Main container with padding
        container = tk.Frame(self, bg=bg)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Top row: Avatar + Name
        top_row = tk.Frame(container, bg=bg)
        top_row.pack(fill=tk.X)

        # Avatar (colored circle with piece icon)
        self._avatar_canvas = tk.Canvas(
            top_row,
            width=36,
            height=36,
            bg=bg,
            highlightthickness=0,
        )
        self._avatar_canvas.pack(side=tk.LEFT, padx=(0, 10))
        self._draw_avatar()

        # Name and status
        name_frame = tk.Frame(top_row, bg=bg)
        name_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._name_label = tk.Label(
            name_frame,
            text=self._name,
            font=FONTS["body_bold"],
            fg=COLORS["text_primary"],
            bg=bg,
            anchor="w",
        )
        self._name_label.pack(fill=tk.X)

        # Color indicator
        color_text = "White" if self._color == chess.WHITE else "Black"
        self._color_label = tk.Label(
            name_frame,
            text=color_text,
            font=("Segoe UI", 9),
            fg=COLORS["text_muted"],
            bg=bg,
            anchor="w",
        )
        self._color_label.pack(fill=tk.X)

        # Timer display (for time control mode)
        self._timer_label = tk.Label(
            top_row,
            text="--:--",
            font=("Consolas", 14, "bold"),
            fg=COLORS["text_primary"],
            bg=COLORS["bg_tertiary"],
            padx=8,
            pady=2,
        )
        self._timer_label.pack(side=tk.RIGHT, padx=(5, 0))

        # Active turn indicator
        self._turn_indicator = tk.Label(
            top_row,
            text="",
            font=("Segoe UI", 10),
            fg=COLORS["accent"],
            bg=bg,
        )
        self._turn_indicator.pack(side=tk.RIGHT)

        # Separator
        sep = tk.Frame(container, bg=COLORS["border"], height=1)
        sep.pack(fill=tk.X, pady=8)

        # Bottom row: Captured pieces
        bottom_row = tk.Frame(container, bg=bg)
        bottom_row.pack(fill=tk.X)

        # Captured pieces display
        self._captured_frame = tk.Frame(bottom_row, bg=bg)
        self._captured_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._captured_label = tk.Label(
            self._captured_frame,
            text="",
            font=("Segoe UI", 14),
            fg=COLORS["text_secondary"],
            bg=bg,
            anchor="w",
        )
        self._captured_label.pack(side=tk.LEFT)

        # Material advantage
        self._material_label = tk.Label(
            bottom_row,
            text="",
            font=FONTS["body_bold"],
            fg=COLORS["success"],
            bg=bg,
        )
        self._material_label.pack(side=tk.RIGHT)

    def _draw_avatar(self) -> None:
        """Draw the player avatar."""
        canvas = self._avatar_canvas
        canvas.delete("all")

        # Background circle
        if self._color == chess.WHITE:
            fill_color = "#f0f0f0"
            piece_color = "#333333"
            outline_color = "#cccccc"
        else:
            fill_color = "#333333"
            piece_color = "#f0f0f0"
            outline_color = "#555555"

        # Outer ring (active indicator)
        if self._is_active:
            canvas.create_oval(
                0,
                0,
                36,
                36,
                fill=COLORS["accent"],
                outline="",
            )
            canvas.create_oval(
                3,
                3,
                33,
                33,
                fill=fill_color,
                outline=outline_color,
                width=1,
            )
        else:
            canvas.create_oval(
                2,
                2,
                34,
                34,
                fill=fill_color,
                outline=outline_color,
                width=1,
            )

        # King piece icon
        king_char = "" if self._color == chess.WHITE else ""
        canvas.create_text(
            18,
            18,
            text=king_char,
            font=("Segoe UI Symbol", 16),
            fill=piece_color,
        )

    def set_name(self, name: str) -> None:
        """Update player name."""
        self._name = name
        self._name_label.configure(text=name)

    def set_timer(self, time_str: str, color: str = None) -> None:
        """
        Update the timer display.

        Args:
            time_str: Formatted time string (e.g., "05:00")
            color: Optional text color for urgency indication
        """
        self._timer_label.configure(text=time_str)
        if color:
            self._timer_label.configure(fg=color)

    def set_active(self, is_active: bool) -> None:
        """Set whether this player's turn is active."""
        self._is_active = is_active
        self._draw_avatar()

        if is_active:
            self._turn_indicator.configure(text="")
            # Highlight timer background when active
            self._timer_label.configure(
                bg=COLORS.get("player_active_bg", COLORS["accent"])
            )
        else:
            self._turn_indicator.configure(text="")
            self._timer_label.configure(bg=COLORS["bg_tertiary"])

    def set_captured(
        self,
        captured_pieces: list[chess.PieceType],
        material_diff: int,
    ) -> None:
        """
        Update captured pieces display.

        Args:
            captured_pieces: List of piece types this player has captured
            material_diff: Material difference (positive = advantage for this player)
        """
        self._captured = captured_pieces
        self._material_diff = material_diff

        # Build captured pieces string (grouped by type)
        piece_counts = {}
        for piece_type in captured_pieces:
            piece_counts[piece_type] = piece_counts.get(piece_type, 0) + 1

        # Order: Queen, Rook, Bishop, Knight, Pawn
        piece_order = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]

        # Get piece characters based on what color this player captured (opposite color pieces)
        opponent_color = not self._color
        captured_str = ""
        for piece_type in piece_order:
            count = piece_counts.get(piece_type, 0)
            if count > 0:
                # Use unicode piece of opponent's color
                piece_char = self._get_piece_char(piece_type, opponent_color)
                captured_str += piece_char * count

        self._captured_label.configure(text=captured_str)

        # Update material advantage
        if material_diff > 0:
            self._material_label.configure(
                text=f"+{material_diff}",
                fg=COLORS["success"],
            )
        elif material_diff < 0:
            self._material_label.configure(
                text=str(material_diff),
                fg=COLORS["error"],
            )
        else:
            self._material_label.configure(text="")

    def _get_piece_char(self, piece_type: chess.PieceType, color: chess.Color) -> str:
        """Get the unicode character for a piece."""
        piece_chars = {
            (chess.PAWN, chess.WHITE): "",
            (chess.KNIGHT, chess.WHITE): "",
            (chess.BISHOP, chess.WHITE): "",
            (chess.ROOK, chess.WHITE): "",
            (chess.QUEEN, chess.WHITE): "",
            (chess.PAWN, chess.BLACK): "",
            (chess.KNIGHT, chess.BLACK): "",
            (chess.BISHOP, chess.BLACK): "",
            (chess.ROOK, chess.BLACK): "",
            (chess.QUEEN, chess.BLACK): "",
        }
        return piece_chars.get((piece_type, color), "?")

    def update_from_board(self, board: chess.Board) -> None:
        """
        Update all info from board state.

        Args:
            board: Current chess board
        """
        # Update active turn
        is_my_turn = board.turn == self._color
        self.set_active(is_my_turn)

        # Calculate captured pieces
        # Start with full set, subtract what's on board
        full_set = {
            chess.PAWN: 8,
            chess.KNIGHT: 2,
            chess.BISHOP: 2,
            chess.ROOK: 2,
            chess.QUEEN: 1,
        }

        # What opponent has lost (I captured)
        opponent_color = not self._color
        my_captures = []
        for piece_type, full_count in full_set.items():
            on_board = len(board.pieces(piece_type, opponent_color))
            captured = full_count - on_board
            for _ in range(max(0, captured)):
                my_captures.append(piece_type)

        # Calculate material difference
        my_material = sum(
            len(board.pieces(pt, self._color)) * PIECE_VALUES.get(pt, 0)
            for pt in PIECE_VALUES
        )
        opponent_material = sum(
            len(board.pieces(pt, opponent_color)) * PIECE_VALUES.get(pt, 0)
            for pt in PIECE_VALUES
        )
        material_diff = my_material - opponent_material

        self.set_captured(my_captures, material_diff)
