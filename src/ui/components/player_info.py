"""
Player information widget.

Horizontal bar shown above / below the board: avatar, name, captured
pieces, material advantage and optional clock.
"""

import tkinter as tk


import chess

try:
    from ..styles import COLORS, FONTS, PIECE_UNICODE, load_piece_image
except ImportError:
    from src.ui.styles import COLORS, FONTS, PIECE_UNICODE, load_piece_image


# Piece values for material calculation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

# Display order for captured pieces (most valuable first)
CAPTURE_ORDER = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]

CAPTURED_ICON_SIZE = 20
AVATAR_SIZE = 34
BAR_HEIGHT = 62
ACTIVE_BG = "#20252e"


class PlayerInfo(tk.Frame):
    """
    Widget displaying player information as a horizontal bar.

    Shows avatar, name, captured pieces, material advantage and clock.
    The bar is highlighted when it is this player's turn.
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
        super().__init__(parent, bg=bg, width=width, height=BAR_HEIGHT, **kwargs)
        # Fixed height: content changes (badge, captures) never resize the bar
        self.pack_propagate(False)
        self._bg = bg
        self._style_applied = False

        self._color = color
        self._name = name
        self._is_active = False
        self._captured: list[chess.PieceType] = []
        self._material_diff = 0
        self._timer_visible = False

        self._create_widgets(bg)

    def _create_widgets(self, bg: str) -> None:
        """Create the player info widgets."""
        # Left stripe that lights up on this player's turn
        self._stripe = tk.Frame(self, bg=bg, width=4)
        self._stripe.pack(side=tk.LEFT, fill=tk.Y)

        container = tk.Frame(self, bg=bg)
        container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 12), pady=7)
        self._container = container

        # Avatar: king image of this player's color
        self._avatar = tk.Label(container, bg=COLORS["bg_tertiary"], bd=0)
        avatar_img = load_piece_image("K" if self._color else "k", AVATAR_SIZE - 6)
        if avatar_img is not None:
            self._avatar.configure(
                image=avatar_img, width=AVATAR_SIZE, height=AVATAR_SIZE
            )
        else:
            self._avatar.configure(
                text=PIECE_UNICODE["K" if self._color else "k"],
                font=("Segoe UI Symbol", 18),
                fg=COLORS["text_primary"],
                width=2,
            )
        self._avatar.pack(side=tk.LEFT, padx=(0, 10))

        # Name + subtitle
        name_frame = tk.Frame(container, bg=bg)
        name_frame.pack(side=tk.LEFT)

        name_row = tk.Frame(name_frame, bg=bg)
        name_row.pack(fill=tk.X)

        self._name_label = tk.Label(
            name_row,
            text=self._name,
            font=FONTS["body_bold"],
            fg=COLORS["text_primary"],
            bg=bg,
            anchor="w",
        )
        self._name_label.pack(side=tk.LEFT)

        # "À jouer" badge (visible on this player's turn)
        self._turn_badge = tk.Label(
            name_row,
            text=" au trait ",
            font=FONTS["caption_bold"],
            fg=bg,
            bg=bg,
        )
        self._turn_badge.pack(side=tk.LEFT, padx=(8, 0))

        # Captured pieces + material
        captures_row = tk.Frame(name_frame, bg=bg)
        captures_row.pack(fill=tk.X, pady=(1, 0))

        self._captured_frame = tk.Frame(captures_row, bg=bg)
        self._captured_frame.pack(side=tk.LEFT)

        self._material_label = tk.Label(
            captures_row,
            text="",
            font=FONTS["small_bold"],
            fg=COLORS["text_secondary"],
            bg=bg,
        )
        self._material_label.pack(side=tk.LEFT, padx=(4, 0))

        self._color_label = tk.Label(
            captures_row,
            text="Blancs" if self._color == chess.WHITE else "Noirs",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=bg,
        )
        self._color_label.pack(side=tk.LEFT)

        # Timer display (only shown with a time control)
        self._timer_label = tk.Label(
            container,
            text="--:--",
            font=("Consolas", 16, "bold"),
            fg=COLORS["text_primary"],
            bg=COLORS["bg_tertiary"],
            padx=12,
            pady=3,
        )

    def _set_bg(self, bg: str) -> None:
        """Recolor the bar background (recursively, except badges)."""
        skip = {self._turn_badge, self._timer_label, self._avatar}

        def recolor(widget: tk.Misc) -> None:
            if widget in skip:
                return
            try:
                widget.configure(bg=bg)
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                recolor(child)

        recolor(self._container)
        self.configure(bg=bg)

    def set_name(self, name: str) -> None:
        """Update player name."""
        self._name = name
        self._name_label.configure(text=name)

    def show_timer(self, visible: bool) -> None:
        """Show or hide the clock."""
        if visible == self._timer_visible:
            return
        self._timer_visible = visible
        if visible:
            self._timer_label.pack(side=tk.RIGHT)
        else:
            self._timer_label.pack_forget()

    def set_timer(self, time_str: str, color: str = None) -> None:
        """
        Update the timer display.

        Args:
            time_str: Formatted time string (e.g., "05:00")
            color: Optional text color for urgency indication
        """
        self.show_timer(True)
        self._timer_label.configure(text=time_str)
        if color:
            self._timer_label.configure(fg=color)

    def set_active(self, is_active: bool) -> None:
        """Set whether this player's turn is active."""
        # Only colors change here, never the geometry: re-packing widgets
        # makes the whole column relayout and the bar flicker.
        if is_active == self._is_active and self._style_applied:
            return
        self._is_active = is_active
        self._style_applied = True

        if is_active:
            bar_bg = ACTIVE_BG
            self._stripe.configure(bg=COLORS["accent"])
            self._set_bg(bar_bg)
            self._turn_badge.configure(fg="#ffffff", bg=COLORS["accent"])
            self._timer_label.configure(bg=COLORS["player_active_bg"])
        else:
            bar_bg = self._bg
            self._stripe.configure(bg=self._bg)
            self._set_bg(bar_bg)
            # Invisible but still packed (keeps the layout stable)
            self._turn_badge.configure(fg=bar_bg, bg=bar_bg)
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
        if captured_pieces == self._captured and material_diff == self._material_diff:
            return
        self._captured = list(captured_pieces)
        self._material_diff = material_diff

        for widget in self._captured_frame.winfo_children():
            widget.destroy()

        bg = self._captured_frame.cget("bg")
        opponent_color = not self._color
        sorted_pieces = sorted(captured_pieces, key=CAPTURE_ORDER.index)
        for piece_type in sorted_pieces:
            symbol = chess.piece_symbol(piece_type)
            symbol = symbol.upper() if opponent_color == chess.WHITE else symbol
            img = load_piece_image(symbol, CAPTURED_ICON_SIZE)
            if img is not None:
                lbl = tk.Label(self._captured_frame, image=img, bg=bg, bd=0)
            else:
                lbl = tk.Label(
                    self._captured_frame,
                    text=PIECE_UNICODE.get(symbol, "?"),
                    font=("Segoe UI Symbol", 12),
                    fg=COLORS["text_secondary"],
                    bg=bg,
                )
            lbl.pack(side=tk.LEFT, padx=0)

        # Update material advantage
        if material_diff > 0:
            self._material_label.configure(text=f"+{material_diff}")
        else:
            self._material_label.configure(text="")

        # Show the color name when nothing has been captured yet
        if sorted_pieces or material_diff > 0:
            self._color_label.pack_forget()
        else:
            self._color_label.pack(side=tk.LEFT)

    def update_from_board(self, board: chess.Board) -> None:
        """
        Update all info from board state.

        Args:
            board: Current chess board
        """
        # Update active turn
        is_my_turn = board.turn == self._color and not board.is_game_over()
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
