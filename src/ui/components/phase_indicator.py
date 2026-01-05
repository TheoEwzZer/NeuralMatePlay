"""
Game phase indicator widget.

Displays the current game phase (Opening, Middlegame, Endgame).
"""

import tkinter as tk
from typing import Optional

import chess

try:
    from ..styles import COLORS, FONTS
except ImportError:
    from src.ui.styles import COLORS, FONTS

try:
    from src.chess_encoding.board_utils import get_game_phase
except ImportError:
    # Fallback implementation
    def get_game_phase(board: chess.Board) -> str:
        piece_count = sum(
            len(board.pieces(pt, c))
            for pt in chess.PIECE_TYPES
            for c in [chess.WHITE, chess.BLACK]
        )
        if piece_count >= 28:
            return "opening"
        elif piece_count <= 12:
            return "endgame"
        return "middlegame"


# Phase display configuration
PHASE_CONFIG = {
    "opening": {
        "label": "Opening",
        "color": "#4ade80",  # Green
        "icon": "",
    },
    "middlegame": {
        "label": "Middlegame",
        "color": "#fbbf24",  # Yellow
        "icon": "",
    },
    "endgame": {
        "label": "Endgame",
        "color": "#f87171",  # Red
        "icon": "",
    },
}


class PhaseIndicator(tk.Frame):
    """
    Widget showing the current game phase.

    Displays Opening, Middlegame, or Endgame with color-coded indicator.
    """

    def __init__(
        self,
        parent: tk.Widget,
        width: int = 180,
        **kwargs,
    ):
        bg = kwargs.pop("bg", COLORS["bg_secondary"])
        super().__init__(parent, bg=bg, **kwargs)

        self.configure(
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )

        self._current_phase = "opening"
        self._create_widgets(bg)

    def _create_widgets(self, bg: str) -> None:
        """Create the indicator widgets."""
        # Title
        title = tk.Label(
            self,
            text="PHASE",
            font=("Segoe UI", 9, "bold"),
            fg=COLORS["text_muted"],
            bg=bg,
        )
        title.pack(pady=(8, 2))

        # Phase indicator frame
        self._indicator_frame = tk.Frame(self, bg=bg)
        self._indicator_frame.pack(pady=(2, 8), padx=10, fill=tk.X)

        # Colored dot
        self._dot_canvas = tk.Canvas(
            self._indicator_frame,
            width=12,
            height=12,
            bg=bg,
            highlightthickness=0,
        )
        self._dot_canvas.pack(side=tk.LEFT, padx=(0, 8))

        # Phase label
        self._phase_label = tk.Label(
            self._indicator_frame,
            text="Opening",
            font=FONTS["body_bold"],
            fg=COLORS["text_primary"],
            bg=bg,
        )
        self._phase_label.pack(side=tk.LEFT)

        # Draw initial dot
        self._draw_dot(PHASE_CONFIG["opening"]["color"])

    def _draw_dot(self, color: str) -> None:
        """Draw the phase indicator dot."""
        self._dot_canvas.delete("all")
        # Outer glow
        self._dot_canvas.create_oval(
            1,
            1,
            11,
            11,
            fill=color,
            outline="",
        )
        # Inner highlight
        self._dot_canvas.create_oval(
            3,
            3,
            7,
            7,
            fill=self._lighten_color(color),
            outline="",
        )

    def _lighten_color(self, color: str) -> str:
        """Create a lighter version of a color for highlight effect."""
        # Parse hex color
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        # Lighten
        r = min(255, r + 60)
        g = min(255, g + 60)
        b = min(255, b + 60)

        return f"#{r:02x}{g:02x}{b:02x}"

    def update_phase(self, board: chess.Board) -> None:
        """
        Update the phase indicator based on board position.

        Args:
            board: Current chess board state
        """
        phase = get_game_phase(board)

        if phase != self._current_phase:
            self._current_phase = phase
            config = PHASE_CONFIG.get(phase, PHASE_CONFIG["middlegame"])

            self._phase_label.configure(text=config["label"])
            self._draw_dot(config["color"])

    def set_phase(self, phase: str) -> None:
        """
        Manually set the phase display.

        Args:
            phase: One of "opening", "middlegame", "endgame"
        """
        if phase in PHASE_CONFIG:
            self._current_phase = phase
            config = PHASE_CONFIG[phase]
            self._phase_label.configure(text=config["label"])
            self._draw_dot(config["color"])

    def get_phase(self) -> str:
        """Get the currently displayed phase."""
        return self._current_phase
