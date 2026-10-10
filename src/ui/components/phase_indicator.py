"""
Game phase indicator widget.

Displays the current game phase (Opening, Middlegame, Endgame).
"""

import tkinter as tk
from typing import Optional

import chess

try:
    from ..styles import COLORS, FONTS, Tooltip
except ImportError:
    from src.ui.styles import COLORS, FONTS, Tooltip

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


# Phase display configuration (label, color and a beginner tip)
PHASE_CONFIG = {
    "opening": {
        "label": "Début de partie",
        "color": "#3ecf8e",
        "tip": "Sortez vos pièces, contrôlez le centre et mettez votre roi à l'abri.",
    },
    "middlegame": {
        "label": "Milieu de partie",
        "color": "#f5b544",
        "tip": "Cherchez des cibles : pièces mal protégées, roi exposé.",
    },
    "endgame": {
        "label": "Finale",
        "color": "#f25f5c",
        "tip": "Activez votre roi et poussez vos pions vers la promotion.",
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
        show_tip: bool = True,
        **kwargs,
    ):
        bg = kwargs.pop("bg", COLORS["bg_secondary"])
        super().__init__(parent, bg=bg, **kwargs)

        self._current_phase = "opening"
        self._show_tip = show_tip
        self._create_widgets(bg)

    def _create_widgets(self, bg: str) -> None:
        """Create the indicator widgets."""
        # Title
        title = tk.Label(
            self,
            text="PHASE DE JEU",
            font=FONTS["caption_bold"],
            fg=COLORS["text_muted"],
            bg=bg,
            anchor="w",
        )
        title.pack(fill=tk.X)

        # Phase indicator frame
        self._indicator_frame = tk.Frame(self, bg=bg)
        self._indicator_frame.pack(fill=tk.X, pady=(2, 0))

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
            text=PHASE_CONFIG["opening"]["label"],
            font=FONTS["small_bold"],
            fg=COLORS["text_primary"],
            bg=bg,
        )
        self._phase_label.pack(side=tk.LEFT)

        # Beginner tip
        self._tip_label = tk.Label(
            self,
            text=PHASE_CONFIG["opening"]["tip"],
            font=FONTS["caption"],
            fg=COLORS["text_secondary"],
            bg=bg,
            wraplength=240,
            justify=tk.LEFT,
            anchor="w",
        )
        if self._show_tip:
            self._tip_label.pack(fill=tk.X, pady=(3, 0))
        # The tip is always available on hover
        self._tooltip = Tooltip(self, PHASE_CONFIG["opening"]["tip"], delay=200)

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
            self._tip_label.configure(text=config["tip"])
            self._tooltip.update_text(config["tip"])
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
            self._tip_label.configure(text=config["tip"])
            self._tooltip.update_text(config["tip"])
            self._draw_dot(config["color"])

    def get_phase(self) -> str:
        """Get the currently displayed phase."""
        return self._current_phase
