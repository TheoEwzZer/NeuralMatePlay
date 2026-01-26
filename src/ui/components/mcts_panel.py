"""
MCTS statistics panel widget.

Displays search statistics for top candidate moves after AI move.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

import chess
import numpy as np

try:
    from ..styles import COLORS, FONTS
except ImportError:
    from src.ui.styles import COLORS, FONTS


@dataclass
class MCTSMoveStats:
    """Statistics for a single move from MCTS search."""

    move: chess.Move
    san: str  # Standard algebraic notation
    visits: int
    q_value: float  # Value estimate (-1 to +1)
    prior: float  # Policy prior probability
    wdl: np.ndarray = field(
        default_factory=lambda: np.array([0.33, 0.34, 0.33], dtype=np.float32)
    )  # [P(win), P(draw), P(loss)]
    our_mate_in: Optional[int] = None  # Forced mate FOR US (winning move)
    opponent_mate_in: Optional[int] = None  # Forced mate for opponent (losing move)
    leads_to_draw_repetition: bool = False  # Move leads to draw by threefold repetition


class MCTSPanel(tk.Frame):
    """
    Panel displaying MCTS search statistics.

    Shows top candidate moves with visits, win rate, and policy prior.
    Chess.com engine analysis style.
    """

    def __init__(
        self,
        parent: tk.Widget,
        width: int = 280,
        height: int = 180,
        **kwargs,
    ):
        bg = kwargs.pop("bg", COLORS["bg_secondary"])
        super().__init__(parent, bg=bg, width=width, height=height, **kwargs)

        self.configure(
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )

        # Maintain fixed height
        self.pack_propagate(False)

        self._moves: List[MCTSMoveStats] = []
        self._total_visits = 0
        self._sort_by = "visits"  # "visits", "q_value", "prior"

        self._create_widgets(bg)

    def _create_widgets(self, bg: str) -> None:
        """Create panel widgets."""
        # Title bar
        title_frame = tk.Frame(self, bg=bg)
        title_frame.pack(fill=tk.X, padx=10, pady=(8, 5))

        title = tk.Label(
            title_frame,
            text="MCTS Analysis",
            font=("Segoe UI", 11, "bold"),
            fg=COLORS["text_primary"],
            bg=bg,
        )
        title.pack(side=tk.LEFT)

        # Total visits label
        self._total_label = tk.Label(
            title_frame,
            text="",
            font=("Segoe UI", 9),
            fg=COLORS["text_muted"],
            bg=bg,
        )
        self._total_label.pack(side=tk.RIGHT)

        # Header row
        header_frame = tk.Frame(self, bg=COLORS["bg_tertiary"])
        header_frame.pack(fill=tk.X, padx=5, pady=(0, 2))

        headers = [
            ("Move", 48, "w"),
            ("Visits", 48, "e"),
            ("%", 40, "e"),
            ("Q-value", 55, "e"),
            ("WR%", 48, "e"),
            ("W", 40, "e"),
            ("D", 40, "e"),
            ("L", 40, "e"),
            ("Prior", 45, "e"),
        ]

        for text, width, anchor in headers:
            lbl = tk.Label(
                header_frame,
                text=text,
                font=("Segoe UI", 9, "bold"),
                fg=COLORS["text_secondary"],
                bg=COLORS["bg_tertiary"],
                width=width // 8,
                anchor=anchor,
            )
            lbl.pack(side=tk.LEFT, padx=2, pady=3)

        # Scrollable list frame
        list_container = tk.Frame(self, bg=bg)
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas for scrolling
        self._canvas = tk.Canvas(
            list_container,
            bg=bg,
            highlightthickness=0,
            height=150,
        )
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(
            list_container,
            orient=tk.VERTICAL,
            command=self._canvas.yview,
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._canvas.configure(yscrollcommand=scrollbar.set)

        # Inner frame for move items
        self._list_frame = tk.Frame(self._canvas, bg=bg)
        self._canvas_window = self._canvas.create_window(
            (0, 0),
            window=self._list_frame,
            anchor="nw",
        )

        # Bind resize
        self._list_frame.bind("<Configure>", self._on_frame_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        # Bind mousewheel
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)

    def _on_frame_configure(self, event: tk.Event) -> None:
        """Update scroll region when frame changes."""
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        """Update frame width when canvas resizes."""
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        """Handle mousewheel scrolling."""
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def update_stats(
        self,
        board: chess.Board,
        stats: List[Dict[str, Any]],
        total_visits: int = 0,
    ) -> None:
        """
        Update panel with MCTS statistics.

        Args:
            board: Current board position (for SAN conversion)
            stats: List of move statistics dictionaries
            total_visits: Total visits at root
        """
        self._moves = []
        self._total_visits = total_visits

        for stat in stats[:10]:  # Top 10 moves
            move = stat["move"]
            san = board.san(move) if isinstance(move, chess.Move) else str(move)

            # Get WDL or use default
            wdl = stat.get("wdl", np.array([0.33, 0.34, 0.33], dtype=np.float32))
            if not isinstance(wdl, np.ndarray):
                wdl = np.array(wdl, dtype=np.float32)

            self._moves.append(
                MCTSMoveStats(
                    move=move,
                    san=san,
                    visits=stat.get("visits", 0),
                    q_value=stat.get("q_value", 0.0),
                    prior=stat.get("prior", 0.0),
                    wdl=wdl,
                    our_mate_in=stat.get("our_mate_in"),
                    opponent_mate_in=stat.get("opponent_mate_in"),
                    leads_to_draw_repetition=stat.get("leads_to_draw_repetition", False),
                )
            )

        # Calculate total if not provided
        if total_visits == 0:
            self._total_visits = sum(m.visits for m in self._moves)

        self._update_display()

    def _update_display(self) -> None:
        """Refresh the display with current moves."""
        # Clear existing items
        for widget in self._list_frame.winfo_children():
            widget.destroy()

        # Update total label
        if self._total_visits > 0:
            self._total_label.configure(text=f"{self._total_visits:,} visits")
        else:
            self._total_label.configure(text="No data")

        if not self._moves:
            # Show "No data" message
            no_data = tk.Label(
                self._list_frame,
                text="AI hasn't moved yet",
                font=("Segoe UI", 10),
                fg=COLORS["text_muted"],
                bg=COLORS["bg_secondary"],
            )
            no_data.pack(pady=20)
            return

        # Create row for each move
        for i, move_stat in enumerate(self._moves):
            self._create_move_row(i, move_stat)

    def _create_move_row(self, index: int, stat: MCTSMoveStats) -> None:
        """Create a row for a move."""
        bg = COLORS["move_white_bg"] if index % 2 == 0 else COLORS["bg_secondary"]

        row = tk.Frame(self._list_frame, bg=bg)
        row.pack(fill=tk.X, pady=1)

        # Move SAN
        move_label = tk.Label(
            row,
            text=stat.san,
            font=("Consolas", 10, "bold"),
            fg=COLORS["text_primary"],
            bg=bg,
            width=6,
            anchor="w",
        )
        move_label.pack(side=tk.LEFT, padx=(5, 2))

        # Visits
        visits_text = self._format_visits(stat.visits)
        visits_label = tk.Label(
            row,
            text=visits_text,
            font=("Consolas", 9),
            fg=COLORS["text_secondary"],
            bg=bg,
            width=6,
            anchor="e",
        )
        visits_label.pack(side=tk.LEFT, padx=2)

        # Visit percentage
        visit_pct = (
            (stat.visits / self._total_visits * 100) if self._total_visits > 0 else 0
        )
        visit_pct_label = tk.Label(
            row,
            text=f"{visit_pct:.1f}%",
            font=("Consolas", 9),
            fg=COLORS["text_secondary"],
            bg=bg,
            width=6,
            anchor="e",
        )
        visit_pct_label.pack(side=tk.LEFT, padx=2)

        # Q-value (raw, with sign)
        q_color = self._get_win_color(stat.q_value)
        q_text = f"{stat.q_value:+.3f}"  # Format like +0.203 or -0.150
        q_label = tk.Label(
            row,
            text=q_text,
            font=("Consolas", 9),
            fg=q_color,
            bg=bg,
            width=6,
            anchor="e",
        )
        q_label.pack(side=tk.LEFT, padx=2)

        # Win Rate (from Q-value)
        wr_text = self._value_to_winrate(stat.q_value)
        wr_color = self._get_win_color(stat.q_value)
        wr_label = tk.Label(
            row,
            text=wr_text,
            font=("Consolas", 9, "bold"),
            fg=wr_color,
            bg=bg,
            width=6,
            anchor="e",
        )
        wr_label.pack(side=tk.LEFT, padx=2)

        # W, D, L as separate columns
        win_pct = stat.wdl[0] * 100
        draw_pct = stat.wdl[1] * 100
        loss_pct = stat.wdl[2] * 100

        # Win (green)
        win_label = tk.Label(
            row,
            text=f"{win_pct:.1f}",
            font=("Consolas", 9),
            fg=COLORS["q_value_positive"],
            bg=bg,
            width=5,
            anchor="e",
        )
        win_label.pack(side=tk.LEFT, padx=2)

        # Draw (gray)
        draw_label = tk.Label(
            row,
            text=f"{draw_pct:.1f}",
            font=("Consolas", 9),
            fg=COLORS["text_muted"],
            bg=bg,
            width=5,
            anchor="e",
        )
        draw_label.pack(side=tk.LEFT, padx=2)

        # Loss (red)
        loss_label = tk.Label(
            row,
            text=f"{loss_pct:.1f}",
            font=("Consolas", 9),
            fg=COLORS["q_value_negative"],
            bg=bg,
            width=5,
            anchor="e",
        )
        loss_label.pack(side=tk.LEFT, padx=2)

        # Prior (policy probability with 1 decimal)
        prior_pct = stat.prior * 100
        prior_label = tk.Label(
            row,
            text=f"{prior_pct:.1f}%",
            font=("Consolas", 9),
            fg=COLORS["prior_bar"],
            bg=bg,
            width=5,
            anchor="e",
        )
        prior_label.pack(side=tk.LEFT, padx=2)

        # Forced mate indicators
        if stat.our_mate_in is not None:
            # Winning mate for current player (green)
            mate_text = f"#{stat.our_mate_in}" if stat.our_mate_in > 1 else "#1"
            mate_label = tk.Label(
                row,
                text=mate_text,
                font=("Consolas", 9, "bold"),
                fg=COLORS["q_value_positive"],
                bg=bg,
                anchor="w",
            )
            mate_label.pack(side=tk.LEFT, padx=(5, 2))
        elif stat.opponent_mate_in is not None:
            # Losing mate - opponent has forced mate (red)
            mate_text = f"#-{stat.opponent_mate_in}" if stat.opponent_mate_in > 0 else "#"
            mate_label = tk.Label(
                row,
                text=mate_text,
                font=("Consolas", 9, "bold"),
                fg=COLORS["q_value_negative"],
                bg=bg,
                anchor="w",
            )
            mate_label.pack(side=tk.LEFT, padx=(5, 2))
        elif stat.leads_to_draw_repetition:
            # Draw by threefold repetition (orange)
            rep_label = tk.Label(
                row,
                text="=rep",
                font=("Consolas", 9, "bold"),
                fg="#FFA500",  # Orange
                bg=bg,
                anchor="w",
            )
            rep_label.pack(side=tk.LEFT, padx=(5, 2))

        # Hover effect
        def on_enter(e, r=row, b=bg):
            r.configure(bg=COLORS["move_hover"])
            for child in r.winfo_children():
                if isinstance(child, tk.Label):
                    child.configure(bg=COLORS["move_hover"])

        def on_leave(e, r=row, b=bg):
            r.configure(bg=b)
            for child in r.winfo_children():
                if isinstance(child, tk.Label):
                    child.configure(bg=b)

        row.bind("<Enter>", on_enter)
        row.bind("<Leave>", on_leave)

    def _format_visits(self, visits: int) -> str:
        """Format visit count for display."""
        if visits >= 1_000_000:
            return f"{visits / 1_000_000:.1f}M"
        elif visits >= 1_000:
            return f"{visits / 1_000:.1f}K"
        return str(visits)

    def _get_win_color(self, q_value: float) -> str:
        """Get color based on win rate."""
        if q_value > 0.1:
            return COLORS["q_value_positive"]
        elif q_value < -0.1:
            return COLORS["q_value_negative"]
        return COLORS["text_secondary"]

    def _value_to_winrate(self, value: float) -> str:
        """Convert value (-1 to +1) to win rate percentage.

        Args:
            value: Evaluation value from -1 (loss) to +1 (win)

        Returns:
            Win rate as percentage string (e.g., "65.3%")
        """
        # value = -1 -> 0%, value = 0 -> 50%, value = +1 -> 100%
        win_rate = (value + 1) / 2
        return f"{win_rate * 100:.1f}%"

    def clear(self) -> None:
        """Clear the panel."""
        self._moves = []
        self._total_visits = 0
        self._update_display()

    def set_sort(self, sort_by: str) -> None:
        """
        Set sort order for moves.

        Args:
            sort_by: "visits", "q_value", or "prior"
        """
        self._sort_by = sort_by
        if self._moves:
            if sort_by == "visits":
                self._moves.sort(key=lambda m: m.visits, reverse=True)
            elif sort_by == "q_value":
                self._moves.sort(key=lambda m: m.q_value, reverse=True)
            elif sort_by == "prior":
                self._moves.sort(key=lambda m: m.prior, reverse=True)
            self._update_display()
