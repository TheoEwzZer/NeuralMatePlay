"""
Move list widget.

Scrollable, clickable list of moves in standard notation.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Optional, Callable, Tuple

import chess

try:
    from ..styles import COLORS, FONTS
except ImportError:
    from src.ui.styles import COLORS, FONTS


class MoveList(tk.Frame):
    """
    Scrollable move list with navigation.

    Displays moves in two-column format (1. e4 e5).
    Click on any move to jump to that position.
    """

    def __init__(
        self,
        parent: tk.Widget,
        width: int = 250,
        height: int = 200,
        **kwargs,
    ):
        bg = kwargs.pop("bg", COLORS["bg_secondary"])
        super().__init__(parent, bg=bg, width=width, height=height, **kwargs)

        self.configure(
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )

        self._moves: List[chess.Move] = []
        self._sans: List[str] = []  # SAN notation strings
        self._current_index = -1  # Current position in move list

        self._create_widgets(bg)

    def _create_widgets(self, bg: str) -> None:
        """Create widget components."""
        # Title bar
        title_frame = tk.Frame(self, bg=bg)
        title_frame.pack(fill=tk.X, padx=10, pady=(8, 5))

        title = tk.Label(
            title_frame,
            text="Moves",
            font=("Segoe UI", 11, "bold"),
            fg=COLORS["text_primary"],
            bg=bg,
        )
        title.pack(side=tk.LEFT)

        # Move count
        self._count_label = tk.Label(
            title_frame,
            text="",
            font=("Segoe UI", 9),
            fg=COLORS["text_muted"],
            bg=bg,
        )
        self._count_label.pack(side=tk.RIGHT)

        # Scrollable content
        container = tk.Frame(self, bg=bg)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas for scrolling
        self._canvas = tk.Canvas(
            container,
            bg=bg,
            highlightthickness=0,
        )
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(
            container,
            orient=tk.VERTICAL,
            command=self._canvas.yview,
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._canvas.configure(yscrollcommand=scrollbar.set)

        # Inner frame for move rows
        self._list_frame = tk.Frame(self._canvas, bg=bg)
        self._canvas_window = self._canvas.create_window(
            (0, 0),
            window=self._list_frame,
            anchor="nw",
        )

        # Bind events
        self._list_frame.bind("<Configure>", self._on_frame_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)

        # Move item references for highlighting
        self._move_labels: List[tk.Label] = []

    def _on_frame_configure(self, event: tk.Event) -> None:
        """Update scroll region."""
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        """Update frame width."""
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        """Handle mousewheel."""
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def set_moves(self, board: chess.Board) -> None:
        """
        Update move list from board history.

        Args:
            board: Board with move history
        """
        # Extract moves from board
        temp_board = chess.Board()
        self._moves = []
        self._sans = []

        try:
            for move in board.move_stack:
                self._moves.append(move)
                san = temp_board.san(move)
                self._sans.append(san)
                temp_board.push(move)
        except (AssertionError, ValueError):
            pass

        self._current_index = len(self._moves) - 1
        self._rebuild_display()

    def set_moves_from_list(
        self,
        moves: List[chess.Move],
        sans: List[str],
        current_index: int = -1,
    ) -> None:
        """
        Set moves directly from lists.

        Args:
            moves: List of moves
            sans: List of SAN notations
            current_index: Current position index
        """
        self._moves = list(moves)
        self._sans = list(sans)
        self._current_index = current_index if current_index >= 0 else len(moves) - 1
        self._rebuild_display()

    def _rebuild_display(self) -> None:
        """Rebuild the entire move list display."""
        # Clear existing
        for widget in self._list_frame.winfo_children():
            widget.destroy()
        self._move_labels = []

        bg = COLORS["bg_secondary"]

        # Update count
        num_moves = len(self._moves)
        full_moves = (num_moves + 1) // 2
        self._count_label.configure(
            text=f"{full_moves} moves" if full_moves != 1 else "1 move"
        )

        if not self._moves:
            # Empty state
            empty_label = tk.Label(
                self._list_frame,
                text="Game start",
                font=("Segoe UI", 10),
                fg=COLORS["text_muted"],
                bg=bg,
            )
            empty_label.pack(pady=20)
            return

        # Build rows (each row = 1 full move: white + black)
        # Use min of both lists to avoid index errors
        num_moves = min(len(self._moves), len(self._sans))
        for i in range(0, num_moves, 2):
            move_num = i // 2 + 1
            self._create_move_row(i, move_num, bg)

        # Scroll to current position
        self._scroll_to_current()

    def _create_move_row(self, index: int, move_num: int, bg: str) -> None:
        """Create a row with move number and 1-2 moves."""
        row_bg = COLORS["move_white_bg"] if move_num % 2 == 1 else bg

        row = tk.Frame(self._list_frame, bg=row_bg)
        row.pack(fill=tk.X, pady=1)

        # Move number
        num_label = tk.Label(
            row,
            text=f"{move_num}.",
            font=("Consolas", 10),
            fg=COLORS["text_muted"],
            bg=row_bg,
            width=4,
            anchor="e",
        )
        num_label.pack(side=tk.LEFT, padx=(5, 2))

        # White's move
        white_san = self._sans[index]
        white_is_current = index == self._current_index

        white_label = self._create_move_label(
            row, white_san, index, white_is_current, row_bg
        )
        white_label.pack(side=tk.LEFT, padx=2)
        self._move_labels.append(white_label)

        # Black's move (if exists)
        if index + 1 < len(self._sans):
            black_san = self._sans[index + 1]
            black_is_current = index + 1 == self._current_index

            black_label = self._create_move_label(
                row, black_san, index + 1, black_is_current, row_bg
            )
            black_label.pack(side=tk.LEFT, padx=2)
            self._move_labels.append(black_label)

    def _create_move_label(
        self,
        parent: tk.Widget,
        san: str,
        index: int,
        is_current: bool,
        bg: str,
    ) -> tk.Label:
        """Create a clickable move label."""
        if is_current:
            label_bg = COLORS["move_current"]
            label_fg = COLORS["bg_primary"]
        else:
            label_bg = bg
            label_fg = COLORS["text_primary"]

        label = tk.Label(
            parent,
            text=san,
            font=("Consolas", 10, "bold") if is_current else ("Consolas", 10),
            fg=label_fg,
            bg=label_bg,
            width=7,
            anchor="w",
            cursor="hand2",
            padx=3,
            pady=1,
        )

        # Store index for click handling
        label._move_index = index

        # Hover effect (if not current)
        if not is_current:
            label.bind(
                "<Enter>", lambda e, l=label: l.configure(bg=COLORS["move_hover"])
            )
            label.bind("<Leave>", lambda e, l=label, b=bg: l.configure(bg=b))

        return label

    def set_current_move(self, index: int) -> None:
        """
        Set the current position marker.

        Args:
            index: 0-based index in move list (-1 for start position)
        """
        self._current_index = index
        self._update_highlighting()
        self._scroll_to_current()

    def _update_highlighting(self) -> None:
        """Update move highlighting without full rebuild."""
        for label in self._move_labels:
            idx = getattr(label, "_move_index", -1)
            if idx == self._current_index:
                label.configure(
                    bg=COLORS["move_current"],
                    fg=COLORS["bg_primary"],
                    font=("Consolas", 10, "bold"),
                )
                # Remove hover bindings
                label.unbind("<Enter>")
                label.unbind("<Leave>")
            else:
                row_num = idx // 2 + 1
                bg = (
                    COLORS["move_white_bg"]
                    if row_num % 2 == 1
                    else COLORS["bg_secondary"]
                )
                label.configure(
                    bg=bg,
                    fg=COLORS["text_primary"],
                    font=("Consolas", 10),
                )
                # Add hover bindings back
                label.bind(
                    "<Enter>", lambda e, l=label: l.configure(bg=COLORS["move_hover"])
                )
                label.bind("<Leave>", lambda e, l=label, b=bg: l.configure(bg=b))

    def _scroll_to_current(self) -> None:
        """Scroll to make current move visible."""
        if self._current_index < 0:
            return

        self._canvas.update_idletasks()

        # Find the label for current move
        for label in self._move_labels:
            if getattr(label, "_move_index", -1) == self._current_index:
                # Get label position relative to canvas
                label_y = label.winfo_y()
                label_height = label.winfo_height()
                canvas_height = self._canvas.winfo_height()

                # Calculate scroll position
                total_height = self._list_frame.winfo_height()
                if total_height <= canvas_height:
                    return  # No scrolling needed

                # Scroll to center the label
                target_y = label_y - canvas_height // 2 + label_height // 2
                scroll_fraction = max(0, min(1, target_y / total_height))
                self._canvas.yview_moveto(scroll_fraction)
                break

    def add_move(self, move: chess.Move, san: str) -> None:
        """
        Add a single move to the list.

        Args:
            move: The move
            san: SAN notation
        """
        self._moves.append(move)
        self._sans.append(san)
        self._current_index = len(self._moves) - 1
        self._rebuild_display()

    def remove_last_move(self) -> Optional[chess.Move]:
        """
        Remove the last move from the list.

        Returns:
            The removed move, or None if list was empty
        """
        if self._moves:
            move = self._moves.pop()
            self._sans.pop()
            self._current_index = len(self._moves) - 1
            self._rebuild_display()
            return move
        return None

    def clear(self) -> None:
        """Clear all moves."""
        self._moves = []
        self._sans = []
        self._current_index = -1
        self._rebuild_display()

    def get_moves(self) -> List[chess.Move]:
        """Get the list of moves."""
        return list(self._moves)

    def get_current_index(self) -> int:
        """Get current position index."""
        return self._current_index
