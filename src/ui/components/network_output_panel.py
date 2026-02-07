"""
Network output visualization panel.

Shows raw network predictions: value and top policy moves.
"""

import tkinter as tk
from typing import List, Tuple, Optional

import chess
import numpy as np

try:
    from ..styles import COLORS, FONTS
except ImportError:
    from src.ui.styles import COLORS, FONTS


class NetworkOutputPanel(tk.Frame):
    """
    Panel showing neural network outputs.

    Displays value prediction and top policy moves.
    """

    def __init__(
        self,
        parent: tk.Widget,
        width: int = 280,
        height: int = 200,
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

        self._value = 0.0
        self._policy_moves: list[Tuple[str, float]] = []
        self._show_overlay = False

        self._create_widgets(bg)

    def _create_widgets(self, bg: str) -> None:
        """Create panel widgets."""
        # Title
        title_frame = tk.Frame(self, bg=bg)
        title_frame.pack(fill=tk.X, padx=10, pady=(8, 5))

        title = tk.Label(
            title_frame,
            text="Network Output",
            font=("Segoe UI", 11, "bold"),
            fg=COLORS["text_primary"],
            bg=bg,
        )
        title.pack(side=tk.LEFT)

        # Value section
        value_frame = tk.Frame(self, bg=bg)
        value_frame.pack(fill=tk.X, padx=10, pady=5)

        value_title = tk.Label(
            value_frame,
            text="Value Prediction:",
            font=("Segoe UI", 9),
            fg=COLORS["text_muted"],
            bg=bg,
        )
        value_title.pack(side=tk.LEFT)

        self._value_label = tk.Label(
            value_frame,
            text="+0.00",
            font=("Consolas", 12, "bold"),
            fg=COLORS["text_primary"],
            bg=bg,
        )
        self._value_label.pack(side=tk.LEFT, padx=(10, 0))

        # Value bar
        self._value_bar = tk.Canvas(
            value_frame,
            width=80,
            height=16,
            bg=COLORS["bg_tertiary"],
            highlightthickness=0,
        )
        self._value_bar.pack(side=tk.RIGHT, padx=5)

        # Separator
        sep = tk.Frame(self, bg=COLORS["border"], height=1)
        sep.pack(fill=tk.X, padx=10, pady=5)

        # Policy section
        policy_title = tk.Label(
            self,
            text="Top Policy Moves:",
            font=("Segoe UI", 9),
            fg=COLORS["text_muted"],
            bg=bg,
        )
        policy_title.pack(anchor="w", padx=10)

        # Policy list frame
        self._policy_frame = tk.Frame(self, bg=bg)
        self._policy_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create policy labels (5 moves)
        self._policy_labels: list[Tuple[tk.Label, tk.Canvas]] = []
        for _ in range(5):
            row = tk.Frame(self._policy_frame, bg=bg)
            row.pack(fill=tk.X, pady=1)

            move_label = tk.Label(
                row,
                text="---",
                font=("Consolas", 10),
                fg=COLORS["text_secondary"],
                bg=bg,
                width=6,
                anchor="w",
            )
            move_label.pack(side=tk.LEFT)

            bar_canvas = tk.Canvas(
                row,
                width=100,
                height=12,
                bg=COLORS["bg_tertiary"],
                highlightthickness=0,
            )
            bar_canvas.pack(side=tk.LEFT, padx=5)

            pct_label = tk.Label(
                row,
                text="",
                font=("Consolas", 9),
                fg=COLORS["prior_bar"],
                bg=bg,
                width=5,
                anchor="e",
            )
            pct_label.pack(side=tk.LEFT)

            self._policy_labels.append((move_label, bar_canvas, pct_label))

    def update_output(
        self,
        board: chess.Board,
        policy: np.ndarray,
        value: float,
    ) -> None:
        """
        Update panel with network output.

        Args:
            board: Current board for move decoding
            policy: Policy probability distribution
            value: Value prediction (-1 to +1)
        """
        self._value = value
        self._update_value_display()

        # Decode top policy moves
        self._decode_policy(board, policy)

    def _update_value_display(self) -> None:
        """Update the value display."""
        value = self._value

        # Format value
        if value >= 0:
            text = f"+{value:.2f}"
            color = COLORS["q_value_positive"]
        else:
            text = f"{value:.2f}"
            color = COLORS["q_value_negative"]

        self._value_label.configure(text=text, fg=color)

        # Draw value bar
        self._value_bar.delete("all")

        # Background
        self._value_bar.create_rectangle(
            0,
            0,
            80,
            16,
            fill=COLORS["bg_tertiary"],
            outline="",
        )

        # Center line
        self._value_bar.create_line(
            40,
            0,
            40,
            16,
            fill=COLORS["text_muted"],
            width=1,
        )

        # Value indicator
        # Map value from [-1, 1] to [0, 80]
        pos = int((value + 1) / 2 * 80)
        bar_color = (
            COLORS["q_value_positive"] if value >= 0 else COLORS["q_value_negative"]
        )

        if value >= 0:
            self._value_bar.create_rectangle(
                40,
                2,
                pos,
                14,
                fill=bar_color,
                outline="",
            )
        else:
            self._value_bar.create_rectangle(
                pos,
                2,
                40,
                14,
                fill=bar_color,
                outline="",
            )

    def _decode_policy(self, board: chess.Board, policy: np.ndarray) -> None:
        """Decode policy into top moves."""
        try:
            from src.alphazero.move_encoding import encode_move_from_perspective
        except ImportError:
            # Fallback: just show raw indices
            self._show_raw_policy(policy)
            return

        # Get legal moves
        legal_moves = list(board.legal_moves)

        # Policy is from current player's perspective
        # Board is flipped for black, so we need to flip moves for black
        flip = board.turn == chess.BLACK

        # Build move -> raw probability mapping
        move_probs_raw = []
        for move in legal_moves:
            try:
                # Encode move from current player's perspective
                action = encode_move_from_perspective(move, flip=flip)
                if action is not None and 0 <= action < len(policy):
                    prob = float(policy[action])
                    move_probs_raw.append((board.san(move), prob))
            except Exception:
                continue

        # Normalize probabilities to sum to 1 over legal moves only
        total_prob = sum(p for _, p in move_probs_raw)
        if total_prob > 0:
            move_probs = [(san, p / total_prob) for san, p in move_probs_raw]
        else:
            move_probs = move_probs_raw

        # Sort by probability
        move_probs.sort(key=lambda x: x[1], reverse=True)
        self._policy_moves = move_probs[:5]

        # Update display
        self._update_policy_display()

    def _show_raw_policy(self, policy: np.ndarray) -> None:
        """Show raw policy indices when decoding not available."""
        # Get top 5 indices
        top_indices = np.argsort(policy)[-5:][::-1]

        self._policy_moves = []
        for idx in top_indices:
            prob = policy[idx]
            if prob > 0.01:
                self._policy_moves.append((f"#{idx}", float(prob)))

        self._update_policy_display()

    def _update_policy_display(self) -> None:
        """Update the policy moves display."""
        max_prob = max((p for _, p in self._policy_moves), default=0.01)

        for i, (move_label, bar_canvas, pct_label) in enumerate(self._policy_labels):
            if i < len(self._policy_moves):
                move_san, prob = self._policy_moves[i]

                move_label.configure(text=move_san, fg=COLORS["text_primary"])
                pct_label.configure(text=f"{prob * 100:.1f}%")

                # Draw bar
                bar_canvas.delete("all")
                bar_width = int(100 * prob / max_prob)
                bar_canvas.create_rectangle(
                    0,
                    1,
                    bar_width,
                    11,
                    fill=COLORS["prior_bar"],
                    outline="",
                )
            else:
                move_label.configure(text="---", fg=COLORS["text_muted"])
                pct_label.configure(text="")
                bar_canvas.delete("all")

    def set_value(self, value: float) -> None:
        """Set just the value (without policy update)."""
        self._value = value
        self._update_value_display()

    def clear(self) -> None:
        """Clear the panel."""
        self._value = 0.0
        self._policy_moves = []
        self._update_value_display()

        for move_label, bar_canvas, pct_label in self._policy_labels:
            move_label.configure(text="---", fg=COLORS["text_muted"])
            pct_label.configure(text="")
            bar_canvas.delete("all")

    def get_policy_overlay_data(self) -> list[tuple[str, float]]:
        """
        Get policy data for board overlay visualization.

        Returns:
            List of (square_name, intensity) tuples
        """
        # This would need to map moves to destination squares
        # for board overlay visualization
        overlay_data = []

        for move_san, prob in self._policy_moves:
            # Extract destination square from SAN (simplified)
            # This is a basic implementation
            if len(move_san) >= 2:
                dest = move_san[-2:] if move_san[-1].isdigit() else move_san[-3:-1]
                if len(dest) == 2 and dest[0] in "abcdefgh" and dest[1] in "12345678":
                    overlay_data.append((dest, prob))

        return overlay_data
