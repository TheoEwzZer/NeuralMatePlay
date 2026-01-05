"""
Evaluation bar widget.

Vertical bar showing position evaluation like chess.com.
"""

import tkinter as tk
from typing import Optional

try:
    from ..styles import COLORS, FONTS
except ImportError:
    from src.ui.styles import COLORS, FONTS


class EvalBar(tk.Canvas):
    """
    Vertical evaluation bar showing who is winning.

    Chess.com-style design with white on top, black on bottom.
    The bar fills from the center based on evaluation.
    """

    def __init__(
        self,
        parent: tk.Widget,
        height: int = 720,
        width: int = 28,
        **kwargs,
    ):
        bg = kwargs.pop("bg", COLORS["bg_secondary"])
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=bg,
            highlightthickness=0,
            **kwargs,
        )

        self._width = width
        self._height = height
        self._evaluation = 0.0  # Range: -10 to +10
        self._animated_eval = 0.0
        self._animation_id: Optional[str] = None

        # Draw initial state
        self._draw_bar()

    def _draw_bar(self) -> None:
        """Draw the evaluation bar."""
        self.delete("all")

        w = self._width
        h = self._height
        padding = 2
        bar_width = w - 2 * padding
        bar_height = h - 2 * padding

        # Background (border effect)
        self.create_rectangle(
            0, 0, w, h,
            fill=COLORS["border"],
            outline="",
        )

        # Calculate fill based on evaluation
        # eval = 0 -> 50% white, 50% black
        # eval = +10 -> 100% white
        # eval = -10 -> 100% black
        clamped_eval = max(-10, min(10, self._animated_eval))
        white_ratio = (clamped_eval + 10) / 20  # 0 to 1

        # White section (top)
        white_height = int(bar_height * white_ratio)
        if white_height > 0:
            self.create_rectangle(
                padding, padding,
                padding + bar_width, padding + white_height,
                fill=COLORS["eval_white"],
                outline="",
            )

        # Black section (bottom)
        black_height = bar_height - white_height
        if black_height > 0:
            self.create_rectangle(
                padding, padding + white_height,
                padding + bar_width, padding + bar_height,
                fill=COLORS["eval_black"],
                outline="",
            )

        # Center line (equal position marker)
        center_y = h // 2
        self.create_line(
            0, center_y, w, center_y,
            fill=COLORS["eval_equal"],
            width=1,
            dash=(2, 2),
        )

        # Draw evaluation text
        self._draw_eval_text()

    def _draw_eval_text(self) -> None:
        """Draw the evaluation value text."""
        eval_val = self._animated_eval
        w = self._width
        h = self._height

        # Format evaluation
        if abs(eval_val) >= 10:
            text = "M" if eval_val > 0 else "-M"  # Mate
        else:
            text = f"{eval_val:+.1f}" if eval_val != 0 else "0.0"

        # Position based on who's winning
        if eval_val >= 0:
            # Show at top (white section)
            y = 15
            fill = COLORS["eval_black"]
            bg_color = COLORS["eval_white"]
        else:
            # Show at bottom (black section)
            y = h - 15
            fill = COLORS["eval_white"]
            bg_color = COLORS["eval_black"]

        # Text background for readability
        self.create_rectangle(
            2, y - 10,
            w - 2, y + 10,
            fill=bg_color,
            outline="",
            tags="eval_text",
        )

        self.create_text(
            w // 2, y,
            text=text,
            font=("Segoe UI", 9, "bold"),
            fill=fill,
            tags="eval_text",
        )

    def set_evaluation(self, value: float, animate: bool = True) -> None:
        """
        Set the evaluation value.

        Args:
            value: Evaluation from -10 to +10 (positive = white winning)
            animate: Whether to animate the transition
        """
        self._evaluation = max(-10, min(10, value))

        if animate:
            self._animate_to(self._evaluation)
        else:
            self._animated_eval = self._evaluation
            self._draw_bar()

    def _animate_to(self, target: float, steps: int = 10) -> None:
        """Animate the bar to target evaluation."""
        if self._animation_id:
            self.after_cancel(self._animation_id)

        diff = target - self._animated_eval
        if abs(diff) < 0.01:
            self._animated_eval = target
            self._draw_bar()
            return

        step_size = diff / steps

        def animate_step():
            self._animated_eval += step_size
            if (step_size > 0 and self._animated_eval >= target) or \
               (step_size < 0 and self._animated_eval <= target):
                self._animated_eval = target
                self._draw_bar()
            else:
                self._draw_bar()
                self._animation_id = self.after(20, animate_step)

        animate_step()

    def get_evaluation(self) -> float:
        """Get the current evaluation value."""
        return self._evaluation

    def clear(self) -> None:
        """Reset to equal position."""
        self.set_evaluation(0.0, animate=False)


class EvalBarHorizontal(tk.Canvas):
    """
    Horizontal evaluation bar (alternative layout).

    White on right, black on left.
    """

    def __init__(
        self,
        parent: tk.Widget,
        width: int = 300,
        height: int = 20,
        **kwargs,
    ):
        bg = kwargs.pop("bg", COLORS["bg_secondary"])
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=bg,
            highlightthickness=0,
            **kwargs,
        )

        self._width = width
        self._height = height
        self._evaluation = 0.0

        self._draw_bar()

    def _draw_bar(self) -> None:
        """Draw the horizontal evaluation bar."""
        self.delete("all")

        w = self._width
        h = self._height
        padding = 2

        # Background
        self.create_rectangle(
            0, 0, w, h,
            fill=COLORS["border"],
            outline="",
        )

        # Calculate fill
        clamped_eval = max(-10, min(10, self._evaluation))
        white_ratio = (clamped_eval + 10) / 20

        # Black section (left)
        black_width = int((w - 2 * padding) * (1 - white_ratio))
        if black_width > 0:
            self.create_rectangle(
                padding, padding,
                padding + black_width, h - padding,
                fill=COLORS["eval_black"],
                outline="",
            )

        # White section (right)
        white_width = (w - 2 * padding) - black_width
        if white_width > 0:
            self.create_rectangle(
                padding + black_width, padding,
                w - padding, h - padding,
                fill=COLORS["eval_white"],
                outline="",
            )

        # Center marker
        center_x = w // 2
        self.create_line(
            center_x, 0, center_x, h,
            fill=COLORS["eval_equal"],
            width=1,
        )

        # Evaluation text
        if abs(self._evaluation) >= 10:
            text = "M" if self._evaluation > 0 else "-M"
        else:
            text = f"{self._evaluation:+.1f}" if self._evaluation != 0 else "="

        self.create_text(
            w // 2, h // 2,
            text=text,
            font=("Segoe UI", 8, "bold"),
            fill=COLORS["text_primary"],
        )

    def set_evaluation(self, value: float) -> None:
        """Set evaluation value (-10 to +10)."""
        self._evaluation = max(-10, min(10, value))
        self._draw_bar()
