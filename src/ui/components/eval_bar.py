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

    The White part grows from the White side of the board (bottom by
    default, top when the board is flipped). The label shows the leading
    side's winning chances as a percentage, which is easier to read than
    a raw evaluation number.
    """

    def __init__(
        self,
        parent: tk.Widget,
        height: int = 720,
        width: int = 35,
        **kwargs,
    ):
        bg = kwargs.pop("bg", COLORS["bg_primary"])
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
        self._evaluation = 0.0  # Range: -10 to +10 (positive = White better)
        self._animated_eval = 0.0
        self._animation_id: Optional[str] = None
        self._flipped = False  # False = White at the bottom
        self._has_data = False

        # Draw initial state
        self._draw_bar()

    def set_flipped(self, flipped: bool) -> None:
        """Put White at the top (flipped) or at the bottom (default)."""
        self._flipped = flipped
        self._draw_bar()

    def _draw_bar(self) -> None:
        """Draw the evaluation bar."""
        self.delete("all")

        w = self._width
        h = self._height

        # Black background, White part drawn on top of it
        self.create_rectangle(0, 0, w, h, fill=COLORS["eval_black"], outline="")

        clamped_eval = max(-10, min(10, self._animated_eval))
        white_ratio = (clamped_eval + 10) / 20  # 0 to 1
        white_height = int(h * white_ratio)

        if white_height > 0:
            if self._flipped:
                self.create_rectangle(
                    0, 0, w, white_height, fill=COLORS["eval_white"], outline=""
                )
            else:
                self.create_rectangle(
                    0, h - white_height, w, h, fill=COLORS["eval_white"], outline=""
                )

        # Center line (equal position marker)
        center_y = h // 2
        self.create_line(0, center_y, w, center_y, fill=COLORS["eval_equal"], width=1)

        # Draw evaluation text
        self._draw_eval_text()

    def _draw_eval_text(self) -> None:
        """Draw the leading side's winning chances."""
        if not self._has_data:
            return

        eval_val = self._animated_eval
        w = self._width
        h = self._height

        if abs(eval_val) >= 10:
            text = "MAT"
        else:
            white_score = (eval_val / 10 + 1) / 2
            text = f"{round(max(white_score, 1 - white_score) * 100)}%"

        white_leads = eval_val >= 0
        white_at_bottom = not self._flipped
        at_bottom = white_leads == white_at_bottom
        y = h - 12 if at_bottom else 12
        fill = COLORS["eval_black"] if white_leads else COLORS["eval_white"]

        self.create_text(
            w // 2,
            y,
            text=text,
            font=("Segoe UI Semibold", 8),
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
        self._has_data = True

        if animate:
            self._animate_to(self._evaluation)
        else:
            self._animated_eval = self._evaluation
            self._draw_bar()

    def _animate_to(self, target: float, steps: int = 12) -> None:
        """Animate the bar to target evaluation."""
        if self._animation_id:
            self.after_cancel(self._animation_id)
            self._animation_id = None

        diff = target - self._animated_eval
        if abs(diff) < 0.01:
            self._animated_eval = target
            self._draw_bar()
            return

        step_size = diff / steps

        def animate_step():
            self._animated_eval += step_size
            if (step_size > 0 and self._animated_eval >= target) or (
                step_size < 0 and self._animated_eval <= target
            ):
                self._animated_eval = target
                self._animation_id = None
                self._draw_bar()
            else:
                self._draw_bar()
                self._animation_id = self.after(16, animate_step)

        animate_step()

    def get_evaluation(self) -> float:
        """Get the current evaluation value."""
        return self._evaluation

    def clear(self) -> None:
        """Reset to equal position."""
        if self._animation_id:
            self.after_cancel(self._animation_id)
            self._animation_id = None
        self._evaluation = 0.0
        self._animated_eval = 0.0
        self._has_data = False
        self._draw_bar()


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
            0,
            0,
            w,
            h,
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
                padding,
                padding,
                padding + black_width,
                h - padding,
                fill=COLORS["eval_black"],
                outline="",
            )

        # White section (right)
        white_width = (w - 2 * padding) - black_width
        if white_width > 0:
            self.create_rectangle(
                padding + black_width,
                padding,
                w - padding,
                h - padding,
                fill=COLORS["eval_white"],
                outline="",
            )

        # Center marker
        center_x = w // 2
        self.create_line(
            center_x,
            0,
            center_x,
            h,
            fill=COLORS["eval_equal"],
            width=1,
        )

        # Evaluation text
        if abs(self._evaluation) >= 10:
            text = "M" if self._evaluation > 0 else "-M"
        else:
            text = f"{self._evaluation:+.1f}" if self._evaluation != 0 else "="

        self.create_text(
            w // 2,
            h // 2,
            text=text,
            font=("Segoe UI", 8, "bold"),
            fill=COLORS["text_primary"],
        )

    def set_evaluation(self, value: float) -> None:
        """Set evaluation value (-10 to +10)."""
        self._evaluation = max(-10, min(10, value))
        self._draw_bar()
