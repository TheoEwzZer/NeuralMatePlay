"""
Evaluation graph widget.

Line chart showing evaluation history over the game.
"""

import tkinter as tk
from typing import List, Optional, Tuple

import numpy as np

try:
    from ..styles import COLORS, FONTS
except ImportError:
    from src.ui.styles import COLORS, FONTS


class EvalGraph(tk.Canvas):
    """
    Line graph showing evaluation over time.

    X-axis: Move number
    Y-axis: Evaluation (-1 to +1)
    Features vertical marker for current position.
    """

    def __init__(
        self,
        parent: tk.Widget,
        width: int = 280,
        height: int = 140,
        **kwargs,
    ):
        bg = kwargs.pop("bg", COLORS["graph_bg"])
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
        self._eval_history: List[float] = []
        self._wdl_history: List[Optional[np.ndarray]] = []  # WDL for each move
        self._current_move = 0
        self._max_eval = 1.0  # Fixed scaling -1 to +1
        self._min_eval = -1.0

        # Margins
        self._margin_left = 30
        self._margin_right = 10
        self._margin_top = 15
        self._margin_bottom = 25  # Extra space for win rate display

        # Bind mouse events for tooltips
        self.bind("<Motion>", self._on_mouse_move)
        self.bind("<Leave>", self._on_mouse_leave)

        self._tooltip_id: Optional[int] = None

        self._draw_graph()

    def _draw_graph(self) -> None:
        """Draw the complete graph."""
        self.delete("all")

        # Draw background
        self._draw_background()

        # Draw grid
        self._draw_grid()

        # Draw axis labels
        self._draw_labels()

        # Draw evaluation line
        if len(self._eval_history) > 0:
            self._draw_eval_line()

        # Draw current position marker
        if 0 < self._current_move <= len(self._eval_history):
            self._draw_current_marker()

    def _draw_background(self) -> None:
        """Draw graph background with sections."""
        plot_left = self._margin_left
        plot_right = self._width - self._margin_right
        plot_top = self._margin_top
        plot_bottom = self._height - self._margin_bottom
        plot_height = plot_bottom - plot_top
        center_y = plot_top + plot_height // 2

        # White advantage section (top half, light)
        self.create_rectangle(
            plot_left,
            plot_top,
            plot_right,
            center_y,
            fill="#252535",
            outline="",
        )

        # Black advantage section (bottom half, dark)
        self.create_rectangle(
            plot_left,
            center_y,
            plot_right,
            plot_bottom,
            fill="#1a1a28",
            outline="",
        )

    def _draw_grid(self) -> None:
        """Draw grid lines."""
        plot_left = self._margin_left
        plot_right = self._width - self._margin_right
        plot_top = self._margin_top
        plot_bottom = self._height - self._margin_bottom

        # Horizontal grid lines (evaluation levels)
        for eval_val in [-1, -0.5, 0, 0.5, 1]:
            y = self._eval_to_y(eval_val)
            color = COLORS["graph_zero"] if eval_val == 0 else COLORS["graph_grid"]
            line_width = 2 if eval_val == 0 else 1

            self.create_line(
                plot_left,
                y,
                plot_right,
                y,
                fill=color,
                width=line_width,
                dash=(2, 2) if eval_val != 0 else None,
            )

        # Border
        self.create_rectangle(
            plot_left,
            plot_top,
            plot_right,
            plot_bottom,
            outline=COLORS["border"],
            width=1,
        )

    def _draw_labels(self) -> None:
        """Draw axis labels."""
        plot_left = self._margin_left
        plot_top = self._margin_top
        plot_bottom = self._height - self._margin_bottom

        # Y-axis labels
        for eval_val in [-1, -0.5, 0, 0.5, 1]:
            y = self._eval_to_y(eval_val)
            if eval_val == 0:
                text = "0"
            elif eval_val == int(eval_val):
                text = f"{int(eval_val):+d}"
            else:
                text = f"{eval_val:+.1f}"
            self.create_text(
                plot_left - 5,
                y,
                text=text,
                font=("Segoe UI", 8),
                fill=COLORS["text_muted"],
                anchor="e",
            )

        # Title
        self.create_text(
            self._width // 2,
            8,
            text="Evaluation",
            font=("Segoe UI", 9, "bold"),
            fill=COLORS["text_secondary"],
        )

        # X-axis label (move number) and Win Rate
        if len(self._eval_history) > 0:
            # Move number on the right
            self.create_text(
                self._width - 5,
                self._height - 5,
                text=f"Move {len(self._eval_history)}",
                font=("Segoe UI", 8),
                fill=COLORS["text_muted"],
                anchor="se",
            )

            # Win Rate on the left (current position)
            if 0 < self._current_move <= len(self._wdl_history):
                wdl = self._wdl_history[self._current_move - 1]
                if wdl is not None:
                    wr_text = self._readable_wdl(wdl)
                    expectation = wdl[0] + 0.5 * wdl[1]
                    wr_color = self._get_wr_color(expectation)
                    self.create_text(
                        plot_left + 5,
                        self._height - 5,
                        text=f"WR: {wr_text}",
                        font=("Segoe UI", 9, "bold"),
                        fill=wr_color,
                        anchor="sw",
                    )

    def _draw_eval_line(self) -> None:
        """Draw the evaluation line."""
        if len(self._eval_history) < 1:
            return

        plot_left = self._margin_left
        plot_right = self._width - self._margin_right
        plot_width = plot_right - plot_left

        points = []
        num_moves = len(self._eval_history)

        for i, eval_val in enumerate(self._eval_history):
            x = (
                plot_left + (i / max(1, num_moves - 1)) * plot_width
                if num_moves > 1
                else plot_left + plot_width // 2
            )
            y = self._eval_to_y(eval_val)
            points.append((x, y))

        # Draw filled area under the line
        if len(points) >= 2:
            # Create polygon for fill
            center_y = self._eval_to_y(0)
            fill_points = []

            # Start from zero line
            fill_points.append((points[0][0], center_y))

            # Add all points
            fill_points.extend(points)

            # End at zero line
            fill_points.append((points[-1][0], center_y))

            # Split into positive and negative regions
            self._draw_filled_regions(points, center_y)

        # Draw the line itself
        if len(points) >= 2:
            flat_points = [coord for point in points for coord in point]
            self.create_line(
                *flat_points,
                fill=COLORS["graph_line"],
                width=2,
                smooth=True,
                tags="eval_line",
            )

        # Draw points
        for x, y in points:
            self.create_oval(
                x - 2,
                y - 2,
                x + 2,
                y + 2,
                fill=COLORS["graph_line"],
                outline="",
                tags="eval_point",
            )

    def _draw_filled_regions(
        self, points: List[Tuple[float, float]], center_y: float
    ) -> None:
        """Draw filled regions above/below zero line."""
        if len(points) < 2:
            return

        # Create segments for positive and negative regions
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            # Both above zero (white winning)
            if y1 <= center_y and y2 <= center_y:
                poly = [x1, center_y, x1, y1, x2, y2, x2, center_y]
                self.create_polygon(
                    poly,
                    fill="#3b5998",
                    outline="",
                    stipple="gray25",
                )
            # Both below zero (black winning)
            elif y1 >= center_y and y2 >= center_y:
                poly = [x1, center_y, x1, y1, x2, y2, x2, center_y]
                self.create_polygon(
                    poly,
                    fill="#8b3a3a",
                    outline="",
                    stipple="gray25",
                )

    def _draw_current_marker(self) -> None:
        """Draw marker for current position."""
        if self._current_move <= 0 or self._current_move > len(self._eval_history):
            return

        plot_left = self._margin_left
        plot_right = self._width - self._margin_right
        plot_top = self._margin_top
        plot_bottom = self._height - self._margin_bottom
        plot_width = plot_right - plot_left
        num_moves = len(self._eval_history)

        idx = self._current_move - 1
        x = (
            plot_left + (idx / max(1, num_moves - 1)) * plot_width
            if num_moves > 1
            else plot_left + plot_width // 2
        )

        # Vertical line
        self.create_line(
            x,
            plot_top,
            x,
            plot_bottom,
            fill=COLORS["graph_marker"],
            width=2,
            tags="marker",
        )

        # Highlighted point
        y = self._eval_to_y(self._eval_history[idx])
        self.create_oval(
            x - 5,
            y - 5,
            x + 5,
            y + 5,
            fill=COLORS["graph_marker"],
            outline=COLORS["text_primary"],
            width=1,
            tags="marker",
        )

        # Value label
        eval_val = self._eval_history[idx]
        text = f"{eval_val:+.1f}"
        self.create_text(
            x,
            plot_top - 5,
            text=text,
            font=("Segoe UI", 8, "bold"),
            fill=COLORS["graph_marker"],
            anchor="s",
            tags="marker",
        )

    def _eval_to_y(self, eval_val: float) -> float:
        """Convert evaluation to Y coordinate."""
        plot_top = self._margin_top
        plot_bottom = self._height - self._margin_bottom
        plot_height = plot_bottom - plot_top

        # Clamp to display range
        clamped = max(self._min_eval, min(self._max_eval, eval_val))

        # Map -max to bottom, +max to top
        normalized = (clamped - self._min_eval) / (self._max_eval - self._min_eval)
        return plot_bottom - normalized * plot_height

    def _readable_wdl(self, wdl: np.ndarray) -> str:
        """Convert WDL array to expectation percentage.

        Args:
            wdl: Array [P(win), P(draw), P(loss)]

        Returns:
            Expectation as percentage string (e.g., "65.3%")
        """
        expectation = wdl[0] + 0.5 * wdl[1]
        return f"{expectation * 100:.1f}%"

    def _get_wr_color(self, expectation: float) -> str:
        """Get color based on win rate expectation (0-1 scale)."""
        if expectation > 0.55:
            return COLORS["q_value_positive"]
        elif expectation < 0.45:
            return COLORS["q_value_negative"]
        return COLORS["text_secondary"]

    def _on_mouse_move(self, event: tk.Event) -> None:
        """Handle mouse movement for tooltips."""
        # Find closest data point
        if len(self._eval_history) == 0:
            return

        plot_left = self._margin_left
        plot_right = self._width - self._margin_right

        if event.x < plot_left or event.x > plot_right:
            self._hide_tooltip()
            return

        # Calculate which move
        plot_width = plot_right - plot_left
        num_moves = len(self._eval_history)

        if num_moves == 1:
            idx = 0
        else:
            ratio = (event.x - plot_left) / plot_width
            idx = int(ratio * (num_moves - 1) + 0.5)
            idx = max(0, min(num_moves - 1, idx))

        # Show tooltip with eval and WDL if available
        eval_val = self._eval_history[idx]
        move_num = idx + 1
        tooltip_text = f"Move {move_num}: {eval_val:+.2f}"

        # Add win rate if WDL is available
        if idx < len(self._wdl_history) and self._wdl_history[idx] is not None:
            wdl = self._wdl_history[idx]
            wr_text = self._readable_wdl(wdl)
            tooltip_text += f" (WR: {wr_text})"

        self._show_tooltip(event.x, event.y, tooltip_text)

    def _on_mouse_leave(self, event: tk.Event) -> None:
        """Hide tooltip when mouse leaves."""
        self._hide_tooltip()

    def _show_tooltip(self, x: int, y: int, text: str) -> None:
        """Show tooltip at position."""
        self._hide_tooltip()

        # Create tooltip
        self._tooltip_id = self.create_text(
            x,
            y - 15,
            text=text,
            font=("Segoe UI", 8),
            fill=COLORS["text_primary"],
            tags="tooltip",
        )

        # Background
        bbox = self.bbox(self._tooltip_id)
        if bbox:
            self.create_rectangle(
                bbox[0] - 3,
                bbox[1] - 2,
                bbox[2] + 3,
                bbox[3] + 2,
                fill=COLORS["bg_tertiary"],
                outline=COLORS["border"],
                tags="tooltip_bg",
            )
            self.tag_raise(self._tooltip_id)

    def _hide_tooltip(self) -> None:
        """Hide the tooltip."""
        self.delete("tooltip")
        self.delete("tooltip_bg")
        self._tooltip_id = None

    def add_evaluation(
        self, value: float, wdl: Optional[np.ndarray] = None
    ) -> None:
        """
        Add a new evaluation point.

        Args:
            value: Evaluation value (-1 to +1)
            wdl: Optional WDL array [P(win), P(draw), P(loss)]
        """
        self._eval_history.append(value)
        self._wdl_history.append(wdl)
        self._current_move = len(self._eval_history)
        self._draw_graph()

    def set_history(
        self,
        history: List[float],
        wdl_history: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        """
        Set the complete evaluation history.

        Args:
            history: List of evaluation values
            wdl_history: Optional list of WDL arrays
        """
        self._eval_history = list(history)
        if wdl_history is not None:
            self._wdl_history = list(wdl_history)
        else:
            self._wdl_history = [None] * len(history)
        self._current_move = len(self._eval_history)
        self._draw_graph()

    def set_current_move(self, move_index: int) -> None:
        """
        Set the current position marker.

        Args:
            move_index: 1-based move index
        """
        self._current_move = move_index
        self._draw_graph()

    def clear(self) -> None:
        """Reset the graph for a new game."""
        self._eval_history = []
        self._wdl_history = []
        self._current_move = 0
        self._max_eval = 1.0
        self._min_eval = -1.0
        self._draw_graph()

    def get_history(self) -> List[float]:
        """Get the evaluation history."""
        return list(self._eval_history)
