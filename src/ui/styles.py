"""
Modern dark theme styling for the chess UI.

Defines colors, fonts, and styling utilities for consistent
appearance across the application.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable


# Color palette - Modern dark theme
COLORS = {
    # Background colors
    "bg_primary": "#1a1a2e",  # Main background
    "bg_secondary": "#16213e",  # Panel background
    "bg_tertiary": "#0f3460",  # Elevated elements
    # Chess board colors
    "light_square": "#eeeed2",  # Light squares (cream)
    "dark_square": "#769656",  # Dark squares (green)
    # Highlight colors
    "selected": "#baca44",  # Selected square
    "legal_move": "#646d40",  # Legal move indicator
    "last_move": "#f6f669",  # Last move highlight
    "check": "#e94560",  # King in check
    # Accent colors
    "accent": "#e94560",  # Primary accent (red-pink)
    "accent_secondary": "#0ead69",  # Secondary accent (green)
    "accent_tertiary": "#4cc9f0",  # Tertiary accent (blue)
    # Text colors
    "text_primary": "#ffffff",  # Primary text
    "text_secondary": "#a0a0a0",  # Secondary text
    "text_muted": "#666666",  # Muted text
    # Button colors
    "button_bg": "#0f3460",
    "button_hover": "#1a4a7a",
    "button_active": "#e94560",
    "button_disabled": "#2a2a3e",
    # Status colors
    "success": "#0ead69",
    "warning": "#ffc107",
    "error": "#e94560",
    "info": "#4cc9f0",
    # Border colors
    "border": "#2a2a4e",
    "border_light": "#3a3a5e",
    # Progress bar
    "progress_bg": "#2a2a4e",
    "progress_fill": "#e94560",
    # Evaluation bar (chess.com style)
    "eval_white": "#ffffff",
    "eval_black": "#1a1a1a",
    "eval_winning": "#4ade80",
    "eval_losing": "#f87171",
    "eval_equal": "#888888",
    # Player info
    "player_active": "#3b82f6",
    "player_active_glow": "#60a5fa",
    "captured_bg": "#262640",
    # Evaluation graph
    "graph_bg": "#1e1e2e",
    "graph_line": "#60a5fa",
    "graph_line_alt": "#f472b6",
    "graph_grid": "#333355",
    "graph_zero": "#666688",
    "graph_marker": "#fbbf24",
    # MCTS panel
    "visit_bar": "#8b5cf6",
    "prior_bar": "#06b6d4",
    "q_value_positive": "#4ade80",
    "q_value_negative": "#f87171",
    # Move list
    "move_current": "#facc15",
    "move_hover": "#4b5563",
    "move_white_bg": "#2a2a3e",
    "move_black_bg": "#1e1e2e",
    # Opening display
    "opening_eco": "#a78bfa",
}

# Font configurations
FONTS = {
    "title": ("Segoe UI", 24, "bold"),
    "heading": ("Segoe UI", 16, "bold"),
    "subheading": ("Segoe UI", 14, "bold"),
    "body": ("Segoe UI", 12),
    "body_bold": ("Segoe UI", 12, "bold"),
    "small": ("Segoe UI", 10),
    "mono": ("Consolas", 12),
    "mono_small": ("Consolas", 10),
    # Chess specific
    "piece": ("Segoe UI Symbol", 36),
    "piece_large": ("Segoe UI Symbol", 48),
    "coordinates": ("Segoe UI", 10),
}

# Chess piece Unicode characters
PIECE_UNICODE = {
    "K": "\u2654",  # White King
    "Q": "\u2655",  # White Queen
    "R": "\u2656",  # White Rook
    "B": "\u2657",  # White Bishop
    "N": "\u2658",  # White Knight
    "P": "\u2659",  # White Pawn
    "k": "\u265a",  # Black King
    "q": "\u265b",  # Black Queen
    "r": "\u265c",  # Black Rook
    "b": "\u265d",  # Black Bishop
    "n": "\u265e",  # Black Knight
    "p": "\u265f",  # Black Pawn
}

# Alternative piece symbols (for better rendering on some systems)
PIECE_UNICODE_ALT = {
    "K": "♔",
    "Q": "♕",
    "R": "♖",
    "B": "♗",
    "N": "♘",
    "P": "♙",
    "k": "♚",
    "q": "♛",
    "r": "♜",
    "b": "♝",
    "n": "♞",
    "p": "♟",
}

# Status icons for accessibility (don't rely on color alone)
STATUS_ICONS = {
    "success": "✓",
    "warning": "⚠",
    "error": "✗",
    "info": "ℹ",
    "loading": "◌",
    "thinking": "◐",
    "check": "♚",
    "white": "○",
    "black": "●",
    "draw": "½",
}


def apply_theme(root: tk.Tk) -> None:
    """
    Apply the dark theme to a Tkinter root window.

    Args:
        root: The root Tkinter window
    """
    # Configure root window
    root.configure(bg=COLORS["bg_primary"])

    # Create and configure ttk style
    style = ttk.Style(root)

    # Try to use a modern theme as base
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    # Configure ttk widgets
    style.configure(
        "TFrame",
        background=COLORS["bg_primary"],
    )

    style.configure(
        "TLabel",
        background=COLORS["bg_primary"],
        foreground=COLORS["text_primary"],
        font=FONTS["body"],
    )

    style.configure(
        "Heading.TLabel",
        background=COLORS["bg_primary"],
        foreground=COLORS["text_primary"],
        font=FONTS["heading"],
    )

    style.configure(
        "TButton",
        background=COLORS["button_bg"],
        foreground=COLORS["text_primary"],
        font=FONTS["body_bold"],
        padding=(20, 10),
    )

    style.map(
        "TButton",
        background=[
            ("active", COLORS["button_hover"]),
            ("pressed", COLORS["button_active"]),
            ("disabled", COLORS["button_disabled"]),
        ],
        foreground=[
            ("disabled", COLORS["text_muted"]),
        ],
    )

    style.configure(
        "Accent.TButton",
        background=COLORS["accent"],
        foreground=COLORS["text_primary"],
    )

    style.map(
        "Accent.TButton",
        background=[
            ("active", "#ff6b8a"),
            ("pressed", "#c93050"),
        ],
    )

    style.configure(
        "TProgressbar",
        background=COLORS["progress_fill"],
        troughcolor=COLORS["progress_bg"],
        borderwidth=0,
        thickness=10,
    )

    style.configure(
        "TScale",
        background=COLORS["bg_primary"],
        troughcolor=COLORS["progress_bg"],
    )

    style.configure(
        "TEntry",
        fieldbackground=COLORS["bg_tertiary"],
        foreground=COLORS["text_primary"],
        insertcolor=COLORS["text_primary"],
    )

    style.configure(
        "TCombobox",
        fieldbackground=COLORS["bg_tertiary"],
        background=COLORS["bg_tertiary"],
        foreground=COLORS["text_primary"],
    )


def create_styled_button(
    parent: tk.Widget,
    text: str,
    command: callable = None,
    style: str = "normal",
    width: int = None,
) -> tk.Button:
    """
    Create a styled button.

    Args:
        parent: Parent widget
        text: Button text
        command: Button command
        style: "normal", "accent", or "outline"
        width: Button width in characters

    Returns:
        Styled tk.Button
    """
    if style == "accent":
        bg = COLORS["accent"]
        hover_bg = "#ff6b8a"
        active_bg = "#c93050"
    elif style == "outline":
        bg = COLORS["bg_secondary"]
        hover_bg = COLORS["bg_tertiary"]
        active_bg = COLORS["accent"]
    else:
        bg = COLORS["button_bg"]
        hover_bg = COLORS["button_hover"]
        active_bg = COLORS["button_active"]

    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=COLORS["text_primary"],
        font=FONTS["body_bold"],
        relief=tk.FLAT,
        cursor="hand2",
        activebackground=active_bg,
        activeforeground=COLORS["text_primary"],
        padx=20,
        pady=10,
    )

    if width:
        btn.configure(width=width)

    # Hover effects
    def on_enter(e):
        btn.configure(bg=hover_bg)

    def on_leave(e):
        btn.configure(bg=bg)

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

    return btn


def create_styled_label(
    parent: tk.Widget,
    text: str,
    style: str = "body",
    **kwargs,
) -> tk.Label:
    """
    Create a styled label.

    Args:
        parent: Parent widget
        text: Label text
        style: "title", "heading", "subheading", "body", "small", or "mono"

    Returns:
        Styled tk.Label
    """
    font = FONTS.get(style, FONTS["body"])
    fg = kwargs.pop("fg", COLORS["text_primary"])
    bg = kwargs.pop("bg", COLORS["bg_primary"])

    return tk.Label(
        parent,
        text=text,
        font=font,
        fg=fg,
        bg=bg,
        **kwargs,
    )


def create_panel(
    parent: tk.Widget,
    title: str = None,
    padding: int = 15,
) -> tk.Frame:
    """
    Create a styled panel with optional title.

    Args:
        parent: Parent widget
        title: Optional panel title
        padding: Internal padding

    Returns:
        Styled tk.Frame
    """
    frame = tk.Frame(
        parent,
        bg=COLORS["bg_secondary"],
        highlightbackground=COLORS["border"],
        highlightthickness=1,
    )

    if title:
        title_label = tk.Label(
            frame,
            text=title,
            font=FONTS["subheading"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_secondary"],
        )
        title_label.pack(anchor="w", padx=padding, pady=(padding, 5))

    return frame


class ProgressBar(tk.Canvas):
    """Custom styled progress bar."""

    def __init__(
        self,
        parent: tk.Widget,
        width: int = 200,
        height: int = 10,
        **kwargs,
    ):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=COLORS["progress_bg"],
            highlightthickness=0,
            **kwargs,
        )
        self.width = width
        self.height = height
        self._value = 0

    def set_value(self, value: float) -> None:
        """Set progress value (0.0 to 1.0)."""
        self._value = max(0, min(1, value))
        self.delete("progress")

        if self._value > 0:
            fill_width = int(self.width * self._value)
            self.create_rectangle(
                0,
                0,
                fill_width,
                self.height,
                fill=COLORS["progress_fill"],
                outline="",
                tags="progress",
            )

    def get_value(self) -> float:
        """Get current progress value."""
        return self._value


class Spinner(tk.Canvas):
    """Animated loading spinner widget."""

    def __init__(
        self,
        parent: tk.Widget,
        size: int = 24,
        color: str = None,
        **kwargs,
    ):
        super().__init__(
            parent,
            width=size,
            height=size,
            bg=kwargs.pop("bg", COLORS["bg_primary"]),
            highlightthickness=0,
            **kwargs,
        )
        self.size = size
        self.color = color or COLORS["accent"]
        self._angle = 0
        self._running = False
        self._animation_id = None

    def start(self) -> None:
        """Start the spinner animation."""
        if not self._running:
            self._running = True
            self._animate()

    def stop(self) -> None:
        """Stop the spinner animation."""
        self._running = False
        if self._animation_id:
            self.after_cancel(self._animation_id)
            self._animation_id = None
        self.delete("spinner")

    def _animate(self) -> None:
        """Animate the spinner."""
        if not self._running:
            return

        self.delete("spinner")
        cx = self.size // 2
        cy = self.size // 2
        radius = self.size // 2 - 3

        # Draw arc segments with varying opacity
        import math

        for i in range(8):
            start_angle = (self._angle + i * 45) % 360
            # Vary the color intensity for each segment
            intensity = int(255 * (i + 1) / 8)
            segment_color = f"#{intensity:02x}{intensity//4:02x}{intensity//3:02x}"

            x1 = cx + radius * math.cos(math.radians(start_angle))
            y1 = cy + radius * math.sin(math.radians(start_angle))
            x2 = cx + radius * math.cos(math.radians(start_angle + 30))
            y2 = cy + radius * math.sin(math.radians(start_angle + 30))

            self.create_line(
                cx,
                cy,
                x1,
                y1,
                fill=self.color if i == 7 else COLORS["text_muted"],
                width=2,
                tags="spinner",
            )

        self._angle = (self._angle + 45) % 360
        self._animation_id = self.after(100, self._animate)

    def destroy(self) -> None:
        """Clean up before destroying."""
        self.stop()
        super().destroy()


class ThinkingIndicator(tk.Label):
    """Animated 'thinking...' text indicator."""

    def __init__(
        self,
        parent: tk.Widget,
        text: str = "Thinking",
        **kwargs,
    ):
        self.base_text = text
        self._dots = 0
        self._running = False
        self._animation_id = None

        super().__init__(
            parent,
            text=f"{text}...",
            font=kwargs.pop("font", FONTS["body"]),
            fg=kwargs.pop("fg", COLORS["accent"]),
            bg=kwargs.pop("bg", COLORS["bg_primary"]),
            **kwargs,
        )

    def start(self) -> None:
        """Start the animation."""
        if not self._running:
            self._running = True
            self._animate()

    def stop(self) -> None:
        """Stop the animation and hide."""
        self._running = False
        if self._animation_id:
            self.after_cancel(self._animation_id)
            self._animation_id = None

    def _animate(self) -> None:
        """Animate the dots."""
        if not self._running:
            return

        self._dots = (self._dots + 1) % 4
        dots = "." * self._dots + " " * (3 - self._dots)
        self.configure(text=f"{self.base_text}{dots}")
        self._animation_id = self.after(400, self._animate)

    def destroy(self) -> None:
        """Clean up before destroying."""
        self.stop()
        super().destroy()


class Tooltip:
    """Tooltip widget that appears on hover."""

    def __init__(
        self,
        widget: tk.Widget,
        text: str,
        delay: int = 500,
    ):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._tooltip_window: Optional[tk.Toplevel] = None
        self._schedule_id: Optional[str] = None

        widget.bind("<Enter>", self._on_enter)
        widget.bind("<Leave>", self._on_leave)
        widget.bind("<ButtonPress>", self._on_leave)

    def _on_enter(self, event=None) -> None:
        """Schedule tooltip display."""
        self._schedule_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, event=None) -> None:
        """Cancel and hide tooltip."""
        if self._schedule_id:
            self.widget.after_cancel(self._schedule_id)
            self._schedule_id = None
        self._hide()

    def _show(self) -> None:
        """Display the tooltip."""
        if self._tooltip_window:
            return

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self._tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        # Create tooltip label
        label = tk.Label(
            tw,
            text=self.text,
            font=FONTS["small"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_tertiary"],
            relief=tk.SOLID,
            borderwidth=1,
            padx=8,
            pady=4,
        )
        label.pack()

    def _hide(self) -> None:
        """Hide the tooltip."""
        if self._tooltip_window:
            self._tooltip_window.destroy()
            self._tooltip_window = None

    def update_text(self, text: str) -> None:
        """Update tooltip text."""
        self.text = text


def create_tooltip(widget: tk.Widget, text: str, delay: int = 500) -> Tooltip:
    """
    Create a tooltip for a widget.

    Args:
        widget: The widget to attach the tooltip to
        text: The tooltip text
        delay: Delay in ms before showing (default 500)

    Returns:
        Tooltip instance
    """
    return Tooltip(widget, text, delay)


class StatusLabel(tk.Frame):
    """Label with status icon and color for accessibility."""

    def __init__(
        self,
        parent: tk.Widget,
        text: str = "",
        status: str = "info",
        **kwargs,
    ):
        bg = kwargs.pop("bg", COLORS["bg_primary"])
        super().__init__(parent, bg=bg, **kwargs)

        self._status = status
        self._text = text

        # Icon label
        self._icon_label = tk.Label(
            self,
            text=STATUS_ICONS.get(status, ""),
            font=FONTS["body_bold"],
            fg=self._get_status_color(status),
            bg=bg,
        )
        self._icon_label.pack(side=tk.LEFT, padx=(0, 5))

        # Text label
        self._text_label = tk.Label(
            self,
            text=text,
            font=FONTS["body"],
            fg=COLORS["text_primary"],
            bg=bg,
        )
        self._text_label.pack(side=tk.LEFT)

    def _get_status_color(self, status: str) -> str:
        """Get color for status type."""
        color_map = {
            "success": COLORS["success"],
            "warning": COLORS["warning"],
            "error": COLORS["error"],
            "info": COLORS["info"],
            "loading": COLORS["text_secondary"],
            "thinking": COLORS["accent"],
        }
        return color_map.get(status, COLORS["text_primary"])

    def set_status(self, text: str, status: str) -> None:
        """Update status text and type."""
        self._text = text
        self._status = status
        self._icon_label.configure(
            text=STATUS_ICONS.get(status, ""),
            fg=self._get_status_color(status),
        )
        self._text_label.configure(text=text)


def create_confirmation_dialog(
    parent: tk.Widget,
    title: str,
    message: str,
    on_confirm: Callable[[], None],
    on_cancel: Callable[[], None] = None,
    confirm_text: str = "Confirm",
    cancel_text: str = "Cancel",
) -> tk.Toplevel:
    """
    Create a styled confirmation dialog.

    Args:
        parent: Parent widget
        title: Dialog title
        message: Confirmation message
        on_confirm: Callback when confirmed
        on_cancel: Callback when cancelled
        confirm_text: Text for confirm button
        cancel_text: Text for cancel button

    Returns:
        Dialog window
    """
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.configure(bg=COLORS["bg_primary"])
    dialog.transient(parent)
    dialog.grab_set()

    # Center on parent
    dialog.geometry("300x150")
    dialog.resizable(False, False)

    # Message
    msg_label = tk.Label(
        dialog,
        text=message,
        font=FONTS["body"],
        fg=COLORS["text_primary"],
        bg=COLORS["bg_primary"],
        wraplength=260,
    )
    msg_label.pack(pady=30)

    # Buttons frame
    btn_frame = tk.Frame(dialog, bg=COLORS["bg_primary"])
    btn_frame.pack(pady=10)

    def confirm():
        dialog.destroy()
        on_confirm()

    def cancel():
        dialog.destroy()
        if on_cancel:
            on_cancel()

    cancel_btn = create_styled_button(btn_frame, cancel_text, cancel, style="outline")
    cancel_btn.pack(side=tk.LEFT, padx=5)

    confirm_btn = create_styled_button(btn_frame, confirm_text, confirm, style="accent")
    confirm_btn.pack(side=tk.LEFT, padx=5)

    # Handle ESC key
    dialog.bind("<Escape>", lambda e: cancel())

    # Center dialog on screen
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() - 300) // 2
    y = (dialog.winfo_screenheight() - 150) // 2
    dialog.geometry(f"+{x}+{y}")

    return dialog
