"""
Modern dark theme styling for the chess UI.

Defines colors, fonts, and styling utilities for consistent
appearance across the application.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from collections.abc import Callable
from typing import Any, Optional


# Color palette - Modern dark "slate" theme with a single blue accent
COLORS: dict[str, str] = {
    # Background colors
    # Main window background
    "bg_primary": "#121418",
    # Card / panel background
    "bg_secondary": "#1b1e24",
    # Elevated elements (inputs, secondary buttons)
    "bg_tertiary": "#252932",
    # Hovered elevated element
    "bg_hover": "#2f343e",
    # Chess board colors (cool blue-grey board)
    "light_square": "#dee3e6",
    "dark_square": "#8ca2ad",
    # Editor palette buttons (mid-tone so both white and black pieces stand out)
    "palette_bg": "#8ca2ad",
    # Coordinates drawn on light / dark squares
    "coord_on_light": "#7d939e",
    "coord_on_dark": "#eef1f3",
    # Highlight colors
    # Selected square
    "selected_light": "#f3e07a",
    "selected_dark": "#c8b44f",
    # Drag hover on light square
    "hover_light": "#ffffff",
    # Drag hover on dark square
    "hover_dark": "#f2f5f7",
    # Legal move indicator
    "legal_move": "#55707e",
    # Last move highlight
    "last_move": "#cfdb8f",
    "last_move_dark": "#a2b56a",
    # King in check
    "check": "#f25f5c",
    # Accent colors
    # Primary accent (blue)
    "accent": "#5b8cff",
    "accent_hover": "#7aa2ff",
    "accent_active": "#4570e0",
    # Selected tile background
    "accent_soft": "#1f2b47",
    # Secondary accent (green)
    "accent_secondary": "#3ecf8e",
    # Tertiary accent (cyan)
    "accent_tertiary": "#4cc9f0",
    # Text colors
    "text_primary": "#eceef2",
    "text_secondary": "#a9b0bc",
    "text_muted": "#6c7380",
    # Button colors
    "button_bg": "#252932",
    "button_hover": "#2f343e",
    "button_active": "#5b8cff",
    "button_disabled": "#1f2228",
    # Status colors
    "success": "#3ecf8e",
    "warning": "#f5b544",
    "error": "#f25f5c",
    "info": "#5b8cff",
    # Border colors
    "border": "#2a2e37",
    "border_light": "#3a404c",
    # Progress bar
    "progress_bg": "#2a2e37",
    "progress_fill": "#5b8cff",
    # Evaluation bar
    "eval_white": "#f2f4f7",
    "eval_black": "#2a2e36",
    "eval_winning": "#3ecf8e",
    "eval_losing": "#f25f5c",
    "eval_equal": "#6c7380",
    # Win / draw / loss segments (beginner "who is winning" bar)
    "white_side": "#f2f4f7",
    "draw_side": "#7b8494",
    "black_side": "#3a3f4a",
    # Player info
    "player_active": "#5b8cff",
    "player_active_glow": "#7aa2ff",
    "captured_bg": "#252932",
    # Evaluation graph
    "graph_bg": "#1b1e24",
    "graph_line": "#5b8cff",
    "graph_line_alt": "#f472b6",
    "graph_grid": "#2a2e37",
    "graph_zero": "#4a5160",
    "graph_marker": "#f5b544",
    # MCTS panel
    "visit_bar": "#5b8cff",
    "prior_bar": "#4cc9f0",
    "q_value_positive": "#3ecf8e",
    "q_value_negative": "#f25f5c",
    # Move list
    "move_current": "#5b8cff",
    "move_hover": "#2f343e",
    "move_white_bg": "#1f2228",
    "move_black_bg": "#1b1e24",
    # Opening display
    "opening_eco": "#a78bfa",
    # Timer colors
    # Red - < 30s
    "timer_urgent": "#f25f5c",
    # Orange - < 1min
    "timer_warning": "#f5b544",
    # Active player timer background
    "player_active_bg": "#5b8cff",
}

# Font configurations
FONTS = {
    "title": ("Segoe UI Semibold", 22),
    "heading": ("Segoe UI Semibold", 15),
    "subheading": ("Segoe UI Semibold", 12),
    "body": ("Segoe UI", 11),
    "body_bold": ("Segoe UI Semibold", 11),
    "small": ("Segoe UI", 10),
    "small_bold": ("Segoe UI Semibold", 10),
    "caption": ("Segoe UI", 9),
    "caption_bold": ("Segoe UI Semibold", 9),
    "big_number": ("Segoe UI Semibold", 24),
    "mono": ("Consolas", 12),
    "mono_small": ("Consolas", 10),
    # Chess specific
    "coordinates": ("Segoe UI Semibold", 10),
    "move": ("Segoe UI Semibold", 11),
}

# Chess piece Unicode characters
PIECE_UNICODE: dict[str, str] = {
    # White King
    "K": "♔",
    # White Queen
    "Q": "♕",
    # White Rook
    "R": "♖",
    # White Bishop
    "B": "♗",
    # White Knight
    "N": "♘",
    # White Pawn
    "P": "♙",
    # Black King
    "k": "♚",
    # Black Queen
    "q": "♛",
    # Black Rook
    "r": "♜",
    # Black Bishop
    "b": "♝",
    # Black Knight
    "n": "♞",
    # Black Pawn
    "p": "♟",
}

# Alternative piece symbols (for better rendering on some systems)
PIECE_UNICODE_ALT: dict[str, str] = {
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
STATUS_ICONS: dict[str, str] = {
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
            ("active", COLORS["accent_hover"]),
            ("pressed", COLORS["accent_active"]),
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
        background=COLORS["accent"],
        troughcolor=COLORS["bg_tertiary"],
        bordercolor=COLORS["bg_secondary"],
        lightcolor=COLORS["accent"],
        darkcolor=COLORS["accent"],
        borderwidth=0,
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

    # Slim dark scrollbars
    for orient in ("Vertical", "Horizontal"):
        style.configure(
            f"{orient}.TScrollbar",
            background=COLORS["bg_tertiary"],
            troughcolor=COLORS["bg_secondary"],
            bordercolor=COLORS["bg_secondary"],
            lightcolor=COLORS["bg_tertiary"],
            darkcolor=COLORS["bg_tertiary"],
            arrowcolor=COLORS["text_muted"],
            gripcount=0,
        )
        style.map(
            f"{orient}.TScrollbar",
            background=[("active", COLORS["bg_hover"])],
        )


# Button style presets: (background, hover, pressed, foreground)
_BUTTON_STYLES: dict[str, tuple[str, str, str, str]] = {
    "accent": (
        COLORS["accent"],
        COLORS["accent_hover"],
        COLORS["accent_active"],
        "#ffffff",
    ),
    "normal": (
        COLORS["button_bg"],
        COLORS["button_hover"],
        COLORS["accent_active"],
        COLORS["text_primary"],
    ),
    "outline": (
        COLORS["bg_tertiary"],
        COLORS["bg_hover"],
        COLORS["accent_active"],
        COLORS["text_primary"],
    ),
    "ghost": (
        COLORS["bg_secondary"],
        COLORS["bg_tertiary"],
        COLORS["bg_hover"],
        COLORS["text_secondary"],
    ),
    "danger": (
        "#3a2226",
        "#4a2a2f",
        COLORS["error"],
        "#ffb4b2",
    ),
}


def create_styled_button(
    parent: tk.Widget,
    text: str,
    command: Callable[[], None] | None = None,
    style: str = "normal",
    width: int | None = None,
    font: tuple | None = None,
    padx: int = 16,
    pady: int = 8,
) -> tk.Button:
    """
    Create a styled flat button with hover feedback.

    Args:
        parent: Parent widget
        text: Button text
        command: Button command
        style: "accent", "normal", "outline", "ghost" or "danger"
        width: Button width in characters
        font: Optional font override
        padx: Horizontal inner padding
        pady: Vertical inner padding

    Returns:
        Styled tk.Button
    """
    bg, hover_bg, active_bg, fg = _BUTTON_STYLES.get(style, _BUTTON_STYLES["normal"])

    btn = tk.Button(
        parent,
        text=text,
        bg=bg,
        fg=fg,
        font=font or FONTS["body_bold"],
        relief=tk.FLAT,
        bd=0,
        highlightthickness=0,
        cursor="hand2",
        activebackground=active_bg,
        activeforeground="#ffffff",
        disabledforeground=COLORS["text_muted"],
        padx=padx,
        pady=pady,
    )
    if command is not None:
        btn.configure(command=command)

    if width:
        btn.configure(width=width)

    # Hover effects
    def on_enter(e: tk.Event[tk.Button]) -> None:
        """Apply hover background color."""
        if str(btn["state"]) != "disabled":
            btn.configure(bg=hover_bg)

    def on_leave(e: tk.Event[tk.Button]) -> None:
        """Restore default background color."""
        btn.configure(bg=bg)

    btn.bind("<Enter>", on_enter, add="+")
    btn.bind("<Leave>", on_leave, add="+")

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
    title: str | None = None,
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


class Card(tk.Frame):
    """
    Card container with an optional header (title + help badge) and a body.

    Children should be packed into ``card.body``.
    """

    def __init__(
        self,
        parent: tk.Widget,
        title: str | None = None,
        help_text: str | None = None,
        padding: int = 14,
        **kwargs,
    ) -> None:
        bg = kwargs.pop("bg", COLORS["bg_secondary"])
        super().__init__(
            parent,
            bg=bg,
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            **kwargs,
        )

        self.header: tk.Frame | None = None
        self.title_label: tk.Label | None = None
        if title:
            self.header = tk.Frame(self, bg=bg)
            self.header.pack(fill=tk.X, padx=padding, pady=(padding - 2, 8))

            self.title_label = tk.Label(
                self.header,
                text=title,
                font=FONTS["subheading"],
                fg=COLORS["text_primary"],
                bg=bg,
                anchor="w",
            )
            self.title_label.pack(side=tk.LEFT)

            if help_text:
                HelpBadge(self.header, help_text).pack(side=tk.LEFT, padx=(8, 0))

        self.body = tk.Frame(self, bg=bg)
        self.body.pack(
            fill=tk.BOTH,
            expand=True,
            padx=padding,
            pady=(0 if title else padding, padding),
        )

    def set_title(self, title: str) -> None:
        """Update the card title text."""
        if self.title_label is not None:
            self.title_label.configure(text=title)


class HelpBadge(tk.Label):
    """Small "?" badge that explains something in plain words on hover."""

    def __init__(self, parent: tk.Widget, text: str, **kwargs) -> None:
        super().__init__(
            parent,
            text=" ? ",
            font=FONTS["caption_bold"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_tertiary"],
            cursor="question_arrow",
            **kwargs,
        )
        Tooltip(self, text, delay=120)


class SegmentedControl(tk.Frame):
    """Row of mutually exclusive buttons bound to a StringVar."""

    def __init__(
        self,
        parent: tk.Widget,
        options: list[tuple[str, str]],
        variable: tk.StringVar,
        command: Callable[[], None] | None = None,
        font: tuple | None = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, bg=COLORS["bg_tertiary"], padx=3, pady=3, **kwargs)
        self._variable: tk.StringVar = variable
        self._command: Callable[[], None] | None = command
        self._buttons: dict[str, tk.Label] = {}

        for i, (label, value) in enumerate(options):
            btn = tk.Label(
                self,
                text=label,
                font=font or FONTS["small_bold"],
                cursor="hand2",
                padx=8,
                pady=6,
            )
            btn.grid(row=0, column=i, sticky="nsew", padx=1)
            self.grid_columnconfigure(i, weight=1, uniform="seg")
            btn.bind("<Button-1>", lambda e, v=value: self._select(v))
            btn.bind("<Enter>", lambda e, b=btn, v=value: self._hover(b, v, True))
            btn.bind("<Leave>", lambda e, b=btn, v=value: self._hover(b, v, False))
            self._buttons[value] = btn

        variable.trace_add("write", lambda *_: self.refresh())
        self.refresh()

    def _select(self, value: str) -> None:
        """Select a value and fire the callback."""
        changed = self._variable.get() != value
        self._variable.set(value)
        if changed and self._command is not None:
            self._command()

    def _hover(self, btn: tk.Label, value: str, entering: bool) -> None:
        """Hover feedback on non-selected segments."""
        if self._variable.get() == value:
            return
        btn.configure(bg=COLORS["bg_hover"] if entering else COLORS["bg_tertiary"])

    def refresh(self) -> None:
        """Redraw the selection state."""
        current = self._variable.get()
        for value, btn in self._buttons.items():
            if value == current:
                btn.configure(bg=COLORS["accent"], fg="#ffffff")
            else:
                btn.configure(bg=COLORS["bg_tertiary"], fg=COLORS["text_secondary"])


class ChoiceList(tk.Frame):
    """Vertical list of large selectable tiles (icon, title, description)."""

    def __init__(
        self,
        parent: tk.Widget,
        options: list[tuple[str, str, str, str]],
        variable: tk.StringVar,
        command: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        bg = kwargs.pop("bg", COLORS["bg_secondary"])
        super().__init__(parent, bg=bg, **kwargs)
        self._variable: tk.StringVar = variable
        self._command: Callable[[], None] | None = command
        # value -> (tile, text_frame, icon, title, desc)
        self._tiles: dict[
            str, tuple[tk.Frame, tk.Frame, tk.Label, tk.Label, tk.Label]
        ] = {}

        for value, icon, title, desc in options:
            tile = tk.Frame(
                self,
                highlightthickness=1,
                cursor="hand2",
            )
            tile.pack(fill=tk.X, pady=2)

            icon_lbl = tk.Label(
                tile,
                text=icon,
                font=("Segoe UI Symbol", 15),
                width=2,
                cursor="hand2",
            )
            icon_lbl.pack(side=tk.LEFT, padx=(8, 6), pady=5)

            text_frame = tk.Frame(tile, cursor="hand2")
            text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=5)

            title_lbl = tk.Label(
                text_frame,
                text=title,
                font=FONTS["small_bold"],
                anchor="w",
                cursor="hand2",
            )
            title_lbl.pack(fill=tk.X)
            desc_lbl = tk.Label(
                text_frame,
                text=desc,
                font=FONTS["caption"],
                anchor="w",
                cursor="hand2",
            )
            desc_lbl.pack(fill=tk.X)

            for w in (tile, text_frame, icon_lbl, title_lbl, desc_lbl):
                w.bind("<Button-1>", lambda e, v=value: self._select(v))
                w.bind("<Enter>", lambda e, v=value: self._hover(v, True))
                w.bind("<Leave>", lambda e, v=value: self._hover(v, False))
            self._tiles[value] = (tile, text_frame, icon_lbl, title_lbl, desc_lbl)

        variable.trace_add("write", lambda *_: self.refresh())
        self.refresh()

    def _select(self, value: str) -> None:
        """Select a tile and fire the callback."""
        changed = self._variable.get() != value
        self._variable.set(value)
        if changed and self._command is not None:
            self._command()

    def _hover(self, value: str, entering: bool) -> None:
        """Hover feedback on non-selected tiles."""
        if self._variable.get() == value:
            return
        tile = self._tiles[value][0]
        tile.configure(
            highlightbackground=(
                COLORS["border_light"] if entering else COLORS["bg_tertiary"]
            )
        )

    def refresh(self) -> None:
        """Redraw the selection state."""
        current = self._variable.get()
        for value, (tile, text_frame, icon, title, desc) in self._tiles.items():
            selected = value == current
            bg = COLORS["accent_soft"] if selected else COLORS["bg_tertiary"]
            tile.configure(
                bg=bg,
                highlightbackground=COLORS["accent"] if selected else bg,
                highlightcolor=COLORS["accent"] if selected else bg,
            )
            for w in (text_frame, icon, title, desc):
                w.configure(bg=bg)
            icon.configure(fg=COLORS["text_primary"])
            title.configure(
                fg=COLORS["text_primary"] if selected else COLORS["text_secondary"]
            )
            desc.configure(fg=COLORS["text_secondary"])
            # Only the selected tile shows its description (keeps the list compact)
            if selected:
                desc.pack(fill=tk.X)
            else:
                desc.pack_forget()


class Slider(tk.Canvas):
    """
    Flat slider with a round knob, bound to an IntVar.

    Uses a logarithmic scale when ``log=True`` so that small values
    (1, 10, 100) are as easy to pick as large ones (1000, 2000).
    """

    KNOB_RADIUS = 8
    HEIGHT = 24

    def __init__(
        self,
        parent: tk.Widget,
        from_: int,
        to: int,
        variable: tk.IntVar,
        command: Callable[[str], None] | None = None,
        log: bool = False,
        **kwargs,
    ) -> None:
        bg = kwargs.pop("bg", COLORS["bg_secondary"])
        super().__init__(
            parent,
            height=self.HEIGHT,
            bg=bg,
            highlightthickness=0,
            cursor="hand2",
            **kwargs,
        )
        self._from: int = from_
        self._to: int = to
        self._variable: tk.IntVar = variable
        self._command: Callable[[str], None] | None = command
        self._log: bool = log

        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Button-1>", self._on_mouse)
        self.bind("<B1-Motion>", self._on_mouse)
        variable.trace_add("write", lambda *_: self._draw())

    def _to_ratio(self, value: float) -> float:
        """Map a value to a position in [0, 1]."""
        import math

        value = max(self._from, min(self._to, value))
        if self._log:
            return math.log(value / self._from) / math.log(self._to / self._from)
        return (value - self._from) / (self._to - self._from)

    def _from_ratio(self, ratio: float) -> int:
        """Map a position in [0, 1] to a value."""
        ratio = max(0.0, min(1.0, ratio))
        if self._log:
            value = self._from * (self._to / self._from) ** ratio
        else:
            value = self._from + ratio * (self._to - self._from)
        # Round to "nice" values so the label doesn't flicker
        if value >= 200:
            return int(round(value / 50) * 50)
        if value >= 20:
            return int(round(value / 10) * 10)
        return max(self._from, int(round(value)))

    def _draw(self) -> None:
        """Redraw track, filled part and knob."""
        self.delete("all")
        r = self.KNOB_RADIUS
        width = max(2 * r + 2, self.winfo_width())
        cy = self.HEIGHT // 2
        left, right = r + 1, width - r - 1
        try:
            value = self._variable.get()
        except tk.TclError:
            value = self._from
        x = left + (right - left) * self._to_ratio(value)

        self.create_line(
            left, cy, right, cy, fill=COLORS["bg_hover"], width=4, capstyle=tk.ROUND
        )
        self.create_line(
            left, cy, x, cy, fill=COLORS["accent"], width=4, capstyle=tk.ROUND
        )
        self.create_oval(
            x - r,
            cy - r,
            x + r,
            cy + r,
            fill="#ffffff",
            outline=COLORS["accent"],
            width=2,
        )

    def _on_mouse(self, event: tk.Event) -> None:
        """Move the knob under the mouse."""
        r = self.KNOB_RADIUS
        width = max(2 * r + 2, self.winfo_width())
        ratio = (event.x - r - 1) / max(1, width - 2 * r - 2)
        value = self._from_ratio(ratio)
        if value != self._variable.get():
            self._variable.set(value)
            if self._command is not None:
                self._command(str(value))


# Piece image cache: (symbol, size) -> PhotoImage
_PIECE_IMAGE_CACHE: dict[tuple[str, int], Any] = {}


def load_piece_image(symbol: str, size: int) -> Any:
    """
    Load a piece PNG from assets/pieces at the requested size.

    Args:
        symbol: python-chess piece symbol ("K", "q", ...)
        size: Edge length in pixels

    Returns:
        A PhotoImage, or None if PIL or the asset is unavailable.
    """
    key = (symbol, size)
    if key in _PIECE_IMAGE_CACHE:
        return _PIECE_IMAGE_CACHE[key]
    try:
        from pathlib import Path
        from PIL import Image, ImageTk

        color = "w" if symbol.isupper() else "b"
        path = (
            Path(__file__).parent.parent.parent
            / "assets"
            / "pieces"
            / f"{color}{symbol.lower()}.png"
        )
        if not path.exists():
            return None
        img = Image.open(path).convert("RGBA")
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
    except Exception:
        return None
    _PIECE_IMAGE_CACHE[key] = photo
    return photo


class ProgressBar(tk.Canvas):
    """Custom styled progress bar."""

    def __init__(
        self,
        parent: tk.Widget,
        width: int = 200,
        height: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=COLORS["progress_bg"],
            highlightthickness=0,
            **kwargs,
        )
        self.width: int = width
        self.height: int = height
        self._value = 0

    def set_value(self, value: float) -> None:
        """Set progress value (0.0 to 1.0)."""
        self._value: float = max(0, min(1, value))
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
        color: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            parent,
            width=size,
            height=size,
            bg=kwargs.pop("bg", COLORS["bg_primary"]),
            highlightthickness=0,
            **kwargs,
        )
        self.size: int = size
        self.color: str = color or COLORS["accent"]
        self._angle: int = 0
        self._running: bool = False
        self._animation_id: str | None = None

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
        cx: int = self.size // 2
        cy: int = self.size // 2
        radius: int = self.size // 2 - 3

        # Draw arc segments with varying opacity
        import math

        for i in range(8):
            start_angle: int = (self._angle + i * 45) % 360

            x1: float = cx + radius * math.cos(math.radians(start_angle))
            y1: float = cy + radius * math.sin(math.radians(start_angle))

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
    ) -> None:
        self.base_text: str = text
        self._dots: int = 0
        self._running: bool = False
        self._animation_id: str | None = None

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
        dots: str = "." * self._dots + " " * (3 - self._dots)
        self.configure(text=f"{self.base_text}{dots}")
        self._animation_id = self.after(400, self._animate)

    def destroy(self) -> None:
        """Clean up before destroying."""
        self.stop()
        super().destroy()


class Tooltip(object):
    """Tooltip widget that appears on hover."""

    def __init__(
        self,
        widget: tk.Widget,
        text: str,
        delay: int = 500,
    ) -> None:
        self.widget: tk.Widget = widget
        self.text: str = text
        self.delay: int = delay
        self._tooltip_window: Optional[tk.Toplevel] = None
        self._schedule_id: Optional[str] = None

        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, event: tk.Event[tk.Widget] | None = None) -> None:
        """Schedule tooltip display."""
        self._schedule_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, event: tk.Event[tk.Widget] | None = None) -> None:
        """Cancel and hide tooltip."""
        if self._schedule_id:
            self.widget.after_cancel(self._schedule_id)
            self._schedule_id = None
        self._hide()

    def _show(self) -> None:
        """Display the tooltip."""
        if self._tooltip_window or not self.text:
            return

        x: int = self.widget.winfo_rootx() + 10
        y: int = self.widget.winfo_rooty() + self.widget.winfo_height() + 6

        self._tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.configure(bg=COLORS["border_light"])

        # Create tooltip label
        label = tk.Label(
            tw,
            text=self.text,
            font=FONTS["small"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_tertiary"],
            justify=tk.LEFT,
            wraplength=320,
            padx=10,
            pady=6,
        )
        label.pack(padx=1, pady=1)

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
    ) -> None:
        bg = kwargs.pop("bg", COLORS["bg_primary"])
        super().__init__(parent, bg=bg, **kwargs)

        self._status: str = status
        self._text: str = text

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
        color_map: dict[str, str] = {
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
    parent: tk.Misc,
    title: str,
    message: str,
    on_confirm: Callable[[], None],
    on_cancel: Callable[[], None] | None = None,
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
    dialog.configure(bg=COLORS["bg_secondary"])
    dialog.transient(parent.winfo_toplevel())
    dialog.resizable(False, False)

    body = tk.Frame(dialog, bg=COLORS["bg_secondary"])
    body.pack(fill=tk.BOTH, expand=True, padx=28, pady=24)

    tk.Label(
        body,
        text=title,
        font=FONTS["heading"],
        fg=COLORS["text_primary"],
        bg=COLORS["bg_secondary"],
        anchor="w",
    ).pack(fill=tk.X)

    # Message
    tk.Label(
        body,
        text=message,
        font=FONTS["body"],
        fg=COLORS["text_secondary"],
        bg=COLORS["bg_secondary"],
        wraplength=320,
        justify=tk.LEFT,
        anchor="w",
    ).pack(fill=tk.X, pady=(8, 20))

    # Buttons frame
    btn_frame = tk.Frame(body, bg=COLORS["bg_secondary"])
    btn_frame.pack(fill=tk.X)

    def confirm() -> None:
        """Close dialog and execute confirm callback."""
        dialog.destroy()
        on_confirm()

    def cancel() -> None:
        """Close dialog and execute cancel callback if provided."""
        dialog.destroy()
        if on_cancel:
            on_cancel()

    confirm_btn: tk.Button = create_styled_button(
        btn_frame, confirm_text, confirm, style="accent"
    )
    confirm_btn.pack(side=tk.RIGHT)

    cancel_btn: tk.Button = create_styled_button(
        btn_frame, cancel_text, cancel, style="outline"
    )
    cancel_btn.pack(side=tk.RIGHT, padx=(0, 8))

    # Handle ESC / Enter keys
    dialog.bind("<Escape>", lambda e: cancel())
    dialog.bind("<Return>", lambda e: confirm())
    dialog.protocol("WM_DELETE_WINDOW", cancel)

    # Center dialog on parent window
    dialog.update_idletasks()
    top = parent.winfo_toplevel()
    w: int = dialog.winfo_reqwidth()
    h: int = dialog.winfo_reqheight()
    x: int = top.winfo_rootx() + (top.winfo_width() - w) // 2
    y: int = top.winfo_rooty() + (top.winfo_height() - h) // 3
    dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
    dialog.grab_set()
    dialog.focus_set()

    return dialog


# French piece names, used to describe moves in plain words
FRENCH_PIECE_NAMES: dict[int, str] = {
    1: "Pion",
    2: "Cavalier",
    3: "Fou",
    4: "Tour",
    5: "Dame",
    6: "Roi",
}

# English SAN piece letter -> French piece name
_SAN_PIECE_NAMES: dict[str, str] = {
    "K": "Roi",
    "Q": "Dame",
    "R": "Tour",
    "B": "Fou",
    "N": "Cavalier",
}


def san_to_words(san: str) -> str:
    """
    Write an English SAN move with full French piece names.

    Examples: "Bc5" -> "Fou c5", "Nxe5+" -> "Cavalier prend e5 échec",
    "exd5" -> "Pion prend d5", "e8=Q" -> "Pion e8 = Dame", "O-O" -> "Petit roque".
    """
    import re

    suffix = ""
    if san.endswith("#"):
        suffix = " mat"
    elif san.endswith("+"):
        suffix = " échec"
    core = san.rstrip("+#")

    if core in ("O-O", "0-0"):
        return "Petit roque" + suffix
    if core in ("O-O-O", "0-0-0"):
        return "Grand roque" + suffix

    match = re.fullmatch(
        r"([KQRBN])?([a-h]?[1-8]?)(x)?([a-h][1-8])(?:=([QRBN]))?", core
    )
    if match is None:
        return san
    piece, disambiguation, capture, square, promotion = match.groups()

    words = [_SAN_PIECE_NAMES[piece] if piece else "Pion"]
    # Disambiguation only matters for pieces ("Cavalier b d7"), not pawn captures
    if piece and disambiguation:
        words.append(f"({disambiguation})")
    if capture:
        words.append("prend")
    words.append(square)
    if promotion:
        words.append(f"= {_SAN_PIECE_NAMES[promotion]}")
    return " ".join(words) + suffix
