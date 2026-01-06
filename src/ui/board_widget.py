"""
Interactive chess board widget for Tkinter.

Provides a canvas-based chess board with:
- Piece rendering using downloaded PNG images or Unicode fallback
- Click-to-move and drag-and-drop support
- Move highlighting (legal moves, last move, check)
- Coordinate labels
"""

import tkinter as tk
from typing import Callable, Any
from pathlib import Path

try:
    import chess
except ImportError:
    raise ImportError(
        "python-chess is required. Install with: pip install python-chess"
    )

# Try to import PIL for image handling
_USE_IMAGES = False
try:
    from PIL import Image, ImageTk

    _USE_IMAGES = True
except ImportError:
    pass  # Fall back to Unicode

# Piece assets directory
_ASSETS_DIR = Path(__file__).parent.parent.parent / "assets" / "pieces"

# Map chess symbols to asset filenames
_SYMBOL_TO_FILE = {
    "K": "wk",
    "Q": "wq",
    "R": "wr",
    "B": "wb",
    "N": "wn",
    "P": "wp",
    "k": "bk",
    "q": "bq",
    "r": "br",
    "b": "bb",
    "n": "bn",
    "p": "bp",
}

from .styles import COLORS, FONTS, PIECE_UNICODE, PIECE_UNICODE_ALT


class PromotionDialog(tk.Toplevel):
    """Dialog for choosing pawn promotion piece."""

    def __init__(
        self,
        parent: tk.Widget,
        color: bool,  # chess.WHITE or chess.BLACK
        position: tuple[int, int] = None,
    ):
        """
        Initialize promotion dialog.

        Args:
            parent: Parent widget
            color: Color of the promoting pawn (chess.WHITE/BLACK)
            position: Optional (x, y) position for the dialog
        """
        super().__init__(parent)

        self.result: int | None = None
        self.color = color

        # Configure dialog
        self.title("Promotion")
        self.transient(parent)
        self.resizable(False, False)
        self.configure(bg=COLORS["bg_primary"])

        # Create piece buttons first to get dimensions
        self._create_buttons()

        # Position dialog after creating buttons
        self.update_idletasks()

        dialog_width = self.winfo_reqwidth()
        dialog_height = self.winfo_reqheight()

        if position:
            x, y = position
            # Ensure dialog stays on screen
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            x = max(0, min(x, screen_width - dialog_width))
            y = max(0, min(y, screen_height - dialog_height))
            self.geometry(f"+{x}+{y}")
        else:
            # Center on parent
            x = parent.winfo_rootx() + (parent.winfo_width() - dialog_width) // 2
            y = parent.winfo_rooty() + (parent.winfo_height() - dialog_height) // 2
            self.geometry(f"+{x}+{y}")

        # Handle escape key and window close
        self.bind("<Escape>", lambda e: self._cancel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        # Grab focus after positioning
        self.grab_set()
        self.focus_set()

        # Lift to top
        self.lift()
        self.attributes("-topmost", True)

    def _create_buttons(self) -> None:
        """Create piece selection buttons."""
        frame = tk.Frame(self, bg=COLORS["bg_secondary"], padx=5, pady=5)
        frame.pack()

        # Title
        title = tk.Label(
            frame,
            text="Choose promotion:",
            font=FONTS["small"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_secondary"],
        )
        title.pack(pady=(5, 10))

        # Piece buttons container
        btn_frame = tk.Frame(frame, bg=COLORS["bg_secondary"])
        btn_frame.pack()

        # Pieces to choose from
        pieces = [
            (chess.QUEEN, "Q" if self.color else "q", "Queen"),
            (chess.ROOK, "R" if self.color else "r", "Rook"),
            (chess.BISHOP, "B" if self.color else "b", "Bishop"),
            (chess.KNIGHT, "N" if self.color else "n", "Knight"),
        ]

        for piece_type, symbol, name in pieces:
            btn = tk.Button(
                btn_frame,
                text=PIECE_UNICODE_ALT.get(symbol, symbol),
                font=("Segoe UI Symbol", 28),
                width=2,
                height=1,
                bg=COLORS["bg_tertiary"],
                fg=COLORS["text_primary"],
                activebackground=COLORS["accent"],
                activeforeground=COLORS["text_primary"],
                relief=tk.FLAT,
                cursor="hand2",
                command=lambda pt=piece_type: self._select(pt),
            )
            btn.pack(side=tk.LEFT, padx=3, pady=3)

            # Hover effect
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=COLORS["button_hover"]))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg=COLORS["bg_tertiary"]))

    def _select(self, piece_type: int) -> None:
        """Select a promotion piece."""
        self.result = piece_type
        self.destroy()

    def _cancel(self) -> None:
        """Cancel promotion (default to queen)."""
        self.result = chess.QUEEN
        self.destroy()

    def show(self) -> int:
        """Show dialog and return selected piece type."""
        self.wait_window()
        return self.result if self.result is not None else chess.QUEEN


class ChessBoardWidget(tk.Canvas):
    """
    Interactive chess board canvas widget.

    Handles board rendering, piece display, and user interaction
    for making moves.
    """

    def __init__(
        self,
        parent: tk.Widget,
        size: int = 480,
        on_move: Callable[[chess.Move], None] | None = None,
        flipped: bool = False,
    ):
        """
        Initialize the chess board widget.

        Args:
            parent: Parent widget
            size: Board size in pixels (must be divisible by 8)
            on_move: Callback when a move is made
            flipped: Whether to flip the board (black at bottom)
        """
        super().__init__(
            parent,
            width=size,
            height=size,
            bg=COLORS["bg_primary"],
            highlightthickness=0,
        )

        self.size = size
        self.square_size = size // 8
        self.on_move = on_move
        self.flipped = flipped

        # Chess state
        self.board = chess.Board()
        self.selected_square: int | None = None
        self.legal_moves: set[chess.Move] = set()
        self.last_move: chess.Move | None = None
        self.dragging = False
        self.drag_piece: str | None = None
        self.drag_from: int | None = None
        self.drag_pos: tuple[int, int] | None = None
        self._drag_hover_square: int | None = None
        self._drag_scale: float = 1.0
        self._drag_scale_start: float = 1.0
        self._drag_scale_target: float = 1.0
        self._drag_scale_anim_id: int | None = None
        self._drag_scale_start_time: float | None = None
        self._drag_scale_duration: float = 0.1
        self._drag_scale_on_complete: callable = None
        # Drag position animation state
        self._drag_pos_start: tuple[int, int] | None = None
        self._drag_pos_target: tuple[int, int] | None = None
        self._drag_pos_anim_id: int | None = None
        self._drag_pos_start_time: float | None = None
        self._drag_pos_duration: float = 0.15
        self._drag_pos_on_complete: callable = None

        # Interaction state
        self.interactive = True
        self.player_color: chess.Color | None = chess.WHITE  # None = both colors

        # Animation state
        self._animation_id: int | None = None
        self._animating = False
        self._anim_rook: dict | None = None  # For castling animation

        # Piece image cache
        self._piece_images: dict[str, Any] = {}
        self._piece_images_raw: dict[str, Any] = {}  # Raw PIL images for scaling
        self._drag_scaled_image: Any = None  # Cached scaled image for drag
        self._shrink_at_square: int | None = None  # Square where shrink animation plays
        self._use_images = _USE_IMAGES

        # Bind events
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

        # Load piece images
        if self._use_images:
            self._load_piece_images()

        # Initial draw
        self._draw_board()

    def set_board(self, board: chess.Board) -> None:
        """Set the board position."""
        self.board = board.copy()
        self.selected_square = None
        self.legal_moves = set()
        self._draw_board()

    def set_position(self, fen: str) -> None:
        """Set board position from FEN string."""
        self.board = chess.Board(fen)
        self.selected_square = None
        self.legal_moves = set()
        self._draw_board()

    def set_last_move(self, move: chess.Move | None) -> None:
        """Set the last move for highlighting."""
        self.last_move = move
        self._draw_board()

    def set_interactive(self, interactive: bool) -> None:
        """Enable/disable interaction."""
        self.interactive = interactive
        if not interactive:
            self.selected_square = None
            self.legal_moves = set()
            self._draw_board()

    def set_player_color(self, color: chess.Color | None) -> None:
        """Set which color the human plays (None = both)."""
        self.player_color = color

    def flip(self) -> None:
        """Flip the board orientation."""
        self.flipped = not self.flipped
        self._draw_board()

    def _square_to_coords(self, square: int) -> tuple[int, int]:
        """Convert square index to canvas coordinates."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        if self.flipped:
            x = (7 - file) * self.square_size
            y = rank * self.square_size
        else:
            x = file * self.square_size
            y = (7 - rank) * self.square_size

        return x, y

    def _coords_to_square(self, x: int, y: int) -> int:
        """Convert canvas coordinates to square index."""
        file = x // self.square_size
        rank = 7 - (y // self.square_size)

        if self.flipped:
            file = 7 - file
            rank = 7 - rank

        return chess.square(file, rank)

    def _draw_board(self) -> None:
        """Redraw the entire board."""
        self.delete("all")

        # Draw squares
        for square in chess.SQUARES:
            self._draw_square(square)

        # Draw coordinates
        self._draw_coordinates()

        # Draw pieces
        for square in chess.SQUARES:
            # Hide piece if dragging from that square or shrinking at that square
            skip_drag = self.dragging and square == self.drag_from
            skip_shrink = (
                self._shrink_at_square is not None and square == self._shrink_at_square
            )
            if not (skip_drag or skip_shrink):
                self._draw_piece(square)

        # Draw dragged piece on top
        if self.dragging and self.drag_piece and self.drag_pos:
            self._draw_dragged_piece()

    def _draw_square(self, square: int) -> None:
        """Draw a single square with appropriate highlighting."""
        x, y = self._square_to_coords(square)
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        # Base color
        is_light = (file + rank) % 2 == 1
        color = COLORS["light_square"] if is_light else COLORS["dark_square"]

        # Highlight selected square
        if square == self.selected_square:
            color = COLORS["selected_light"] if is_light else COLORS["selected_dark"]

        # Highlight last move
        elif self.last_move and (
            square == self.last_move.from_square or square == self.last_move.to_square
        ):
            # Blend with last move highlight
            color = COLORS["last_move"] if is_light else "#c4c444"

        # Highlight king in check
        elif self.board.is_check():
            king_square = self.board.king(self.board.turn)
            if square == king_square:
                color = COLORS["check"]

        self.create_rectangle(
            x,
            y,
            x + self.square_size,
            y + self.square_size,
            fill=color,
            outline="",
        )

        # Draw legal move indicators
        for move in self.legal_moves:
            if move.to_square == square:
                cx = x + self.square_size // 2
                cy = y + self.square_size // 2

                if self.board.piece_at(square):
                    # Capture indicator (ring)
                    r = self.square_size // 2 - 4
                    self.create_oval(
                        cx - r,
                        cy - r,
                        cx + r,
                        cy + r,
                        outline=COLORS["legal_move"],
                        width=4,
                    )
                else:
                    # Move indicator (dot)
                    r = self.square_size // 6
                    self.create_oval(
                        cx - r,
                        cy - r,
                        cx + r,
                        cy + r,
                        fill=COLORS["legal_move"],
                        outline="",
                    )

        # Draw drag hover indicator (inner border)
        if self.dragging and square == self._drag_hover_square:
            border_width = 4
            hover_color = COLORS["hover_light"] if is_light else COLORS["hover_dark"]
            self.create_rectangle(
                x + border_width / 2,
                y + border_width / 2,
                x + self.square_size - border_width / 2,
                y + self.square_size - border_width / 2,
                fill="",
                outline=hover_color,
                width=border_width,
            )

    def _load_piece_images(self) -> None:
        """Load and resize piece images from assets directory."""
        for symbol, filename in _SYMBOL_TO_FILE.items():
            file_path = _ASSETS_DIR / f"{filename}.png"
            if file_path.exists():
                img = Image.open(file_path).convert("RGBA")
                self._piece_images_raw[symbol] = img  # Keep raw for scaling
                img_resized = img.resize(
                    (self.square_size, self.square_size), Image.Resampling.LANCZOS
                )
                self._piece_images[symbol] = ImageTk.PhotoImage(img_resized)

    def _get_piece_image(self, symbol: str) -> Any:
        """Get piece image."""
        return self._piece_images.get(symbol)

    def _get_scaled_drag_image(self, symbol: str, scale: float) -> Any:
        """Get a scaled piece image for dragging."""
        if symbol not in self._piece_images_raw:
            return None
        size = int(self.square_size * scale)
        img = self._piece_images_raw[symbol].resize(
            (size, size), Image.Resampling.LANCZOS
        )
        self._drag_scaled_image = ImageTk.PhotoImage(img)
        return self._drag_scaled_image

    def _start_drag_scale_animation(
        self, target: float, on_complete: callable = None
    ) -> None:
        """Start a scale animation to target value."""
        import time

        if self._drag_scale_anim_id:
            self.after_cancel(self._drag_scale_anim_id)
            self._drag_scale_anim_id = None
        self._drag_scale_start = self._drag_scale
        self._drag_scale_target = target
        self._drag_scale_start_time = time.time()
        self._drag_scale_on_complete = on_complete
        self._animate_drag_scale()

    def _animate_drag_scale(self) -> None:
        """Animate the drag scale towards target (0.2s duration)."""
        import time

        if self._drag_scale_start_time is None:
            return

        elapsed = time.time() - self._drag_scale_start_time
        progress = min(elapsed / self._drag_scale_duration, 1.0)

        # Ease-out cubic for smooth deceleration
        eased = 1 - (1 - progress) ** 3

        self._drag_scale = (
            self._drag_scale_start
            + (self._drag_scale_target - self._drag_scale_start) * eased
        )
        self._draw_board()

        if progress < 1.0:
            self._drag_scale_anim_id = self.after(16, self._animate_drag_scale)
        else:
            self._drag_scale = self._drag_scale_target
            self._drag_scale_anim_id = None
            self._drag_scale_start_time = None
            if self._drag_scale_on_complete:
                callback = self._drag_scale_on_complete
                self._drag_scale_on_complete = None
                callback()

    def _start_drag_pos_animation(
        self, target: tuple[int, int], on_complete: callable = None
    ) -> None:
        """Start a position animation for the dragged piece."""
        import time

        if self._drag_pos_anim_id:
            self.after_cancel(self._drag_pos_anim_id)
            self._drag_pos_anim_id = None
        self._drag_pos_start = self.drag_pos
        self._drag_pos_target = target
        self._drag_pos_start_time = time.time()
        self._drag_pos_on_complete = on_complete
        self._animate_drag_pos()

    def _animate_drag_pos(self) -> None:
        """Animate the drag position towards target."""
        import time

        if self._drag_pos_start_time is None or self._drag_pos_start is None:
            return

        elapsed = time.time() - self._drag_pos_start_time
        progress = min(elapsed / self._drag_pos_duration, 1.0)

        # Ease-out cubic for smooth deceleration
        eased = 1 - (1 - progress) ** 3

        start_x, start_y = self._drag_pos_start
        target_x, target_y = self._drag_pos_target
        current_x = int(start_x + (target_x - start_x) * eased)
        current_y = int(start_y + (target_y - start_y) * eased)
        self.drag_pos = (current_x, current_y)
        self._draw_board()

        if progress < 1.0:
            self._drag_pos_anim_id = self.after(16, self._animate_drag_pos)
        else:
            self.drag_pos = self._drag_pos_target
            self._drag_pos_anim_id = None
            self._drag_pos_start_time = None
            self._drag_pos_start = None
            self._drag_pos_target = None
            if self._drag_pos_on_complete:
                callback = self._drag_pos_on_complete
                self._drag_pos_on_complete = None
                callback()

    def _draw_piece(self, square: int) -> None:
        """Draw a piece on a square."""
        piece = self.board.piece_at(square)
        if piece is None:
            return

        x, y = self._square_to_coords(square)
        cx = x + self.square_size // 2
        cy = y + self.square_size // 2

        if self._use_images:
            # Use downloaded PNG images
            image = self._get_piece_image(piece.symbol())
            if image:
                self.create_image(cx, cy, image=image, anchor="center")
            else:
                self._draw_piece_unicode(cx, cy, piece)
        else:
            # Fallback to Unicode
            self._draw_piece_unicode(cx, cy, piece)

    def _draw_piece_unicode(self, cx: int, cy: int, piece: chess.Piece) -> None:
        """Draw piece using Unicode characters (fallback)."""
        symbol = piece.symbol()
        unicode_char = PIECE_UNICODE.get(symbol, "?")

        # Use different colors for pieces for better visibility
        if piece.color == chess.WHITE:
            fill_color = "#ffffff"
            outline_color = "#000000"
        else:
            fill_color = "#000000"
            outline_color = "#ffffff"

        # Draw piece with outline for visibility
        font = ("Segoe UI Symbol", self.square_size // 2)

        # Shadow/outline effect
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            self.create_text(
                cx + dx,
                cy + dy,
                text=unicode_char,
                font=font,
                fill=outline_color,
            )

        self.create_text(
            cx,
            cy,
            text=unicode_char,
            font=font,
            fill=fill_color,
        )

    def _draw_dragged_piece(self) -> None:
        """Draw the piece being dragged."""
        if not self.drag_piece or not self.drag_pos:
            return

        x, y = self.drag_pos

        if self._use_images:
            # Use scaled PNG image for drag effect
            image = self._get_scaled_drag_image(self.drag_piece, self._drag_scale)
            if image:
                self.create_image(x, y, image=image, anchor="center")
                return

        # Fallback to Unicode
        unicode_char = PIECE_UNICODE.get(self.drag_piece, "?")

        is_white = self.drag_piece.isupper()
        fill_color = "#ffffff" if is_white else "#000000"
        outline_color = "#000000" if is_white else "#ffffff"

        font_size = int(self.square_size // 2 * self._drag_scale)
        font = ("Segoe UI Symbol", font_size)

        # Shadow/outline
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            self.create_text(
                x + dx,
                y + dy,
                text=unicode_char,
                font=font,
                fill=outline_color,
            )

        self.create_text(
            x,
            y,
            text=unicode_char,
            font=font,
            fill=fill_color,
        )

    def _draw_coordinates(self) -> None:
        """Draw file and rank labels with improved visibility."""
        bold_font = (FONTS["coordinates"][0], FONTS["coordinates"][1], "bold")

        for i in range(8):
            # File labels (a-h) - bottom row
            file_label = chr(ord("a") + (7 - i if self.flipped else i))
            x = i * self.square_size + self.square_size - 3
            y = self.size - 1

            # Use high contrast colors for better readability
            is_light = (i + 0) % 2 == 1
            if is_light:
                # Light square: use dark text with light outline
                text_color = "#739552"  # Darker green for contrast
            else:
                # Dark square: use light text
                text_color = "#ebecd0"  # Light cream for contrast

            # Draw text (shadow effect removed for Tkinter compatibility)
            self.create_text(
                x, y, text=file_label, font=bold_font, fill=text_color, anchor="se"
            )

            # Rank labels (1-8) - left column
            rank_label = str(i + 1 if self.flipped else 8 - i)
            x = 2
            y = i * self.square_size

            is_light = (0 + (7 - i)) % 2 == 1
            if is_light:
                text_color = "#739552"
            else:
                text_color = "#ebecd0"

            # Draw text (shadow effect removed for Tkinter compatibility)
            self.create_text(
                x, y, text=rank_label, font=bold_font, fill=text_color, anchor="nw"
            )

    @property
    def is_animating(self) -> bool:
        """Check if an animation is in progress."""
        return self._animating

    def _on_click(self, event: tk.Event) -> None:
        """Handle mouse click."""
        if not self.interactive or self._animating:
            return

        # Check if it's the player's turn
        if self.player_color is not None and self.board.turn != self.player_color:
            return

        square = self._coords_to_square(event.x, event.y)
        piece = self.board.piece_at(square)

        if self.selected_square is None:
            # First click - select piece
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.legal_moves = {
                    m for m in self.board.legal_moves if m.from_square == square
                }

                # Start potential drag
                self.drag_from = square
                self.drag_piece = piece.symbol()
                self.drag_pos = (event.x, event.y)

                self._draw_board()
        else:
            # Second click - try to make move
            move = self._find_move(self.selected_square, square)

            if move:
                self._make_move(move)
            elif piece and piece.color == self.board.turn:
                # Select different piece
                self.selected_square = square
                self.legal_moves = {
                    m for m in self.board.legal_moves if m.from_square == square
                }
                self.drag_from = square
                self.drag_piece = piece.symbol()
                self.drag_pos = (event.x, event.y)
            else:
                # Deselect
                self.selected_square = None
                self.legal_moves = set()
                self.drag_from = None
                self.drag_piece = None

            self._draw_board()

    def _on_drag(self, event: tk.Event) -> None:
        """Handle mouse drag."""
        if not self.interactive or self._animating or self.drag_piece is None:
            return

        # Start scale-up animation when drag begins
        if not self.dragging:
            self._start_drag_scale_animation(2)

        self.dragging = True
        self.drag_pos = (event.x, event.y)
        # Track hover square for visual feedback
        self._drag_hover_square = self._coords_to_square(event.x, event.y)
        self._draw_board()

    def _on_release(self, event: tk.Event) -> None:
        """Handle mouse release."""
        if not self.interactive or self._animating:
            return

        if not self.dragging or self.drag_from is None:
            self.dragging = False
            self.drag_pos = None
            self._drag_hover_square = None
            return

        target_square = self._coords_to_square(event.x, event.y)
        move = self._find_move(self.drag_from, target_square)
        self._drag_hover_square = None

        # Scale down callback (used after position animation if needed)
        def on_scale_complete():
            self.dragging = False
            self.drag_pos = None
            self.drag_piece = None
            self._drag_scale = 1.0
            self._shrink_at_square = None
            self._draw_board()

        if move:
            # Valid move: snap to destination, process move, shrink
            final_square = move.to_square
            fx, fy = self._square_to_coords(final_square)
            self.drag_pos = (fx + self.square_size // 2, fy + self.square_size // 2)
            self._shrink_at_square = final_square

            # Apply the move immediately
            self.selected_square = None
            self.legal_moves = set()
            self.drag_from = None
            self.last_move = move
            self.board.push(move)
            if self.on_move:
                self.on_move(move)

            self._draw_board()
            self._start_drag_scale_animation(1.0, on_scale_complete)

        elif target_square != self.drag_from:
            # Invalid move: animate back AND shrink simultaneously
            original_square = self.drag_from
            ox, oy = self._square_to_coords(original_square)
            target_pos = (ox + self.square_size // 2, oy + self.square_size // 2)

            # Clear selection state
            self.selected_square = None
            self.legal_moves = set()
            self.drag_from = None
            self._shrink_at_square = original_square

            # Track completion of both animations
            anim_complete = {"pos": False, "scale": False}

            def check_both_complete():
                if anim_complete["pos"] and anim_complete["scale"]:
                    self.dragging = False
                    self.drag_pos = None
                    self.drag_piece = None
                    self._drag_scale = 1.0
                    self._shrink_at_square = None
                    self._draw_board()

            def on_pos_done():
                anim_complete["pos"] = True
                check_both_complete()

            def on_scale_done():
                anim_complete["scale"] = True
                check_both_complete()

            # Start both animations simultaneously
            self._start_drag_pos_animation(target_pos, on_pos_done)
            self._start_drag_scale_animation(1.0, on_scale_done)

        else:
            # Same square: just shrink in place
            self._shrink_at_square = self.drag_from
            self.selected_square = None
            self.legal_moves = set()
            self.drag_from = None
            self._draw_board()
            self._start_drag_scale_animation(1.0, on_scale_complete)

    def _find_move(self, from_sq: int, to_sq: int) -> chess.Move | None:
        """Find a legal move from the given squares."""
        # Check for promotion
        piece = self.board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_sq)
            if (piece.color == chess.WHITE and to_rank == 7) or (
                piece.color == chess.BLACK and to_rank == 0
            ):
                # Check if any promotion move is legal
                test_move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                if test_move in self.board.legal_moves:
                    # Show promotion dialog
                    promotion_piece = self._show_promotion_dialog(piece.color, to_sq)
                    move = chess.Move(from_sq, to_sq, promotion=promotion_piece)
                    if move in self.board.legal_moves:
                        return move

        # Regular move
        move = chess.Move(from_sq, to_sq)
        if move in self.board.legal_moves:
            return move

        return None

    def _show_promotion_dialog(self, color: bool, to_square: int) -> int:
        """Show promotion dialog and return selected piece type."""
        # Calculate dialog position near the promotion square
        x, y = self._square_to_coords(to_square)
        screen_x = self.winfo_rootx() + x
        screen_y = self.winfo_rooty() + y

        # Adjust position to not go off screen
        if color == chess.WHITE:
            screen_y -= 80  # Show above the square for white
        else:
            screen_y += self.square_size + 10  # Show below for black

        dialog = PromotionDialog(self, color, (screen_x, screen_y))
        return dialog.show()

    def _make_move(self, move: chess.Move, animate: bool = True) -> None:
        """Make a move and trigger callback."""
        # Skip animation if piece was dragged (user already "animated" it manually)
        should_animate = animate and not self.dragging

        # Clear selection state
        self.selected_square = None
        self.legal_moves = set()
        self.drag_from = None
        self.drag_piece = None

        if should_animate:
            # Animate the move, then apply it
            def on_animation_complete():
                self.last_move = move
                self.board.push(move)
                self._draw_board()
                if self.on_move:
                    self.on_move(move)

            self.animate_move(move, duration_ms=200, on_complete=on_animation_complete)
        else:
            # Instant move (no animation for drag-and-drop)
            self.last_move = move
            self.board.push(move)
            self._draw_board()
            if self.on_move:
                self.on_move(move)

    def _make_move_after_drag(self, move: chess.Move) -> None:
        """Make a move after drag-and-drop (no animation needed)."""
        self.selected_square = None
        self.legal_moves = set()
        self.drag_from = None
        self.drag_piece = None
        self.last_move = move
        self.board.push(move)
        self._draw_board()
        if self.on_move:
            self.on_move(move)

    def get_board(self) -> chess.Board:
        """Get the current board state."""
        return self.board.copy()

    def undo_move(self) -> bool:
        """Undo the last move."""
        if self.board.move_stack:
            self.board.pop()
            if self.board.move_stack:
                self.last_move = self.board.peek()
            else:
                self.last_move = None
            self.selected_square = None
            self.legal_moves = set()
            self._draw_board()
            return True
        return False

    def reset(self) -> None:
        """Reset to starting position."""
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = set()
        self.last_move = None
        self._draw_board()

    def animate_move(
        self,
        move: chess.Move,
        duration_ms: int = 300,
        on_complete: Callable[[], None] | None = None,
        reverse: bool = False,
    ) -> None:
        """
        Animate a piece moving from one square to another.

        Args:
            move: The move to animate.
            duration_ms: Animation duration in milliseconds.
            on_complete: Callback when animation completes.
            reverse: If True, animate the move in reverse (for undo).
        """
        # Cancel any ongoing animation
        if self._animation_id is not None:
            self.after_cancel(self._animation_id)
            self._animation_id = None

        # Determine from/to squares based on direction
        if reverse:
            from_sq, to_sq = move.to_square, move.from_square
        else:
            from_sq, to_sq = move.from_square, move.to_square

        # Get the piece to animate
        piece = self.board.piece_at(from_sq)
        if piece is None:
            # No piece to animate, just call callback
            if on_complete:
                on_complete()
            return

        self._animating = True

        # Set last_move NOW so squares are highlighted during animation
        # Store the original move (not reversed) for highlighting
        self._anim_highlight_move = move
        self.last_move = move

        # Calculate start and end positions (center of squares)
        start_x, start_y = self._square_to_coords(from_sq)
        end_x, end_y = self._square_to_coords(to_sq)

        # Center positions
        start_x += self.square_size // 2
        start_y += self.square_size // 2
        end_x += self.square_size // 2
        end_y += self.square_size // 2

        # Store animation state for main piece (king for castling)
        self._anim_piece_symbol = piece.symbol()
        self._anim_start = (start_x, start_y)
        self._anim_end = (end_x, end_y)
        self._anim_duration = duration_ms
        self._anim_start_time = None
        self._anim_on_complete = on_complete
        self._anim_from_square = from_sq

        # Check for castling - need to animate rook too
        self._anim_rook = None
        is_castling = (not reverse and self.board.is_castling(move)) or (
            reverse
            and piece.piece_type == chess.KING
            and abs(
                chess.square_file(move.from_square) - chess.square_file(move.to_square)
            )
            == 2
        )

        if is_castling:
            # Determine rook's from and to squares
            if chess.square_file(move.to_square) == 6:  # Kingside (g-file)
                rook_from = chess.square(
                    7, chess.square_rank(move.from_square)
                )  # h-file
                rook_to = chess.square(5, chess.square_rank(move.from_square))  # f-file
            else:  # Queenside (c-file)
                rook_from = chess.square(
                    0, chess.square_rank(move.from_square)
                )  # a-file
                rook_to = chess.square(3, chess.square_rank(move.from_square))  # d-file

            # Swap for reverse animation
            if reverse:
                rook_from, rook_to = rook_to, rook_from

            rook = self.board.piece_at(rook_from)
            if rook:
                rook_start_x, rook_start_y = self._square_to_coords(rook_from)
                rook_end_x, rook_end_y = self._square_to_coords(rook_to)

                self._anim_rook = {
                    "symbol": rook.symbol(),
                    "start": (
                        rook_start_x + self.square_size // 2,
                        rook_start_y + self.square_size // 2,
                    ),
                    "end": (
                        rook_end_x + self.square_size // 2,
                        rook_end_y + self.square_size // 2,
                    ),
                    "from_square": rook_from,
                }

        # Start animation loop
        self._animate_step()

    def _animate_step(self) -> None:
        """Perform one step of the animation."""
        import time

        if self._anim_start_time is None:
            self._anim_start_time = time.time() * 1000  # ms

        elapsed = time.time() * 1000 - self._anim_start_time
        progress = min(elapsed / self._anim_duration, 1.0)

        # Ease-out cubic for smooth deceleration
        eased = 1 - (1 - progress) ** 3

        # Calculate current position for main piece (king)
        start_x, start_y = self._anim_start
        end_x, end_y = self._anim_end
        current_x = start_x + (end_x - start_x) * eased
        current_y = start_y + (end_y - start_y) * eased

        # Collect squares to exclude (for castling, exclude both king and rook)
        exclude_squares = [self._anim_from_square]
        if self._anim_rook:
            exclude_squares.append(self._anim_rook["from_square"])

        # Redraw board without the animated pieces
        self._draw_board_without_squares(exclude_squares)

        # Draw the animated king at current position
        self._draw_animated_piece(self._anim_piece_symbol, current_x, current_y)

        # Draw the animated rook if castling
        if self._anim_rook:
            rook_start_x, rook_start_y = self._anim_rook["start"]
            rook_end_x, rook_end_y = self._anim_rook["end"]
            rook_current_x = rook_start_x + (rook_end_x - rook_start_x) * eased
            rook_current_y = rook_start_y + (rook_end_y - rook_start_y) * eased
            self._draw_animated_piece(
                self._anim_rook["symbol"], rook_current_x, rook_current_y
            )

        if progress < 1.0:
            # Continue animation (~60 FPS)
            self._animation_id = self.after(16, self._animate_step)
        else:
            # Animation complete
            self._animating = False
            self._animation_id = None
            self._anim_rook = None
            self.delete("animated_piece")

            if self._anim_on_complete:
                self._anim_on_complete()

    def _draw_animated_piece(self, symbol: str, x: float, y: float) -> None:
        """Draw a piece at the given position during animation."""
        if self._use_images:
            image = self._get_piece_image(symbol)
            if image:
                self.create_image(
                    x,
                    y,
                    image=image,
                    anchor="center",
                    tags="animated_piece",
                )
                return

        # Unicode fallback
        unicode_char = PIECE_UNICODE.get(symbol, "?")
        is_white = symbol.isupper()
        fill_color = "#ffffff" if is_white else "#000000"
        outline_color = "#000000" if is_white else "#ffffff"
        font = ("Segoe UI Symbol", self.square_size // 2)

        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            self.create_text(
                x + dx,
                y + dy,
                text=unicode_char,
                font=font,
                fill=outline_color,
                tags="animated_piece",
            )
        self.create_text(
            x,
            y,
            text=unicode_char,
            font=font,
            fill=fill_color,
            tags="animated_piece",
        )

    def _draw_board_without_squares(self, exclude_squares: list[int]) -> None:
        """Redraw the board but skip drawing pieces at excluded squares."""
        self.delete("all")

        # Draw squares
        for square in chess.SQUARES:
            self._draw_square(square)

        # Draw coordinates
        self._draw_coordinates()

        # Draw pieces except those being animated
        for square in chess.SQUARES:
            if square not in exclude_squares:
                if not (self.dragging and square == self.drag_from):
                    self._draw_piece(square)

        # Draw dragged piece on top
        if self.dragging and self.drag_piece and self.drag_pos:
            self._draw_dragged_piece()
