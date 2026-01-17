"""
Main chess game application.

Provides the complete UI for playing chess against the AlphaZero AI,
watching AI vs AI games, and managing training.

Professional chess.com-inspired design with AI debugging features.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Any, Optional, List
import threading
import queue
import os
import json
import time
import math

import numpy as np

try:
    import chess
except ImportError:
    raise ImportError(
        "python-chess is required. Install with: pip install python-chess"
    )

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .styles import (
    COLORS,
    FONTS,
    apply_theme,
    create_styled_button,
    create_styled_label,
    create_panel,
    create_tooltip,
    create_confirmation_dialog,
    ThinkingIndicator,
)
from .board_widget import ChessBoardWidget
from .training_panel import TrainingPanel
from .components import (
    EvalBar,
    EvalGraph,
    MCTSPanel,
    MoveList,
    PlayerInfo,
    OpeningDisplay,
    PhaseIndicator,
    SearchTreePanel,
)
from src.chess_encoding.board_utils import get_material_count


class ChessGameApp:
    """
    Main application window for playing chess against AI.

    Features:
    - Human vs AI gameplay
    - Human vs Human gameplay
    - AI vs AI spectating
    - Training panel for self-play
    - Game controls (new game, undo, flip board)
    - AI debugging: MCTS stats, eval bar, eval graph, network output
    """

    # Default MCTS settings for play mode (optimized for quality over speed)
    DEFAULT_MCTS_BATCH_SIZE = 8
    DEFAULT_HISTORY_LENGTH = 3

    # Dynamic time management constants
    MIN_SIMULATIONS = 100  # Minimum simulations floor (even in time pressure)
    HISTORY_WINDOW = 10  # Number of moves for time-per-sim calculation
    DECAY_FACTOR = 0.85  # Exponential decay for older moves
    EMERGENCY_THRESHOLD = 15  # Seconds - triggers emergency mode
    INSTANT_MOVE_THRESHOLD = 3  # Seconds - play instantly from tree, no search

    def __init__(
        self,
        network_path: str | None = None,
        num_simulations: int = 200,
        mcts_batch_size: int = DEFAULT_MCTS_BATCH_SIZE,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        verbose: bool = False,
        time_control: int | None = None,
    ):
        """
        Initialize the chess game application.

        Args:
            network_path: Path to pre-trained network file
            num_simulations: MCTS simulations for AI moves
            mcts_batch_size: Batch size for MCTS GPU inference (default: 8)
            history_length: Number of past positions for encoding (default: 3)
            verbose: If True, print MCTS search tree after each AI move
            time_control: Time in minutes per player (None = no time limit)
        """
        self.root = tk.Tk()
        self.root.title("NeuralMate Chess - AlphaZero")
        self.root.resizable(True, True)
        self.root.geometry("1700x950")  # Wider for new layout
        self.root.minsize(1400, 850)

        apply_theme(self.root)

        # Network and AI
        self.network = None
        self.mcts = None
        self.pure_mcts_player = None
        self.network_path = network_path
        self.num_simulations = num_simulations
        self.mcts_batch_size = mcts_batch_size
        self.history_length = history_length
        self.verbose = verbose
        self.use_pure_mcts = network_path and network_path.lower() == "mcts"

        # Load c_puct from config.json
        self.c_puct = 1.5  # default
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config",
            "config.json",
        )
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self.c_puct = config.get("training", {}).get("c_puct", 1.5)
            except Exception:
                pass  # Use default on error

        # Game state
        self.game_mode = "human_vs_ai"
        self.human_color = chess.WHITE
        self.ai_thinking = False
        self.ai_thread: threading.Thread | None = None
        self.ai_cancelled = False

        # Move history for back/forward navigation
        self.undone_moves: list = []

        # Full game history for MCTS (all positions from start, not limited to 3)
        # This is needed because MCTS takes _initial_history[:3] (first 3 positions)
        self.game_positions: list = []  # [pos_initial, pos_after_move1, ...]

        # Evaluation history for graph
        self.eval_history: List[float] = []
        self.wdl_history: List[Optional[np.ndarray]] = []  # WDL for each eval

        # Move timing
        self.turn_start_time: float = time.time()
        self.last_move_time: float = 0.0

        # Time control
        self.time_control: int | None = (
            time_control * 60 if time_control else None
        )  # Convert minutes to seconds
        self.white_time: float = 0.0
        self.black_time: float = 0.0
        self.base_simulations: int = num_simulations  # Store original for scaling
        self.timer_update_id: str | None = None
        self.game_started: bool = False  # Timer only runs after first move

        # Dynamic time management stats
        self.ai_move_times: List[float] = []  # Time taken for each AI move
        self.ai_sims_used: List[int] = []  # Simulations used for each AI move
        self.last_wdl: tuple | None = None  # Last WDL evaluation (win, draw, loss)

        # Update queue for thread-safe UI updates
        self.update_queue = queue.Queue()

        # Training state
        self.trainer = None
        self.training_thread: threading.Thread | None = None
        self.training_cancelled = False

        # Editor mode state
        self.app_mode = "play"  # "play" or "editor"
        self.editor_selected_piece: chess.Piece | None = None
        self.editor_palette_buttons: dict = {}

        self._create_ui()
        self._bind_keyboard_shortcuts()
        self._load_network()
        self._start_update_loop()

        # Initialize time control if enabled
        self._init_time_control()

    def _bind_keyboard_shortcuts(self) -> None:
        """Bind keyboard shortcuts for common actions."""
        self.root.bind("<Control-n>", lambda e: self._new_game_with_confirm())
        self.root.bind("<Control-N>", lambda e: self._new_game_with_confirm())
        self.root.bind("<Control-z>", lambda e: self._back_one_move())
        self.root.bind("<Control-Z>", lambda e: self._back_one_move())
        self.root.bind("<Left>", lambda e: self._back_one_move())
        self.root.bind("<Right>", lambda e: self._forward_one_move())
        self.root.bind("<f>", lambda e: self._flip_board())
        self.root.bind("<F>", lambda e: self._flip_board())
        self.root.bind("<Escape>", lambda e: self._cancel_selection())

    def _create_ui(self) -> None:
        """Create the main UI layout - chess.com inspired design."""
        # Main container
        main_frame = tk.Frame(self.root, bg=COLORS["bg_primary"])
        main_frame.pack(padx=20, pady=15, fill="both", expand=True)

        # Top bar with controls
        self._create_top_bar(main_frame)

        # Content area - 3 column layout
        content_frame = tk.Frame(main_frame, bg=COLORS["bg_primary"])
        content_frame.pack(fill="both", expand=True, pady=(10, 0))

        # Left panel - Player info, opening, phase
        self._create_left_panel(content_frame)

        # Center panel - Board + eval bar
        self._create_center_panel(content_frame)

        # Right panel - Analysis (MCTS, eval graph, network output)
        self._create_right_panel(content_frame)

    def _create_top_bar(self, parent: tk.Widget) -> None:
        """Create the top bar with game controls."""
        top_frame = tk.Frame(parent, bg=COLORS["bg_primary"])
        top_frame.pack(fill="x")

        # Left side - Game control buttons
        left_btns = tk.Frame(top_frame, bg=COLORS["bg_primary"])
        left_btns.pack(side="left")

        self.new_game_btn = create_styled_button(
            left_btns,
            "New Game",
            command=self._new_game,
            style="accent",
        )
        self.new_game_btn.pack(side="left", padx=(0, 10))

        self.back_btn = create_styled_button(
            left_btns,
            "◀",
            command=self._back_one_move,
            style="outline",
        )
        self.back_btn.pack(side="left", padx=(0, 3))

        self.forward_btn = create_styled_button(
            left_btns,
            "▶",
            command=self._forward_one_move,
            style="outline",
        )
        self.forward_btn.pack(side="left", padx=(0, 10))

        self.flip_btn = create_styled_button(
            left_btns,
            "Flip",
            command=self._flip_board,
            style="outline",
        )
        self.flip_btn.pack(side="left", padx=(0, 10))

        self.load_btn = create_styled_button(
            left_btns,
            "Load Network",
            command=self._load_network_dialog,
            style="outline",
        )
        self.load_btn.pack(side="left")

        # Mode toggle (Play / Editor)
        mode_toggle_frame = tk.Frame(left_btns, bg=COLORS["bg_primary"])
        mode_toggle_frame.pack(side="left", padx=(20, 0))

        tk.Label(
            mode_toggle_frame,
            text="Mode:",
            bg=COLORS["bg_primary"],
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(side="left", padx=(0, 5))

        self.app_mode_var = tk.StringVar(value="play")

        self.play_mode_btn = tk.Radiobutton(
            mode_toggle_frame,
            text="Play",
            variable=self.app_mode_var,
            value="play",
            command=self._on_app_mode_change,
            bg=COLORS["bg_primary"],
            fg=COLORS["text_primary"],
            selectcolor=COLORS["bg_tertiary"],
            activebackground=COLORS["bg_primary"],
            activeforeground=COLORS["accent"],
            font=FONTS["body_bold"],
            indicatoron=0,
            padx=10,
            pady=3,
        )
        self.play_mode_btn.pack(side="left")

        self.editor_mode_btn = tk.Radiobutton(
            mode_toggle_frame,
            text="Editor",
            variable=self.app_mode_var,
            value="editor",
            command=self._on_app_mode_change,
            bg=COLORS["bg_primary"],
            fg=COLORS["text_primary"],
            selectcolor=COLORS["bg_tertiary"],
            activebackground=COLORS["bg_primary"],
            activeforeground=COLORS["accent"],
            font=FONTS["body_bold"],
            indicatoron=0,
            padx=10,
            pady=3,
        )
        self.editor_mode_btn.pack(side="left")

        # Tooltips
        create_tooltip(self.new_game_btn, "Start a new game (Ctrl+N)")
        create_tooltip(self.back_btn, "Go back one move (Ctrl+Z / ←)")
        create_tooltip(self.forward_btn, "Go forward (→)")
        create_tooltip(self.flip_btn, "Flip board (F)")
        create_tooltip(self.load_btn, "Load neural network")

        # Right side - Status and network info
        right_frame = tk.Frame(top_frame, bg=COLORS["bg_primary"])
        right_frame.pack(side="right")

        self.network_status = tk.Label(
            right_frame,
            text="No network loaded",
            bg=COLORS["bg_primary"],
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        )
        self.network_status.pack(side="right")

        self.status_label = tk.Label(
            right_frame,
            text="Your turn",
            bg=COLORS["bg_primary"],
            fg=COLORS["success"],
            font=FONTS["body_bold"],
        )
        self.status_label.pack(side="right", padx=(0, 20))

    def _create_left_panel(self, parent: tk.Widget) -> None:
        """Create the left panel with player info and game info."""
        left_panel = tk.Frame(parent, bg=COLORS["bg_primary"], width=200)
        left_panel.pack(side="left", fill="y", padx=(0, 15))
        left_panel.pack_propagate(False)

        # Play mode content
        self.play_left_content = tk.Frame(left_panel, bg=COLORS["bg_primary"])
        self.play_left_content.pack(fill="both", expand=True)

        # Black player info (top, opponent when playing white)
        self.black_player_info = PlayerInfo(
            self.play_left_content,
            color=chess.BLACK,
            name="AI",
        )
        self.black_player_info.pack(fill="x", pady=(0, 10))

        # Opening display
        self.opening_display = OpeningDisplay(self.play_left_content)
        self.opening_display.pack(fill="x", pady=(0, 10))

        # Phase indicator
        self.phase_indicator = PhaseIndicator(self.play_left_content)
        self.phase_indicator.pack(fill="x", pady=(0, 10))

        # Spacer
        spacer = tk.Frame(self.play_left_content, bg=COLORS["bg_primary"])
        spacer.pack(fill="both", expand=True)

        # White player info (bottom, you when playing white)
        self.white_player_info = PlayerInfo(
            self.play_left_content,
            color=chess.WHITE,
            name="You",
        )
        self.white_player_info.pack(fill="x", pady=(10, 0))

        # Game mode selection (collapsed panel)
        self._create_mode_panel(self.play_left_content)

        # Editor mode content (initially hidden)
        self.editor_left_content = tk.Frame(left_panel, bg=COLORS["bg_primary"])
        self._create_editor_panel(self.editor_left_content)

    def _create_mode_panel(self, parent: tk.Widget) -> None:
        """Create the game mode selection panel."""
        mode_panel = create_panel(parent, title="Game Mode")
        mode_panel.pack(fill="x", pady=(10, 0))

        mode_content = tk.Frame(mode_panel, bg=COLORS["bg_secondary"])
        mode_content.pack(fill="x", padx=10, pady=(0, 10))

        self.mode_var = tk.StringVar(value="human_vs_ai")

        modes = [
            ("Human vs AI", "human_vs_ai"),
            ("AI vs AI", "ai_vs_ai"),
            ("Human vs Human", "human_vs_human"),
        ]

        for text, value in modes:
            rb = tk.Radiobutton(
                mode_content,
                text=text,
                variable=self.mode_var,
                value=value,
                command=self._on_mode_change,
                bg=COLORS["bg_secondary"],
                fg=COLORS["text_primary"],
                selectcolor=COLORS["bg_tertiary"],
                activebackground=COLORS["bg_secondary"],
                activeforeground=COLORS["text_primary"],
                font=FONTS["small"],
            )
            rb.pack(anchor="w")

        # Color selection
        self.color_frame = tk.Frame(mode_content, bg=COLORS["bg_secondary"])
        self.color_frame.pack(fill="x", pady=(5, 0))

        tk.Label(
            self.color_frame,
            text="Play as:",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(side="left")

        self.color_var = tk.StringVar(value="white")

        for text, value in [("White", "white"), ("Black", "black")]:
            rb = tk.Radiobutton(
                self.color_frame,
                text=text,
                variable=self.color_var,
                value=value,
                command=self._on_color_change,
                bg=COLORS["bg_secondary"],
                fg=COLORS["text_primary"],
                selectcolor=COLORS["bg_tertiary"],
                font=FONTS["small"],
            )
            rb.pack(side="left", padx=(5, 0))

        # AI Settings panel
        self._create_ai_settings_panel(parent)

    def _create_ai_settings_panel(self, parent: tk.Widget) -> None:
        """Create the AI settings panel with simulations slider."""
        ai_panel = create_panel(parent, title="AI Settings")
        ai_panel.pack(fill="x", pady=(10, 0))

        ai_content = tk.Frame(ai_panel, bg=COLORS["bg_secondary"])
        ai_content.pack(fill="x", padx=10, pady=(0, 10))

        # Simulations label and value
        sim_header = tk.Frame(ai_content, bg=COLORS["bg_secondary"])
        sim_header.pack(fill="x")

        tk.Label(
            sim_header,
            text="Simulations:",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(side="left")

        self.sim_value_label = tk.Label(
            sim_header,
            text=str(self.num_simulations),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            font=("Consolas", 10, "bold"),
        )
        self.sim_value_label.pack(side="right")

        # Simulations slider
        self.sim_var = tk.IntVar(value=self.num_simulations)
        self.sim_slider = ttk.Scale(
            ai_content,
            from_=50,
            to=2000,
            orient="horizontal",
            variable=self.sim_var,
            command=self._on_simulations_change,
        )
        self.sim_slider.pack(fill="x", pady=(5, 0))

        # Preset buttons
        preset_frame = tk.Frame(ai_content, bg=COLORS["bg_secondary"])
        preset_frame.pack(fill="x", pady=(5, 0))

        presets = [100, 200, 400, 800, 1600]
        for preset in presets:
            btn = tk.Button(
                preset_frame,
                text=str(preset),
                command=lambda p=preset: self._set_simulations(p),
                bg=COLORS["bg_tertiary"],
                fg=COLORS["text_primary"],
                font=("Segoe UI", 8),
                relief="flat",
                width=4,
                cursor="hand2",
            )
            btn.pack(side="left", padx=(0, 3))

    def _on_simulations_change(self, value: str) -> None:
        """Handle simulations slider change."""
        new_sims = int(float(value))
        self.num_simulations = new_sims
        self.sim_value_label.configure(text=str(new_sims))

        # Update MCTS if it exists
        if self.mcts is not None:
            self.mcts.num_simulations = new_sims

    def _set_simulations(self, value: int) -> None:
        """Set simulations to a preset value."""
        self.sim_var.set(value)
        self._on_simulations_change(str(value))

    def _create_editor_panel(self, parent: tk.Widget) -> None:
        """Create the position editor panel with piece palette and controls."""
        from .styles import PIECE_UNICODE

        # Piece palette
        palette_panel = create_panel(parent, title="Pieces")
        palette_panel.pack(fill="x", pady=(0, 10))

        palette_content = tk.Frame(palette_panel, bg=COLORS["bg_secondary"])
        palette_content.pack(fill="x", padx=10, pady=(0, 10))

        # White pieces label
        tk.Label(
            palette_content,
            text="White",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(anchor="w")

        # White pieces grid (2 rows of 3)
        white_grid = tk.Frame(palette_content, bg=COLORS["bg_secondary"])
        white_grid.pack(pady=(0, 5))

        white_pieces = [
            (chess.KING, "K"), (chess.QUEEN, "Q"), (chess.ROOK, "R"),
            (chess.BISHOP, "B"), (chess.KNIGHT, "N"), (chess.PAWN, "P"),
        ]
        for i, (piece_type, symbol) in enumerate(white_pieces):
            btn = tk.Button(
                white_grid,
                text=PIECE_UNICODE.get(symbol, symbol),
                command=lambda pt=piece_type: self._select_editor_piece(chess.WHITE, pt),
                bg=COLORS["bg_tertiary"],
                fg=COLORS["text_primary"],
                font=("Segoe UI Symbol", 16),
                width=2,
                relief="flat",
                cursor="hand2",
            )
            btn.grid(row=i // 3, column=i % 3, padx=2, pady=2)
            self.editor_palette_buttons[f"w{symbol}"] = btn

        # Black pieces label
        tk.Label(
            palette_content,
            text="Black",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(anchor="w")

        # Black pieces grid (2 rows of 3)
        black_grid = tk.Frame(palette_content, bg=COLORS["bg_secondary"])
        black_grid.pack(pady=(0, 5))

        black_pieces = [
            (chess.KING, "k"), (chess.QUEEN, "q"), (chess.ROOK, "r"),
            (chess.BISHOP, "b"), (chess.KNIGHT, "n"), (chess.PAWN, "p"),
        ]
        for i, (piece_type, symbol) in enumerate(black_pieces):
            btn = tk.Button(
                black_grid,
                text=PIECE_UNICODE.get(symbol, symbol),
                command=lambda pt=piece_type: self._select_editor_piece(chess.BLACK, pt),
                bg=COLORS["bg_tertiary"],
                fg=COLORS["text_primary"],
                font=("Segoe UI Symbol", 16),
                width=2,
                relief="flat",
                cursor="hand2",
            )
            btn.grid(row=i // 3, column=i % 3, padx=2, pady=2)
            self.editor_palette_buttons[f"b{symbol.lower()}"] = btn

        # Eraser button
        eraser_frame = tk.Frame(palette_content, bg=COLORS["bg_secondary"])
        eraser_frame.pack(fill="x")

        self.eraser_btn = tk.Button(
            eraser_frame,
            text="🗑 Eraser (Right-click)",
            command=self._select_editor_eraser,
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            font=FONTS["small"],
            relief="flat",
            cursor="hand2",
        )
        self.eraser_btn.pack(fill="x")

        # Position controls panel
        controls_panel = create_panel(parent, title="Position")
        controls_panel.pack(fill="x", pady=(0, 10))

        controls_content = tk.Frame(controls_panel, bg=COLORS["bg_secondary"])
        controls_content.pack(fill="x", padx=10, pady=(0, 10))

        # Turn to move
        turn_frame = tk.Frame(controls_content, bg=COLORS["bg_secondary"])
        turn_frame.pack(fill="x", pady=(0, 5))

        tk.Label(
            turn_frame,
            text="Turn:",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(side="left")

        self.editor_turn_var = tk.StringVar(value="white")

        tk.Radiobutton(
            turn_frame,
            text="White",
            variable=self.editor_turn_var,
            value="white",
            command=self._on_editor_position_change,
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            selectcolor=COLORS["bg_tertiary"],
            font=FONTS["small"],
        ).pack(side="left", padx=(5, 0))

        tk.Radiobutton(
            turn_frame,
            text="Black",
            variable=self.editor_turn_var,
            value="black",
            command=self._on_editor_position_change,
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            selectcolor=COLORS["bg_tertiary"],
            font=FONTS["small"],
        ).pack(side="left", padx=(5, 0))

        # Castling rights
        castle_frame = tk.Frame(controls_content, bg=COLORS["bg_secondary"])
        castle_frame.pack(fill="x", pady=(0, 5))

        tk.Label(
            castle_frame,
            text="Castling:",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(anchor="w")

        castle_checks = tk.Frame(castle_frame, bg=COLORS["bg_secondary"])
        castle_checks.pack(fill="x")

        self.castle_wk_var = tk.BooleanVar(value=True)
        self.castle_wq_var = tk.BooleanVar(value=True)
        self.castle_bk_var = tk.BooleanVar(value=True)
        self.castle_bq_var = tk.BooleanVar(value=True)

        for text, var in [("W O-O", self.castle_wk_var), ("W O-O-O", self.castle_wq_var),
                          ("B O-O", self.castle_bk_var), ("B O-O-O", self.castle_bq_var)]:
            tk.Checkbutton(
                castle_checks,
                text=text,
                variable=var,
                command=self._on_editor_position_change,
                bg=COLORS["bg_secondary"],
                fg=COLORS["text_primary"],
                selectcolor=COLORS["bg_tertiary"],
                font=("Segoe UI", 8),
            ).pack(side="left")

        # FEN display
        fen_frame = tk.Frame(controls_content, bg=COLORS["bg_secondary"])
        fen_frame.pack(fill="x", pady=(5, 0))

        tk.Label(
            fen_frame,
            text="FEN:",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(anchor="w")

        self.editor_fen_var = tk.StringVar(value=chess.STARTING_FEN)
        self.editor_fen_entry = tk.Entry(
            fen_frame,
            textvariable=self.editor_fen_var,
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            font=("Consolas", 8),
            insertbackground=COLORS["text_primary"],
        )
        self.editor_fen_entry.pack(fill="x", pady=(2, 0))

        tk.Button(
            fen_frame,
            text="Load FEN",
            command=self._load_editor_fen,
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            font=FONTS["small"],
            relief="flat",
            cursor="hand2",
        ).pack(fill="x", pady=(3, 0))

        # Action buttons
        actions_panel = create_panel(parent, title="Actions")
        actions_panel.pack(fill="x", pady=(0, 10))

        actions_content = tk.Frame(actions_panel, bg=COLORS["bg_secondary"])
        actions_content.pack(fill="x", padx=10, pady=(0, 10))

        tk.Button(
            actions_content,
            text="Clear Board",
            command=self._editor_clear_board,
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            font=FONTS["small"],
            relief="flat",
            cursor="hand2",
        ).pack(fill="x", pady=(0, 3))

        tk.Button(
            actions_content,
            text="Reset to Start",
            command=self._editor_reset_position,
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            font=FONTS["small"],
            relief="flat",
            cursor="hand2",
        ).pack(fill="x", pady=(0, 3))

        # Start game button (prominent)
        self.start_game_btn = tk.Button(
            actions_content,
            text="▶ Start Game!",
            command=self._start_game_from_editor,
            bg=COLORS["accent"],
            fg=COLORS["text_primary"],
            font=FONTS["body_bold"],
            relief="flat",
            cursor="hand2",
            pady=8,
        )
        self.start_game_btn.pack(fill="x", pady=(5, 0))

    def _create_center_panel(self, parent: tk.Widget) -> None:
        """Create the center panel with board and eval bar."""
        center_panel = tk.Frame(parent, bg=COLORS["bg_primary"])
        center_panel.pack(side="left", fill="both")

        # Board container (board + eval bar side by side)
        board_container = tk.Frame(center_panel, bg=COLORS["bg_primary"])
        board_container.pack()

        # Chess board
        self.board_widget = ChessBoardWidget(
            board_container,
            size=680,
            on_move=self._on_human_move,
        )
        self.board_widget.pack(side="left")

        # Evaluation bar (right of board)
        self.eval_bar = EvalBar(
            board_container,
            height=680,
            width=35,
        )
        self.eval_bar.pack(side="left", padx=(8, 0))

        # Move list below board
        move_list_frame = tk.Frame(center_panel, bg=COLORS["bg_primary"])
        move_list_frame.pack(fill="x", pady=(10, 0))

        self.move_list = MoveList(
            move_list_frame,
            width=716,
            height=120,
        )
        self.move_list.pack()

    def _create_right_panel(self, parent: tk.Widget) -> None:
        """Create the right panel with analysis tools (no scroll, fills space)."""
        right_panel = tk.Frame(parent, bg=COLORS["bg_primary"], width=300)
        right_panel.pack(side="left", fill="both", expand=True, padx=(15, 0))

        # Title
        title_label = tk.Label(
            right_panel,
            text="Analysis",
            font=FONTS["heading"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_primary"],
        )
        title_label.pack(anchor="w", pady=(0, 5))

        # AI Thinking indicator
        self.thinking_frame = tk.Frame(right_panel, bg=COLORS["bg_primary"])
        self.thinking_frame.pack(fill="x")

        self.thinking_indicator = ThinkingIndicator(
            self.thinking_frame,
            text="AI thinking",
            bg=COLORS["bg_primary"],
        )
        self.thinking_indicator.pack()
        self.thinking_indicator.pack_forget()

        # MCTS Panel
        self.mcts_panel = MCTSPanel(right_panel, width=280, height=350)
        self.mcts_panel.pack(fill="x", pady=(0, 5))

        # Search Tree Panel
        self.search_tree_panel = SearchTreePanel(right_panel, width=280, height=350)
        self.search_tree_panel.pack(fill="x", pady=(0, 5))

        # Eval Graph
        self.eval_graph = EvalGraph(right_panel, width=280, height=100)
        self.eval_graph.pack(fill="x", pady=(0, 5))

        # Last move time display
        time_frame = tk.Frame(right_panel, bg=COLORS["bg_secondary"])
        time_frame.pack(fill="x", pady=(0, 5))

        # Value Prediction (simple label below graph)
        value_frame = tk.Frame(right_panel, bg=COLORS["bg_secondary"])
        value_frame.pack(fill="x")

        tk.Label(
            value_frame,
            text="Value:",
            font=("Segoe UI", 10),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_secondary"],
        ).pack(side="left", padx=(10, 5), pady=5)

        self.value_label = tk.Label(
            value_frame,
            text="0.00",
            font=("Consolas", 12, "bold"),
            fg=COLORS["text_primary"],
            bg=COLORS["bg_secondary"],
        )
        self.value_label.pack(side="left", pady=5)

        # Training panel - hidden, kept for compatibility
        self.training_panel = None

    def _load_network(self) -> None:
        """Load the neural network or pure MCTS player."""

        def get_history_length(network):
            planes = network.num_input_planes
            # 72 planes = (history_length + 1) * 12 + 24 (metadata + semantic + tactical)
            return (planes - 24) // 12 - 1

        if self.use_pure_mcts:
            try:
                from alphazero.arena import PureMCTSPlayer

                self.pure_mcts_player = PureMCTSPlayer(
                    num_simulations=self.num_simulations,
                    verbose=self.verbose,
                    name="PureMCTS",
                )
                self.network_status.configure(
                    text=f"Pure MCTS ({self.num_simulations} sims)",
                    fg=COLORS["accent"],
                )
                return
            except Exception as e:
                self.network_status.configure(
                    text=f"Error: {e}",
                    fg=COLORS["error"],
                )
                return

        if self.network_path and os.path.exists(self.network_path):
            try:
                from alphazero import DualHeadNetwork, MCTS

                self.network = DualHeadNetwork.load(self.network_path)
                self.history_length = get_history_length(self.network)
                self.mcts = MCTS(
                    network=self.network,
                    num_simulations=self.num_simulations,
                    batch_size=self.mcts_batch_size,
                    history_length=self.history_length,
                    c_puct=self.c_puct,
                    # WDL-aware settings
                    contempt=0.5,
                    uncertainty_weight=0.2,
                    draw_sibling_fpu=True,
                )
                # Initialize game history with starting position
                self.game_positions = [self.board_widget.get_board().copy()]
                self.network_status.configure(
                    text=f"Loaded: {os.path.basename(self.network_path)}",
                    fg=COLORS["success"],
                )
            except Exception as e:
                self.network_status.configure(
                    text=f"Error: {e}",
                    fg=COLORS["error"],
                )
        else:
            try:
                from alphazero import DualHeadNetwork, MCTS

                self.network = DualHeadNetwork()
                self.history_length = get_history_length(self.network)
                self.mcts = MCTS(
                    network=self.network,
                    num_simulations=self.num_simulations,
                    batch_size=self.mcts_batch_size,
                    history_length=self.history_length,
                    c_puct=self.c_puct,
                    # WDL-aware settings
                    contempt=0.5,
                    uncertainty_weight=0.2,
                    draw_sibling_fpu=True,
                )
                # Initialize game history with starting position
                self.game_positions = [self.board_widget.get_board().copy()]
                self.network_status.configure(
                    text="Untrained network (random)",
                    fg=COLORS["warning"],
                )
            except Exception as e:
                self.network_status.configure(
                    text=f"Error: {e}",
                    fg=COLORS["error"],
                )

    def _load_network_dialog(self) -> None:
        """Show dialog to load a network file."""
        filepath = filedialog.askopenfilename(
            title="Load Network",
            filetypes=[("Network files", "*.pt"), ("All files", "*.*")],
        )
        if filepath:
            self.network_path = filepath
            self._load_network()

    def _on_mode_change(self) -> None:
        """Handle game mode change."""
        self.game_mode = self.mode_var.get()

        if self.game_mode == "ai_vs_ai":
            self.color_frame.pack_forget()
            self.board_widget.set_interactive(False)
            self._update_player_names("AI (White)", "AI (Black)")
        elif self.game_mode == "human_vs_human":
            self.color_frame.pack_forget()
            self.board_widget.set_interactive(True)
            self._update_player_names("White", "Black")
        else:
            self.color_frame.pack(fill="x", pady=(5, 0))
            self.board_widget.set_interactive(True)
            self._update_player_names_for_human_ai()

        self._new_game()

    def _on_color_change(self) -> None:
        """Handle player color change."""
        self.human_color = (
            chess.WHITE if self.color_var.get() == "white" else chess.BLACK
        )
        self.board_widget.set_player_color(self.human_color)

        if self.human_color == chess.BLACK and not self.board_widget.flipped:
            self.board_widget.flip()
        elif self.human_color == chess.WHITE and self.board_widget.flipped:
            self.board_widget.flip()

        self._update_player_names_for_human_ai()
        self._new_game()

    def _update_player_names(self, white_name: str, black_name: str) -> None:
        """Update player info names."""
        self.white_player_info.set_name(white_name)
        self.black_player_info.set_name(black_name)

    def _update_player_names_for_human_ai(self) -> None:
        """Update player names for human vs AI mode."""
        if self.human_color == chess.WHITE:
            self._update_player_names("You", "AI")
        else:
            self._update_player_names("AI", "You")

    def _new_game_with_confirm(self) -> None:
        """Start a new game with confirmation."""
        board = self.board_widget.get_board()
        if len(board.move_stack) > 0 and not board.is_game_over():
            create_confirmation_dialog(
                self.root,
                "New Game",
                "Start new game? Current game will be lost.",
                on_confirm=self._new_game,
            )
        else:
            self._new_game()

    def _new_game(self) -> None:
        """Start a new game."""
        # Cancel any existing timer update loop
        if self.timer_update_id:
            self.root.after_cancel(self.timer_update_id)
            self.timer_update_id = None

        self.ai_cancelled = True
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=0.5)

        self.ai_cancelled = False
        self.ai_thinking = False

        self.thinking_indicator.stop()
        self.thinking_indicator.pack_forget()

        # Clear MCTS tree for new game
        if self.mcts is not None:
            self.mcts.clear_cache()

        self.undone_moves.clear()
        self.eval_history.clear()
        self.wdl_history.clear()

        self.board_widget.reset()

        # Reset game history for new game
        self.game_positions = [self.board_widget.get_board().copy()]

        # Reset all analysis panels
        self.mcts_panel.clear()
        self.search_tree_panel.clear()
        self.eval_bar.clear()
        self.eval_graph.clear()
        self.value_label.configure(text="0.00", fg=COLORS["text_primary"])
        self.opening_display.clear()
        self.move_list.clear()

        self._update_game_info()

        # Reset move time display and start timer
        self.turn_start_time = time.time()

        # Initialize time control if enabled
        self._init_time_control()

        if self.game_mode == "human_vs_ai":
            self.board_widget.set_player_color(self.human_color)
            if self.human_color == chess.BLACK:
                self._trigger_ai_move()
        elif self.game_mode == "human_vs_human":
            self.board_widget.set_player_color(None)
        else:
            self.board_widget.set_player_color(None)
            self._trigger_ai_move()

    def _cancel_selection(self) -> None:
        """Cancel current piece selection."""
        self.board_widget.selected_square = None
        self.board_widget.legal_moves = set()
        self.board_widget.drag_from = None
        self.board_widget.drag_piece = None
        self.board_widget._draw_board()

    def _rebuild_game_positions(self) -> None:
        """Rebuild game positions from current board state.

        Called after undo/redo to ensure history is consistent.
        Reconstructs all positions from the move stack.
        """
        board = self.board_widget.get_board()
        move_stack = list(board.move_stack)

        # Create a fresh board and replay to get all positions
        temp_board = chess.Board()
        self.game_positions = [temp_board.copy()]  # Starting position

        for move in move_stack:
            temp_board.push(move)
            self.game_positions.append(temp_board.copy())

    def _back_one_move(self) -> None:
        """Go back one move (or move pair in human vs AI)."""
        if self.ai_thinking or self.board_widget.is_animating:
            return

        board = self.board_widget.get_board()
        if not board.move_stack:
            return

        # Clear MCTS tree when going back (tree is no longer valid)
        if self.mcts is not None:
            self.mcts.clear_cache()

        self.ai_cancelled = True
        anim_duration = 100

        if self.game_mode == "human_vs_ai":
            if board.turn == self.human_color:
                if len(board.move_stack) >= 2:
                    ai_move = board.peek()
                    self.undone_moves.append((board.move_stack[-2], ai_move))

                    def after_ai_undo():
                        board.pop()
                        human_move = board.peek()

                        def after_human_undo():
                            board.pop()
                            self.board_widget.set_board(board)
                            self.board_widget.set_last_move(
                                board.peek() if board.move_stack else None
                            )
                            self._update_game_info()
                            self._update_eval_for_position()
                            self._rebuild_game_positions()

                        self.board_widget.animate_move(
                            human_move,
                            duration_ms=anim_duration,
                            on_complete=after_human_undo,
                            reverse=True,
                        )

                    self.board_widget.animate_move(
                        ai_move,
                        duration_ms=anim_duration,
                        on_complete=after_ai_undo,
                        reverse=True,
                    )
            else:
                if len(board.move_stack) >= 1:
                    human_move = board.peek()
                    self.undone_moves.append((human_move, None))

                    def after_undo():
                        board.pop()
                        self.board_widget.set_board(board)
                        self.board_widget.set_last_move(
                            board.peek() if board.move_stack else None
                        )
                        self._update_game_info()
                        self._update_eval_for_position()
                        self._rebuild_game_positions()

                    self.board_widget.animate_move(
                        human_move,
                        duration_ms=anim_duration,
                        on_complete=after_undo,
                        reverse=True,
                    )
        else:
            last_move = board.peek()
            self.undone_moves.append((last_move, None))

            def after_undo():
                board.pop()
                self.board_widget.set_board(board)
                self.board_widget.set_last_move(
                    board.peek() if board.move_stack else None
                )
                self._update_game_info()
                self._rebuild_game_positions()
                self._update_eval_for_position()

            self.board_widget.animate_move(
                last_move,
                duration_ms=anim_duration,
                on_complete=after_undo,
                reverse=True,
            )

    def _forward_one_move(self) -> None:
        """Go forward (replay undone moves)."""
        if self.ai_thinking or self.board_widget.is_animating:
            return

        if not self.undone_moves:
            return

        # Clear MCTS tree when going forward (for consistency after back)
        if self.mcts is not None:
            self.mcts.clear_cache()

        board = self.board_widget.get_board()
        human_move, ai_move = self.undone_moves.pop()

        if human_move not in board.legal_moves:
            self.undone_moves.clear()
            return

        anim_duration = 100

        def after_human_move():
            board.push(human_move)
            self.board_widget.set_board(board)

            if ai_move and ai_move in board.legal_moves:

                def after_ai_move():
                    board.push(ai_move)
                    self.board_widget.set_board(board)
                    self.board_widget.set_last_move(ai_move)
                    self._update_game_info()
                    self._update_eval_for_position()
                    self._rebuild_game_positions()

                    if (
                        self.game_mode == "human_vs_ai"
                        and board.turn != self.human_color
                        and not board.is_game_over()
                        and not self.undone_moves
                    ):
                        self._trigger_ai_move()

                self.board_widget.animate_move(
                    ai_move, duration_ms=anim_duration, on_complete=after_ai_move
                )
            else:
                self.board_widget.set_last_move(human_move)
                self._update_game_info()
                self._update_eval_for_position()
                self._rebuild_game_positions()

        self.board_widget.animate_move(
            human_move, duration_ms=anim_duration, on_complete=after_human_move
        )

    def _flip_board(self) -> None:
        """Flip the board orientation."""
        self.board_widget.flip()

    def _update_move_time_display(self, player: str, elapsed: float) -> None:
        """Update the last move time display."""
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes}m {seconds:.1f}s"

    # ==================== Time Control Methods ====================

    def _init_time_control(self) -> None:
        """Initialize timers for a new game with time control."""
        if self.time_control is None:
            return
        self.white_time = float(self.time_control)
        self.black_time = float(self.time_control)
        self.game_started = False  # Timer doesn't run until first move

        # Reset dynamic time management stats
        self.ai_move_times.clear()
        self.ai_sims_used.clear()
        self.last_wdl = None

        self._update_player_timer_display(chess.WHITE)
        self._update_player_timer_display(chess.BLACK)
        # Don't start timer loop yet - wait for first move

    def _start_timer_loop(self) -> None:
        """Start the timer update loop (100ms interval)."""
        if self.time_control is None:
            return
        self._update_timer()

    def _update_timer(self) -> None:
        """Update the active player's timer display."""
        if self.time_control is None:
            return

        board = self.board_widget.get_board()
        if board.is_game_over():
            return

        # Calculate elapsed time since turn started
        elapsed = time.time() - self.turn_start_time
        current_turn = board.turn

        # Compute display time (remaining - elapsed)
        if current_turn == chess.WHITE:
            display_time = max(0, self.white_time - elapsed)
        else:
            display_time = max(0, self.black_time - elapsed)

        self._update_player_timer_display(current_turn, display_time)

        # Check for timeout
        if display_time <= 0:
            self._handle_timeout(current_turn)
            return

        # Schedule next update
        self.timer_update_id = self.root.after(100, self._update_timer)

    def _update_player_timer_display(
        self, color: chess.Color, time_remaining: float = None
    ) -> None:
        """Update the timer display in PlayerInfo widget."""
        if time_remaining is None:
            time_remaining = (
                self.white_time if color == chess.WHITE else self.black_time
            )

        # Format as MM:SS
        minutes = int(time_remaining // 60)
        seconds = int(time_remaining % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        # Color based on urgency
        if time_remaining < 30:
            timer_color = COLORS.get("timer_urgent", "#ef4444")
        elif time_remaining < 60:
            timer_color = COLORS.get("timer_warning", "#f59e0b")
        else:
            timer_color = COLORS["text_primary"]

        # Update the correct PlayerInfo widget
        player_info = (
            self.white_player_info if color == chess.WHITE else self.black_player_info
        )
        player_info.set_timer(time_str, timer_color)

    def _commit_time_for_turn(self, color: chess.Color) -> None:
        """Deduct time used for the completed turn."""
        if self.time_control is None:
            return
        elapsed = time.time() - self.turn_start_time
        if color == chess.WHITE:
            self.white_time = max(0, self.white_time - elapsed)
        else:
            self.black_time = max(0, self.black_time - elapsed)

    def _handle_timeout(self, color: chess.Color) -> None:
        """Handle game over by timeout."""
        # Cancel timer loop
        if self.timer_update_id:
            self.root.after_cancel(self.timer_update_id)
            self.timer_update_id = None

        # Cancel AI if thinking
        self.ai_cancelled = True
        self.ai_thinking = False
        self.thinking_indicator.stop()
        self.thinking_indicator.pack_forget()

        loser = "White" if color == chess.WHITE else "Black"
        winner = "Black" if color == chess.WHITE else "White"

        self.status_label.configure(
            text=f"{loser} ran out of time! {winner} wins", fg=COLORS["accent"]
        )
        messagebox.showinfo(
            "Game Over", f"{loser} ran out of time!\n{winner} wins on time."
        )

    def _estimate_moves_remaining(self, board: chess.Board) -> int:
        """Estimate remaining moves based on game phase."""
        pieces = len(board.piece_map())

        # Opening (> 28 pieces): ~28 moves remaining
        if pieces > 28:
            return 28

        # Middlegame (16-28 pieces): linear estimation
        elif pieces > 16:
            return int(15 + (pieces - 16) * 0.8)

        # Endgame (< 16 pieces): can be long
        else:
            # Pawn presence = longer endgame
            pawns = len(board.pieces(chess.PAWN, chess.WHITE)) + len(
                board.pieces(chess.PAWN, chess.BLACK)
            )
            if pawns > 4:
                return max(20, pieces * 2)  # Pawn endgames are long
            return max(10, pieces)  # Light piece endgames

    def _get_smoothed_time_per_sim(self) -> float:
        """Get exponentially smoothed time per simulation from history."""
        if not self.ai_move_times or not self.ai_sims_used:
            return 0.001  # Fallback

        n = min(len(self.ai_move_times), self.HISTORY_WINDOW)
        total_weighted_time = 0.0
        total_weighted_sims = 0.0
        weight = 1.0

        for i in range(n):
            idx = -(i + 1)
            total_weighted_time += self.ai_move_times[idx] * weight
            total_weighted_sims += self.ai_sims_used[idx] * weight
            weight *= self.DECAY_FACTOR

        return total_weighted_time / max(1, total_weighted_sims)

    def _get_wdl_factor(self) -> float:
        """Get time factor based on WDL entropy (position uncertainty)."""
        if self.last_wdl is None:
            return 1.0

        win, draw, loss = self.last_wdl

        # Shannon entropy (max ~1.58 for uniform distribution)
        entropy = 0.0
        for p in [win, draw, loss]:
            if p > 0.001:
                entropy -= p * math.log2(p)

        # Gentle adjustments only (0.9 - 1.1 range)
        # Low entropy = clear position = slightly less time
        # High entropy = uncertain = slightly more time
        if entropy < 0.3:
            return 0.9  # Very clear (winning/losing)
        elif entropy > 1.4:
            return 1.1  # Very uncertain

        return 1.0

    def _adjust_for_complexity(self, base_sims: int, board: chess.Board) -> int:
        """Adjust simulations based on position complexity from MCTS data."""
        if self.mcts is None:
            return base_sims

        try:
            branching = self.mcts.get_max_branching_factor(board)
            depth = self.mcts.get_tree_depth(board)

            # Simple position (few branches, deep search) - gentle reduction
            if branching < 5 and depth > 15:
                return int(base_sims * 0.9)

            # Complex position (many branches) - slight increase
            if branching > 25:
                return int(base_sims * 1.1)
        except Exception:
            pass  # MCTS methods may fail if tree is empty

        return base_sims

    def _get_adapted_simulations(self) -> int:
        """
        Dynamically calculate optimal simulations using:
        - Exponentially smoothed time per simulation
        - Phase-aware move estimation
        - WDL entropy analysis
        - MCTS complexity data
        - Emergency mode for time pressure

        First move uses base_simulations. Adaptation starts from 2nd move.
        """
        if self.time_control is None:
            return self.base_simulations

        # First move = use base simulations (no history yet)
        if not self.ai_move_times or not self.ai_sims_used:
            return self.base_simulations

        # Get AI's remaining time
        ai_color = chess.BLACK if self.human_color == chess.WHITE else chess.WHITE
        ai_time = self.black_time if ai_color == chess.BLACK else self.white_time
        board = self.board_widget.get_board()

        if ai_time <= 0:
            return 30  # Absolute minimum

        # Get smoothed time per simulation
        time_per_sim = self._get_smoothed_time_per_sim()

        # EMERGENCY MODE: < 15 seconds
        if ai_time < self.EMERGENCY_THRESHOLD:
            emergency_sims = max(30, int(ai_time / time_per_sim * 0.8))
            return min(emergency_sims, self.MIN_SIMULATIONS)

        # Estimate remaining moves (phase-aware)
        moves_left = self._estimate_moves_remaining(board)

        # Time budget per move
        time_budget = ai_time / moves_left

        # Max simulations we can afford
        if time_per_sim > 0:
            max_sims = int(time_budget / time_per_sim)
        else:
            max_sims = self.base_simulations

        # Apply WDL entropy factor
        wdl_factor = self._get_wdl_factor()
        target_sims = int(max_sims * wdl_factor)

        # Adjust for position complexity (uses MCTS data)
        target_sims = self._adjust_for_complexity(target_sims, board)

        # Clamp to reasonable bounds (max 2x base when time allows)
        final_sims = max(self.MIN_SIMULATIONS, min(self.base_simulations, target_sims))

        return final_sims

    def _update_ai_simulations_for_time(self) -> None:
        """Adapt AI simulations based on dynamic time management."""
        if self.time_control is None:
            return

        new_sims = self._get_adapted_simulations()
        print(f"New simulations: {new_sims}")

        if self.mcts is not None:
            self.mcts.num_simulations = new_sims

    # ==================== End Time Control Methods ====================

    def _on_human_move(self, move: chess.Move) -> None:
        """Handle human move."""
        # Start game timer on first move (don't commit time for first move)
        if self.time_control is not None and not self.game_started:
            self.game_started = True
            self.turn_start_time = time.time()  # Start timing from now
            self._start_timer_loop()
        else:
            # Commit time for human's turn (time control)
            self._commit_time_for_turn(self.human_color)

        # Record human thinking time
        self.last_move_time = time.time() - self.turn_start_time
        self._update_move_time_display("You", self.last_move_time)

        self.undone_moves.clear()
        self._update_game_info()

        # Update game history
        board = self.board_widget.get_board()
        self.game_positions.append(board.copy())

        # Tree reuse: preserve subtree for next AI search
        if self.mcts is not None:
            self.mcts.advance_root(board)

        # Update evaluation
        self._update_eval_after_move()

        board = self.board_widget.get_board()
        if board.is_game_over():
            self._show_game_over()
        elif self.game_mode == "human_vs_ai":
            # Start timer for AI
            self.turn_start_time = time.time()
            # Adapt AI simulations based on remaining time
            self._update_ai_simulations_for_time()
            self._trigger_ai_move()
        else:
            # Human vs Human: start timer for next player
            self.turn_start_time = time.time()

    def _trigger_ai_move(self) -> None:
        """Start AI move calculation in background thread."""
        if self.mcts is None and self.pure_mcts_player is None:
            return

        # Check for instant move (< 3 seconds) - play from tree without search
        if self.time_control is not None:
            ai_color = chess.BLACK if self.human_color == chess.WHITE else chess.WHITE
            ai_time = self.black_time if ai_color == chess.BLACK else self.white_time

            if ai_time < self.INSTANT_MOVE_THRESHOLD:
                instant_move = self._get_instant_move()
                if instant_move is not None:
                    # Play immediately without search
                    self._apply_instant_move(instant_move)
                    return

        self.ai_thinking = True
        self.ai_cancelled = False

        self.thinking_indicator.pack()
        self.thinking_indicator.start()

        self.ai_thread = threading.Thread(target=self._compute_ai_move, daemon=True)
        self.ai_thread.start()

    def _get_instant_move(self) -> chess.Move | None:
        """Get best move from current MCTS tree without searching."""
        board = self.board_widget.get_board()

        # Try to get move with most visits from existing tree
        if self.mcts is not None:
            try:
                visit_counts = self.mcts.get_visit_counts(board)
                if visit_counts:
                    best_move = max(visit_counts, key=visit_counts.get)
                    print(
                        f"Instant move from tree: {best_move} ({visit_counts[best_move]} visits)"
                    )
                    return best_move
            except Exception:
                pass

        # Fallback: random legal move
        legal_moves = list(board.legal_moves)
        if legal_moves:
            import random

            move = random.choice(legal_moves)
            print(f"Instant move (random fallback): {move}")
            return move

        return None

    def _apply_instant_move(self, move: chess.Move) -> None:
        """Apply an instant move without full AI processing."""
        board = self.board_widget.get_board()

        # Commit AI time
        ai_color = chess.BLACK if self.human_color == chess.WHITE else chess.WHITE
        self._commit_time_for_turn(ai_color)

        # Apply the move
        board.push(move)
        self.board_widget.push_move(move)

        # Update game history
        self.game_positions.append(board.copy())

        # Update UI
        self._update_game_info()
        self.turn_start_time = time.time()

        # Check game over
        if board.is_game_over():
            self._show_game_over()

    def _compute_ai_move(self) -> None:
        """Compute AI move in background thread."""
        try:
            board = self.board_widget.get_board()

            if board.is_game_over():
                return

            root_value = 0.0
            mcts_stats = None
            tree_data = None
            tree_depth = 0
            tree_branching = 0
            mate_in = None

            if self.use_pure_mcts and self.pure_mcts_player is not None:
                move = self.pure_mcts_player.select_move(board)
            else:
                # Pass full game history to MCTS (excluding current position)
                # IMPORTANT: MCTS does _initial_history[:history_length], taking first positions
                # So we pass all positions from game start in order [pos_initial, pos_after_move1, ...]
                history_boards = self.game_positions[:-1] if len(self.game_positions) > 1 else None
                move = self.mcts.get_best_move(board, add_noise=False, history_boards=history_boards)

                # Capture MCTS statistics for display
                mcts_stats = self.mcts.get_root_statistics(board)
                root_value = self.mcts.get_root_value(board)
                tree_data = self.mcts.get_search_tree_data(board, top_n=5, max_depth=5)
                tree_depth = self.mcts.get_tree_depth(board)
                tree_branching = self.mcts.get_max_branching_factor(board)
                mate_in = self.mcts.get_mate_in(board)

                if self.verbose and not self.ai_cancelled:
                    self.mcts.print_search_tree(board, top_n=20, max_depth=20)

            if not self.ai_cancelled and move:
                self.update_queue.put(
                    (
                        "ai_move",
                        {
                            "move": move,
                            "mcts_stats": mcts_stats,
                            "root_value": root_value,
                            "tree_data": tree_data,
                            "tree_depth": tree_depth,
                            "tree_branching": tree_branching,
                            "mate_in": mate_in,
                        },
                    )
                )

        except Exception as e:
            self.update_queue.put(("error", str(e)))

    def _start_update_loop(self) -> None:
        """Start the UI update loop."""
        self._process_updates()

    def _process_updates(self) -> None:
        """Process queued updates from threads."""
        try:
            while True:
                update_type, data = self.update_queue.get_nowait()

                if update_type == "ai_move":
                    self._apply_ai_move(data)
                elif update_type == "training":
                    if self.training_panel:
                        self.training_panel.update_progress(data)
                    if isinstance(data, dict) and data.get("phase") == "move":
                        fen = data.get("fen")
                        if fen:
                            self._show_training_position(fen)
                elif update_type == "error":
                    messagebox.showerror("Error", data)

        except queue.Empty:
            pass

        self.root.after(50, self._process_updates)

    def _show_training_position(self, fen: str) -> None:
        """Show a position from training on the board."""
        try:
            board = chess.Board(fen)
            self.board_widget.set_board(board)
        except Exception:
            pass

    def _apply_ai_move(self, data: dict) -> None:
        """Apply AI move to the board with animation."""
        self.ai_thinking = False

        ai_color = chess.BLACK if self.human_color == chess.WHITE else chess.WHITE

        # Start game timer on first move (AI plays first when human is Black)
        if self.time_control is not None and not self.game_started:
            self.game_started = True
            self.turn_start_time = time.time()  # Start timing from now
            self._start_timer_loop()
        else:
            # Commit time for AI's turn (time control)
            self._commit_time_for_turn(ai_color)

        # Record AI thinking time
        self.last_move_time = time.time() - self.turn_start_time
        self._update_move_time_display("AI", self.last_move_time)

        self.thinking_indicator.stop()
        self.thinking_indicator.pack_forget()

        move = data["move"]
        mcts_stats = data.get("mcts_stats")
        root_value = data.get("root_value", 0.0)
        tree_data = data.get("tree_data")
        tree_depth = data.get("tree_depth", 0)
        tree_branching = data.get("tree_branching", 0)
        mate_in = data.get("mate_in")

        board = self.board_widget.get_board()

        # Update MCTS panel with search statistics
        if mcts_stats:
            total_visits = sum(s["visits"] for s in mcts_stats)
            self.mcts_panel.update_stats(board, mcts_stats, total_visits)

            # Record stats for dynamic time management
            if self.time_control is not None:
                self.ai_move_times.append(self.last_move_time)
                self.ai_sims_used.append(total_visits)

                # Extract WDL from best move (first in list, sorted by visits)
                if mcts_stats and "wdl" in mcts_stats[0]:
                    wdl = mcts_stats[0]["wdl"]
                    # wdl is numpy array [win, draw, loss]
                    self.last_wdl = (float(wdl[0]), float(wdl[1]), float(wdl[2]))

        # Update Search Tree panel
        if tree_data:
            self.search_tree_panel.update_tree(
                tree_data, tree_depth, tree_branching, mate_in
            )

        # Update Value label
        value_color = (
            COLORS["q_value_positive"]
            if root_value > 0.05
            else (
                COLORS["q_value_negative"]
                if root_value < -0.05
                else COLORS["text_primary"]
            )
        )
        self.value_label.configure(text=f"{root_value:+.2f}", fg=value_color)

        if move in board.legal_moves:

            # Extract WDL from best move for eval graph
            best_wdl = None
            if mcts_stats and "wdl" in mcts_stats[0]:
                best_wdl = mcts_stats[0]["wdl"]
                if not isinstance(best_wdl, np.ndarray):
                    best_wdl = np.array(best_wdl, dtype=np.float32)

            def on_animation_complete():
                board.push(move)
                self.board_widget.set_board(board)
                self.board_widget.set_last_move(move)
                self._update_game_info()

                # Update game history
                self.game_positions.append(board.copy())

                # Tree reuse: preserve subtree for next search
                if self.mcts is not None:
                    self.mcts.advance_root(board)

                # Update evaluation after AI move (with WDL for win rate display)
                self._update_eval_after_move(ai_value=root_value, wdl=best_wdl)

                updated_board = self.board_widget.get_board()
                if updated_board.is_game_over():
                    self._show_game_over()
                elif self.game_mode == "ai_vs_ai":
                    # Start timer for next AI
                    self.turn_start_time = time.time()
                    self.root.after(300, self._trigger_ai_move)
                else:
                    # Start timer for human
                    self.turn_start_time = time.time()

            self.board_widget.animate_move(
                move, duration_ms=300, on_complete=on_animation_complete
            )
        else:
            self._update_game_info()

    def _update_eval_after_move(
        self, ai_value: float = None, wdl: Optional[np.ndarray] = None
    ) -> None:
        """Update evaluation displays after a move."""
        # Only update eval graph after AI moves (when ai_value is provided)
        if ai_value is None:
            return

        # Use AI's evaluation directly (matches Value Prediction display)
        # Positive = AI thinks it's winning, Negative = AI thinks it's losing
        display_eval = ai_value

        # Update eval bar (scale from [-1,1] to [-10,10])
        self.eval_bar.set_evaluation(display_eval * 10)

        # Add to eval history and update graph
        self.eval_history.append(display_eval)
        self.wdl_history.append(wdl)
        self.eval_graph.add_evaluation(display_eval, wdl)

    def _update_eval_for_position(self) -> None:
        """Update evaluation for current position (after back/forward)."""
        board = self.board_widget.get_board()
        move_index = len(board.move_stack)

        # Update eval bar and graph to current position
        if move_index < len(self.eval_history):
            # Trim eval history to current position
            self.eval_history = self.eval_history[:move_index]
            self.wdl_history = self.wdl_history[:move_index]
            self.eval_graph.set_history(self.eval_history, self.wdl_history)

        if self.eval_history:
            self.eval_bar.set_evaluation(self.eval_history[-1] * 10)
        else:
            self.eval_bar.clear()

        # Note: Network Output and MCTS panels stay showing the last AI thinking position

        self.move_list.set_current_move(move_index)

    def _update_game_info(self) -> None:
        """Update all game information displays."""
        board = self.board_widget.get_board()

        # Update player info widgets
        self.white_player_info.update_from_board(board)
        self.black_player_info.update_from_board(board)

        # Update opening display
        self.opening_display.update_opening(board)

        # Update phase indicator
        self.phase_indicator.update_phase(board)

        # Update move list
        self.move_list.set_moves(board)

        # Status
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            self.status_label.configure(
                text=f"Checkmate! {winner} wins",
                fg=COLORS["accent"],
            )
        elif board.is_stalemate():
            self.status_label.configure(text="Stalemate!", fg=COLORS["warning"])
        elif board.is_insufficient_material():
            self.status_label.configure(
                text="Draw - Insufficient material",
                fg=COLORS["warning"],
            )
        elif board.is_check():
            self.status_label.configure(text="Check!", fg=COLORS["error"])
        elif self.ai_thinking:
            self.status_label.configure(text="AI thinking...", fg=COLORS["accent"])
        elif self.game_mode == "human_vs_ai" and board.turn == self.human_color:
            self.status_label.configure(text="Your turn", fg=COLORS["success"])
        elif self.game_mode == "human_vs_human":
            turn = "White" if board.turn == chess.WHITE else "Black"
            self.status_label.configure(text=f"{turn} to play", fg=COLORS["success"])
        else:
            self.status_label.configure(text="Playing", fg=COLORS["text_secondary"])

    def _show_game_over(self) -> None:
        """Show game over dialog."""
        board = self.board_widget.get_board()

        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            message = f"Checkmate! {winner} wins!"
        elif board.is_stalemate():
            message = "Game over - Stalemate!"
        elif board.is_insufficient_material():
            message = "Game over - Draw by insufficient material"
        elif board.can_claim_fifty_moves():
            message = "Game over - Draw by fifty-move rule"
        elif board.is_repetition():
            message = "Game over - Draw by repetition"
        else:
            message = "Game over!"

        messagebox.showinfo("Game Over", message)

    def _start_training(self) -> None:
        """Start self-play training."""
        if not self.training_panel:
            return
        if self.training_thread and self.training_thread.is_alive():
            return

        self.training_cancelled = False
        self.training_iterations = self.training_panel.get_iterations()
        self.training_output_file = self.training_panel.get_output_file()

        try:
            from alphazero import AlphaZeroTrainer, TrainingConfig

            config = TrainingConfig(
                games_per_iteration=10,
                num_simulations=50,
                epochs_per_iteration=5,
                batch_size=64,
            )

            if self.network is None:
                from alphazero import DualHeadNetwork

                self.network = DualHeadNetwork()

            self.trainer = AlphaZeroTrainer(self.network, config)

            self.training_thread = threading.Thread(
                target=self._run_training,
                daemon=True,
            )
            self.training_thread.start()

        except Exception as e:
            messagebox.showerror("Training Error", str(e))
            if self.training_panel:
                self.training_panel.reset()

    def _run_training(self) -> None:
        """Run training loop in background thread."""
        try:

            def callback(data):
                if not self.training_cancelled:
                    self.update_queue.put(("training", data))

            for i in range(self.training_iterations):
                if self.training_cancelled:
                    break

                self.update_queue.put(
                    (
                        "training",
                        {
                            "phase": "iteration_start",
                            "iteration": i + 1,
                            "total_iterations": self.training_iterations,
                        },
                    )
                )

                self.trainer.train_iteration(callback=callback)

                self.update_queue.put(
                    (
                        "training",
                        {
                            "phase": "iteration_complete",
                            "iteration": i + 1,
                        },
                    )
                )

            if not self.training_cancelled:
                self.network.save(self.training_output_file)
                self.update_queue.put(
                    (
                        "training",
                        {
                            "phase": "complete",
                            "saved_to": self.training_output_file,
                        },
                    )
                )

            from alphazero import MCTS

            self.mcts = MCTS(
                network=self.network,
                num_simulations=self.num_simulations,
                batch_size=self.mcts_batch_size,
                history_length=self.history_length,
                c_puct=self.c_puct,
                # WDL-aware settings
                contempt=0.5,
                uncertainty_weight=0.2,
                draw_sibling_fpu=True,
            )

        except Exception as e:
            self.update_queue.put(("error", f"Training error: {e}"))

    def _stop_training(self) -> None:
        """Stop training."""
        self.training_cancelled = True

    # ==================== Editor Mode Methods ====================

    def _on_app_mode_change(self) -> None:
        """Handle switch between Play and Editor mode."""
        mode = self.app_mode_var.get()
        self.app_mode = mode

        if mode == "editor":
            # Switch to editor mode
            self.play_left_content.pack_forget()
            self.editor_left_content.pack(fill="both", expand=True)

            # Enable editor mode on board widget
            self.board_widget.editor_mode = True
            self.board_widget.on_position_changed = self._on_editor_position_change

            # Cancel any AI thinking
            self.ai_cancelled = True
            self.ai_thinking = False
            self.thinking_indicator.stop()
            self.thinking_indicator.pack_forget()

            # Update status
            self.status_label.configure(
                text="Editor Mode - Place pieces", fg=COLORS["accent"]
            )

            # Sync editor controls with current board state
            self._sync_editor_controls_from_board()

        else:
            # Switch to play mode
            self.editor_left_content.pack_forget()
            self.play_left_content.pack(fill="both", expand=True)

            # Disable editor mode
            self.board_widget.editor_mode = False
            self.board_widget.on_position_changed = None
            self.board_widget.selected_piece_to_place = None

            # Clear editor selection highlight
            self._clear_palette_highlight()

            # Update status
            self._update_game_info()

    def _select_editor_piece(self, color: chess.Color, piece_type: int) -> None:
        """Select a piece from the palette to place on the board."""
        piece = chess.Piece(piece_type, color)
        self.editor_selected_piece = piece
        self.board_widget.selected_piece_to_place = piece

        # Highlight selected button
        self._update_palette_highlight(color, piece_type)

    def _select_editor_eraser(self) -> None:
        """Select eraser mode (right-click to remove pieces)."""
        self.editor_selected_piece = None
        self.board_widget.selected_piece_to_place = None
        self._clear_palette_highlight()

        # Highlight eraser button
        self.eraser_btn.configure(bg=COLORS["accent"])

    def _update_palette_highlight(self, color: chess.Color, piece_type: int) -> None:
        """Highlight the selected piece in the palette."""
        self._clear_palette_highlight()

        # Map piece type to symbol
        piece_symbols = {
            chess.KING: "K", chess.QUEEN: "Q", chess.ROOK: "R",
            chess.BISHOP: "B", chess.KNIGHT: "N", chess.PAWN: "P",
        }
        symbol = piece_symbols.get(piece_type, "")

        if color == chess.WHITE:
            key = f"w{symbol}"
        else:
            key = f"b{symbol.lower()}"

        if key in self.editor_palette_buttons:
            self.editor_palette_buttons[key].configure(bg=COLORS["accent"])

    def _clear_palette_highlight(self) -> None:
        """Clear all palette button highlights."""
        for btn in self.editor_palette_buttons.values():
            btn.configure(bg=COLORS["bg_tertiary"])
        self.eraser_btn.configure(bg=COLORS["bg_tertiary"])

    def _on_editor_position_change(self) -> None:
        """Called when the board position changes in editor mode."""
        self._update_editor_fen()

    def _sync_editor_controls_from_board(self) -> None:
        """Sync editor controls (turn, castling) from current board state."""
        board = self.board_widget.get_board()

        # Sync turn
        self.editor_turn_var.set("white" if board.turn == chess.WHITE else "black")

        # Sync castling rights
        self.castle_wk_var.set(board.has_kingside_castling_rights(chess.WHITE))
        self.castle_wq_var.set(board.has_queenside_castling_rights(chess.WHITE))
        self.castle_bk_var.set(board.has_kingside_castling_rights(chess.BLACK))
        self.castle_bq_var.set(board.has_queenside_castling_rights(chess.BLACK))

        # Update FEN display
        self._update_editor_fen()

    def _update_editor_fen(self) -> None:
        """Update the FEN display in editor panel."""
        board = self.board_widget.get_board().copy()

        # Apply turn from editor control
        board.turn = chess.WHITE if self.editor_turn_var.get() == "white" else chess.BLACK

        # Apply castling rights from editor controls
        castling = ""
        if self.castle_wk_var.get():
            castling += "K"
        if self.castle_wq_var.get():
            castling += "Q"
        if self.castle_bk_var.get():
            castling += "k"
        if self.castle_bq_var.get():
            castling += "q"
        board.set_castling_fen(castling or "-")

        self.editor_fen_var.set(board.fen())

    def _load_editor_fen(self) -> None:
        """Load a position from the FEN entry field."""
        fen = self.editor_fen_var.get().strip()
        if not fen:
            messagebox.showwarning("Invalid FEN", "Please enter a FEN string.")
            return

        try:
            board = chess.Board(fen)
            self.board_widget.set_board(board)
            self.board_widget._draw_board()

            # Sync controls from loaded FEN
            self._sync_editor_controls_from_board()

        except ValueError as e:
            messagebox.showerror("Invalid FEN", f"Could not parse FEN:\n{e}")

    def _editor_clear_board(self) -> None:
        """Clear all pieces from the board."""
        board = self.board_widget.get_board()
        board.clear()
        self.board_widget.set_board(board)
        self.board_widget._draw_board()
        self._update_editor_fen()

    def _editor_reset_position(self) -> None:
        """Reset to the starting position."""
        board = chess.Board()
        self.board_widget.set_board(board)
        self.board_widget._draw_board()
        self._sync_editor_controls_from_board()

    def _start_game_from_editor(self) -> None:
        """Start a game from the current editor position."""
        board = self.board_widget.get_board().copy()

        # Apply turn from editor
        board.turn = chess.WHITE if self.editor_turn_var.get() == "white" else chess.BLACK

        # Apply castling rights
        castling = ""
        if self.castle_wk_var.get():
            castling += "K"
        if self.castle_wq_var.get():
            castling += "Q"
        if self.castle_bk_var.get():
            castling += "k"
        if self.castle_bq_var.get():
            castling += "q"
        board.set_castling_fen(castling or "-")

        # Validate position
        if not self._validate_editor_position(board):
            return

        # Set the board with applied settings
        self.board_widget.set_board(board)

        # Switch to play mode
        self.app_mode_var.set("play")
        self._on_app_mode_change()

        # Clear MCTS cache for new position
        if self.mcts is not None:
            self.mcts.clear_cache()

        # Reset game state
        self.undone_moves.clear()
        self.eval_history.clear()
        self.wdl_history.clear()
        self.mcts_panel.clear()
        self.search_tree_panel.clear()
        self.eval_bar.clear()
        self.eval_graph.clear()
        self.value_label.configure(text="0.00", fg=COLORS["text_primary"])
        self.opening_display.clear()
        self.move_list.clear()

        # Reset time control if enabled
        self._init_time_control()

        # Update displays
        self._update_game_info()

        # Trigger AI move if it's AI's turn
        if self.game_mode == "human_vs_ai":
            if board.turn != self.human_color:
                self.turn_start_time = time.time()
                self._trigger_ai_move()
        elif self.game_mode == "ai_vs_ai":
            self.turn_start_time = time.time()
            self._trigger_ai_move()

    def _validate_editor_position(self, board: chess.Board) -> bool:
        """Validate the editor position before starting a game."""
        errors = []

        # Check for both kings
        white_kings = len(board.pieces(chess.KING, chess.WHITE))
        black_kings = len(board.pieces(chess.KING, chess.BLACK))

        if white_kings == 0:
            errors.append("White king is missing")
        elif white_kings > 1:
            errors.append("Multiple white kings")

        if black_kings == 0:
            errors.append("Black king is missing")
        elif black_kings > 1:
            errors.append("Multiple black kings")

        # Check for pawns on first/last rank
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                rank = chess.square_rank(square)
                if rank == 0 or rank == 7:
                    errors.append(f"Pawn on invalid rank at {chess.square_name(square)}")

        # Check if opponent king is in check (illegal position)
        if white_kings == 1 and black_kings == 1:
            # Temporarily flip turn to check if opponent is in check
            test_board = board.copy()
            test_board.turn = not board.turn
            if test_board.is_check():
                opponent = "White" if board.turn == chess.BLACK else "Black"
                errors.append(f"{opponent} king would be in check (illegal)")

        if errors:
            messagebox.showerror(
                "Invalid Position",
                "Cannot start game:\n\n• " + "\n• ".join(errors)
            )
            return False

        return True

    # ==================== End Editor Mode Methods ====================

    def run(self) -> None:
        """Run the application main loop."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"+{x}+{y}")

        self.root.mainloop()


def main():
    """Main entry point for the chess game."""
    import argparse

    parser = argparse.ArgumentParser(description="NeuralMate Chess - AlphaZero")
    parser.add_argument("--network", "-n", type=str, help="Path to network file")
    parser.add_argument(
        "--simulations", "-s", type=int, default=200, help="MCTS simulations per move"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print MCTS search tree"
    )
    parser.add_argument(
        "--time",
        "-t",
        type=int,
        default=None,
        help="Time control in minutes per player (e.g., --time 5 for 5 min each)",
    )

    args = parser.parse_args()

    app = ChessGameApp(
        network_path=args.network,
        num_simulations=args.simulations,
        verbose=args.verbose,
        time_control=args.time,
    )
    app.run()


if __name__ == "__main__":
    main()
