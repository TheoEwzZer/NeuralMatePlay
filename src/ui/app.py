"""
Main chess game application.

Provides the complete UI for playing chess against the AlphaZero AI,
watching AI vs AI games, and managing training.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Any
import threading
import queue
import os

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
    Spinner,
)
from .board_widget import ChessBoardWidget
from .training_panel import TrainingPanel, TrainingConfigDialog


class ChessGameApp:
    """
    Main application window for playing chess against AI.

    Features:
    - Human vs AI gameplay
    - Human vs Human gameplay
    - AI vs AI spectating
    - Training panel for self-play
    - Game controls (new game, undo, flip board)
    """

    # Default MCTS settings for play mode (optimized for quality over speed)
    DEFAULT_MCTS_BATCH_SIZE = 8
    DEFAULT_HISTORY_LENGTH = 3

    def __init__(
        self,
        network_path: str | None = None,
        num_simulations: int = 200,
        mcts_batch_size: int = DEFAULT_MCTS_BATCH_SIZE,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        verbose: bool = False,
    ):
        """
        Initialize the chess game application.

        Args:
            network_path: Path to pre-trained network file
            num_simulations: MCTS simulations for AI moves
            mcts_batch_size: Batch size for MCTS GPU inference (default: 8)
            history_length: Number of past positions for encoding (default: 3)
            verbose: If True, print MCTS search tree after each AI move
        """
        self.root = tk.Tk()
        self.root.title("NeuralMate Chess - AlphaZero")
        self.root.resizable(True, True)  # Allow resize
        self.root.geometry("1600x900")  # Default size for 1920x1080 screens
        self.root.minsize(1200, 800)  # Minimum size

        apply_theme(self.root)

        # Network and AI
        self.network = None
        self.mcts = None
        self.pure_mcts_player = None  # For pure MCTS mode (no neural network)
        self.network_path = network_path
        self.num_simulations = num_simulations
        self.mcts_batch_size = mcts_batch_size
        self.history_length = history_length
        self.verbose = verbose
        self.use_pure_mcts = network_path and network_path.lower() == "mcts"

        # Game state
        self.game_mode = "human_vs_ai"  # human_vs_ai, human_vs_human, ai_vs_ai
        self.human_color = chess.WHITE
        self.ai_thinking = False
        self.ai_thread: threading.Thread | None = None
        self.ai_cancelled = False

        # Move history for back/forward navigation
        self.undone_moves: list = []  # Stack of undone moves for forward

        # Update queue for thread-safe UI updates
        self.update_queue = queue.Queue()

        # Training state
        self.trainer = None
        self.training_thread: threading.Thread | None = None
        self.training_cancelled = False

        self._create_ui()
        self._bind_keyboard_shortcuts()
        self._load_network()
        self._start_update_loop()

    def _bind_keyboard_shortcuts(self) -> None:
        """Bind keyboard shortcuts for common actions."""
        # Ctrl+N: New game
        self.root.bind("<Control-n>", lambda e: self._new_game_with_confirm())
        self.root.bind("<Control-N>", lambda e: self._new_game_with_confirm())

        # Ctrl+Z: Back (same as Left arrow)
        self.root.bind("<Control-z>", lambda e: self._back_one_move())
        self.root.bind("<Control-Z>", lambda e: self._back_one_move())

        # Arrow keys: Navigate history
        self.root.bind("<Left>", lambda e: self._back_one_move())
        self.root.bind("<Right>", lambda e: self._forward_one_move())

        # F: Flip board
        self.root.bind("<f>", lambda e: self._flip_board())
        self.root.bind("<F>", lambda e: self._flip_board())

        # Escape: Cancel selection / deselect
        self.root.bind("<Escape>", lambda e: self._cancel_selection())

    def _create_ui(self) -> None:
        """Create the main UI layout."""
        # Main container (larger padding for 1080p)
        main_frame = tk.Frame(self.root, bg=COLORS["bg_primary"])
        main_frame.pack(padx=40, pady=30, fill="both", expand=True)

        # Title
        title_frame = tk.Frame(main_frame, bg=COLORS["bg_primary"])
        title_frame.pack(fill="x", pady=(0, 25))

        title = create_styled_label(
            title_frame,
            "NeuralMate Chess",
            style="title",
        )
        title.pack(side="left")

        subtitle = create_styled_label(
            title_frame,
            "AlphaZero-lite",
            style="body",
            fg=COLORS["accent"],
        )
        subtitle.pack(side="left", padx=(10, 0), pady=(8, 0))

        # Content area
        content_frame = tk.Frame(main_frame, bg=COLORS["bg_primary"])
        content_frame.pack(fill="both", expand=True)

        # Left side - Chess board (larger for 1080p)
        board_frame = tk.Frame(content_frame, bg=COLORS["bg_primary"])
        board_frame.pack(side="left", padx=(0, 30))

        self.board_widget = ChessBoardWidget(
            board_frame,
            size=720,  # Larger board for 1080p displays
            on_move=self._on_human_move,
        )
        self.board_widget.pack()

        # Right side - Controls and info with tabs (wider panel)
        side_panel = tk.Frame(content_frame, bg=COLORS["bg_primary"], width=450)
        side_panel.pack(side="left", fill="both", expand=True)
        side_panel.pack_propagate(False)

        # Create notebook (tabs)
        style = ttk.Style()
        style.configure("Dark.TNotebook", background=COLORS["bg_primary"])
        style.configure(
            "Dark.TNotebook.Tab",
            background=COLORS["bg_secondary"],
            foreground=COLORS["text_primary"],
            padding=[15, 8],
            font=FONTS["body_bold"],
        )
        style.map(
            "Dark.TNotebook.Tab",
            background=[("selected", COLORS["bg_tertiary"])],
            foreground=[("selected", COLORS["text_primary"])],
        )

        self.notebook = ttk.Notebook(side_panel, style="Dark.TNotebook")
        self.notebook.pack(fill="both", expand=True)

        # Game tab
        game_tab = tk.Frame(self.notebook, bg=COLORS["bg_primary"])
        self.notebook.add(game_tab, text="Game")

        # Training tab
        training_tab = tk.Frame(self.notebook, bg=COLORS["bg_primary"])
        self.notebook.add(training_tab, text="Training")

        # Game info panel (in game tab)
        info_panel = create_panel(game_tab, title="Game Info")
        info_panel.pack(fill="x", pady=(0, 10))

        info_content = tk.Frame(info_panel, bg=COLORS["bg_secondary"])
        info_content.pack(fill="x", padx=15, pady=(0, 15))

        # Turn indicator
        turn_frame = tk.Frame(info_content, bg=COLORS["bg_secondary"])
        turn_frame.pack(fill="x", pady=5)

        tk.Label(
            turn_frame,
            text="Turn:",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["body"],
        ).pack(side="left")

        self.turn_label = tk.Label(
            turn_frame,
            text="White",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            font=FONTS["body_bold"],
        )
        self.turn_label.pack(side="right")

        # Status
        status_frame = tk.Frame(info_content, bg=COLORS["bg_secondary"])
        status_frame.pack(fill="x", pady=5)

        tk.Label(
            status_frame,
            text="Status:",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["body"],
        ).pack(side="left")

        self.status_label = tk.Label(
            status_frame,
            text="Your turn",
            bg=COLORS["bg_secondary"],
            fg=COLORS["success"],
            font=FONTS["body_bold"],
        )
        self.status_label.pack(side="right")

        # Move count
        move_frame = tk.Frame(info_content, bg=COLORS["bg_secondary"])
        move_frame.pack(fill="x", pady=5)

        tk.Label(
            move_frame,
            text="Moves:",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["body"],
        ).pack(side="left")

        self.move_label = tk.Label(
            move_frame,
            text="0",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            font=FONTS["body_bold"],
        )
        self.move_label.pack(side="right")

        # Material display
        material_frame = tk.Frame(info_content, bg=COLORS["bg_secondary"])
        material_frame.pack(fill="x", pady=(10, 5))

        tk.Label(
            material_frame,
            text="Material:",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["body"],
        ).pack(anchor="w")

        # White material
        white_mat_frame = tk.Frame(material_frame, bg=COLORS["bg_secondary"])
        white_mat_frame.pack(fill="x", pady=2)

        tk.Label(
            white_mat_frame,
            text="White:",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
        ).pack(side="left", padx=(10, 0))

        self.white_material_label = tk.Label(
            white_mat_frame,
            text="39",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            font=FONTS["mono"],
        )
        self.white_material_label.pack(side="right")

        # Black material
        black_mat_frame = tk.Frame(material_frame, bg=COLORS["bg_secondary"])
        black_mat_frame.pack(fill="x", pady=2)

        tk.Label(
            black_mat_frame,
            text="Black:",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
        ).pack(side="left", padx=(10, 0))

        self.black_material_label = tk.Label(
            black_mat_frame,
            text="39",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            font=FONTS["mono"],
        )
        self.black_material_label.pack(side="right")

        # Material advantage
        adv_frame = tk.Frame(material_frame, bg=COLORS["bg_secondary"])
        adv_frame.pack(fill="x", pady=2)

        tk.Label(
            adv_frame,
            text="Advantage:",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
        ).pack(side="left", padx=(10, 0))

        self.material_advantage_label = tk.Label(
            adv_frame,
            text="=",
            bg=COLORS["bg_secondary"],
            fg=COLORS["accent"],
            font=FONTS["mono"],
        )
        self.material_advantage_label.pack(side="right")

        # AI thinking indicator (animated)
        self.thinking_frame = tk.Frame(info_content, bg=COLORS["bg_secondary"])
        self.thinking_frame.pack(fill="x", pady=10)

        self.thinking_indicator = ThinkingIndicator(
            self.thinking_frame,
            text="AI thinking",
            bg=COLORS["bg_secondary"],
        )
        self.thinking_indicator.pack()
        self.thinking_indicator.pack_forget()  # Hidden by default

        # Game mode selection
        mode_panel = create_panel(game_tab, title="Game Mode")
        mode_panel.pack(fill="x", pady=(0, 10))

        mode_content = tk.Frame(mode_panel, bg=COLORS["bg_secondary"])
        mode_content.pack(fill="x", padx=15, pady=(0, 15))

        self.mode_var = tk.StringVar(value="human_vs_ai")

        mode_human = tk.Radiobutton(
            mode_content,
            text="Human vs AI",
            variable=self.mode_var,
            value="human_vs_ai",
            command=self._on_mode_change,
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            selectcolor=COLORS["bg_tertiary"],
            activebackground=COLORS["bg_secondary"],
            activeforeground=COLORS["text_primary"],
            font=FONTS["body"],
        )
        mode_human.pack(anchor="w", pady=2)

        mode_ai = tk.Radiobutton(
            mode_content,
            text="AI vs AI",
            variable=self.mode_var,
            value="ai_vs_ai",
            command=self._on_mode_change,
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            selectcolor=COLORS["bg_tertiary"],
            activebackground=COLORS["bg_secondary"],
            activeforeground=COLORS["text_primary"],
            font=FONTS["body"],
        )
        mode_ai.pack(anchor="w", pady=2)

        mode_human_vs_human = tk.Radiobutton(
            mode_content,
            text="Human vs Human",
            variable=self.mode_var,
            value="human_vs_human",
            command=self._on_mode_change,
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            selectcolor=COLORS["bg_tertiary"],
            activebackground=COLORS["bg_secondary"],
            activeforeground=COLORS["text_primary"],
            font=FONTS["body"],
        )
        mode_human_vs_human.pack(anchor="w", pady=2)

        # Color selection (for human vs AI)
        self.color_frame = tk.Frame(mode_content, bg=COLORS["bg_secondary"])
        self.color_frame.pack(fill="x", pady=(10, 0))

        tk.Label(
            self.color_frame,
            text="Play as:",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
        ).pack(side="left")

        self.color_var = tk.StringVar(value="white")

        color_white = tk.Radiobutton(
            self.color_frame,
            text="White",
            variable=self.color_var,
            value="white",
            command=self._on_color_change,
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            selectcolor=COLORS["bg_tertiary"],
            font=FONTS["small"],
        )
        color_white.pack(side="left", padx=(10, 5))

        color_black = tk.Radiobutton(
            self.color_frame,
            text="Black",
            variable=self.color_var,
            value="black",
            command=self._on_color_change,
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            selectcolor=COLORS["bg_tertiary"],
            font=FONTS["small"],
        )
        color_black.pack(side="left")

        # Game controls
        controls_frame = tk.Frame(game_tab, bg=COLORS["bg_primary"])
        controls_frame.pack(fill="x", pady=(0, 10))

        btn_row1 = tk.Frame(controls_frame, bg=COLORS["bg_primary"])
        btn_row1.pack(fill="x", pady=(0, 5))

        self.new_game_btn = create_styled_button(
            btn_row1,
            "New Game",
            command=self._new_game,
            style="accent",
        )
        self.new_game_btn.pack(side="left", expand=True, fill="x")

        # Navigation controls (back/forward one move)
        btn_row_analysis = tk.Frame(controls_frame, bg=COLORS["bg_primary"])
        btn_row_analysis.pack(fill="x", pady=(5, 5))

        self.back_btn = create_styled_button(
            btn_row_analysis,
            "< Back",
            command=self._back_one_move,
            style="outline",
        )
        self.back_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.forward_btn = create_styled_button(
            btn_row_analysis,
            "Forward >",
            command=self._forward_one_move,
            style="outline",
        )
        self.forward_btn.pack(side="left", expand=True, fill="x", padx=(5, 0))

        btn_row2 = tk.Frame(controls_frame, bg=COLORS["bg_primary"])
        btn_row2.pack(fill="x")

        self.flip_btn = create_styled_button(
            btn_row2,
            "Flip Board",
            command=self._flip_board,
            style="outline",
        )
        self.flip_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.load_btn = create_styled_button(
            btn_row2,
            "Load Network",
            command=self._load_network_dialog,
            style="outline",
        )
        self.load_btn.pack(side="left", expand=True, fill="x", padx=(5, 0))

        # Add tooltips to buttons
        create_tooltip(self.new_game_btn, "Start a new game (Ctrl+N)")
        create_tooltip(self.back_btn, "Go back one move (Ctrl+Z / Left arrow)")
        create_tooltip(self.forward_btn, "Go forward one move (Right arrow)")
        create_tooltip(self.flip_btn, "Flip the board orientation (F)")
        create_tooltip(self.load_btn, "Load a trained neural network")

        # Training panel (in Training tab)
        self.training_panel = TrainingPanel(
            training_tab,
            on_start=self._start_training,
            on_stop=self._stop_training,
        )
        self.training_panel.pack(fill="both", expand=True)

        # Network status bar
        status_bar = tk.Frame(main_frame, bg=COLORS["bg_secondary"])
        status_bar.pack(fill="x", pady=(15, 0))

        self.network_status = tk.Label(
            status_bar,
            text="No network loaded",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_muted"],
            font=FONTS["small"],
            padx=10,
            pady=5,
        )
        self.network_status.pack(side="left")

    def _load_network(self) -> None:
        """Load the neural network or pure MCTS player."""
        def get_history_length(network):
            """Detect history_length from network's input planes."""
            planes = network.num_input_planes
            if planes == 18:
                return 0
            return (planes - 6) // 12 - 1  # 54 planes = 3 history

        # Check for pure MCTS mode
        if self.use_pure_mcts:
            try:
                from alphazero.arena import PureMCTSPlayer

                self.pure_mcts_player = PureMCTSPlayer(
                    num_simulations=self.num_simulations,
                    verbose=self.verbose,
                    name="PureMCTS",
                )
                self.network_status.configure(
                    text=f"Pure MCTS ({self.num_simulations} simulations, no neural network)",
                    fg=COLORS["accent"],
                )
                return
            except Exception as e:
                self.network_status.configure(
                    text=f"Error creating pure MCTS: {e}",
                    fg=COLORS["error"],
                )
                return

        if self.network_path and os.path.exists(self.network_path):
            try:
                from alphazero import DualHeadNetwork, MCTS

                self.network = DualHeadNetwork.load(self.network_path)
                # Auto-detect history_length from loaded network
                self.history_length = get_history_length(self.network)
                self.mcts = MCTS(
                    network=self.network,
                    num_simulations=self.num_simulations,
                    batch_size=self.mcts_batch_size,
                    history_length=self.history_length,
                )
                self.network_status.configure(
                    text=f"Loaded: {os.path.basename(self.network_path)}",
                    fg=COLORS["success"],
                )
            except Exception as e:
                self.network_status.configure(
                    text=f"Error loading network: {e}",
                    fg=COLORS["error"],
                )
        else:
            # Create a new untrained network
            try:
                from alphazero import DualHeadNetwork, MCTS

                self.network = DualHeadNetwork()
                self.history_length = get_history_length(self.network)
                self.mcts = MCTS(
                    network=self.network,
                    num_simulations=self.num_simulations,
                    batch_size=self.mcts_batch_size,
                    history_length=self.history_length,
                )
                self.network_status.configure(
                    text="Using untrained network (random moves)",
                    fg=COLORS["warning"],
                )
            except Exception as e:
                self.network_status.configure(
                    text=f"Error creating network: {e}",
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
        elif self.game_mode == "human_vs_human":
            self.color_frame.pack_forget()
            self.board_widget.set_interactive(True)
        else:
            self.color_frame.pack(fill="x", pady=(10, 0))
            self.board_widget.set_interactive(True)

        self._new_game()

    def _on_color_change(self) -> None:
        """Handle player color change."""
        self.human_color = (
            chess.WHITE if self.color_var.get() == "white" else chess.BLACK
        )
        self.board_widget.set_player_color(self.human_color)

        # Flip board if playing as black
        if self.human_color == chess.BLACK and not self.board_widget.flipped:
            self.board_widget.flip()
        elif self.human_color == chess.WHITE and self.board_widget.flipped:
            self.board_widget.flip()

        self._new_game()

    def _new_game_with_confirm(self) -> None:
        """Start a new game with confirmation if game is in progress."""
        board = self.board_widget.get_board()

        # Ask for confirmation if game is in progress
        if len(board.move_stack) > 0 and not board.is_game_over():
            create_confirmation_dialog(
                self.root,
                "New Game",
                "Are you sure you want to start a new game? Current game will be lost.",
                on_confirm=self._new_game,
            )
        else:
            self._new_game()

    def _new_game(self) -> None:
        """Start a new game."""
        # Cancel any ongoing AI calculation
        self.ai_cancelled = True
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=0.5)

        self.ai_cancelled = False
        self.ai_thinking = False

        # Hide thinking indicator if visible
        self.thinking_indicator.stop()
        self.thinking_indicator.pack_forget()

        # Clear navigation history
        self.undone_moves.clear()

        self.board_widget.reset()
        self._update_game_info()

        if self.game_mode == "human_vs_ai":
            self.board_widget.set_player_color(self.human_color)
            # If AI plays white, make first move
            if self.human_color == chess.BLACK:
                self._trigger_ai_move()
        elif self.game_mode == "human_vs_human":
            # Both players are human, no color restriction
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

    def _back_one_move(self) -> None:
        """
        Go back to before your last move (for analysis/exploration).

        In Human vs AI: undoes your move + AI's response so you can try different moves.
        In AI vs AI / Human vs Human: undoes one move.
        """
        if self.ai_thinking or self.board_widget.is_animating:
            return

        board = self.board_widget.get_board()
        if not board.move_stack:
            return

        # Cancel AI if it's the AI's turn
        self.ai_cancelled = True

        # Animation duration for back/forward (2x faster than normal)
        anim_duration = 100

        if self.game_mode == "human_vs_ai":
            if board.turn == self.human_color:
                # It's your turn - AI already responded, undo both moves with animation
                if len(board.move_stack) >= 2:
                    ai_move = board.peek()
                    self.undone_moves.append((board.move_stack[-2], ai_move))

                    # Animate AI move in reverse first
                    def after_ai_undo():
                        board.pop()  # Remove AI move
                        human_move = board.peek()

                        # Now animate human move in reverse
                        def after_human_undo():
                            board.pop()  # Remove human move
                            self.board_widget.set_board(board)
                            self.board_widget.set_last_move(
                                board.peek() if board.move_stack else None
                            )
                            self._update_game_info()
                            self.status_label.configure(
                                text="Try a different move!",
                                fg=COLORS["accent"],
                            )

                        self.board_widget.animate_move(
                            human_move, duration_ms=anim_duration,
                            on_complete=after_human_undo, reverse=True
                        )

                    self.board_widget.animate_move(
                        ai_move, duration_ms=anim_duration,
                        on_complete=after_ai_undo, reverse=True
                    )
            else:
                # It's AI's turn - just undo your move
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
                        self.status_label.configure(
                            text="Try a different move!",
                            fg=COLORS["accent"],
                        )

                    self.board_widget.animate_move(
                        human_move, duration_ms=anim_duration,
                        on_complete=after_undo, reverse=True
                    )
        else:
            # AI vs AI / Human vs Human - just undo one move with animation
            last_move = board.peek()
            self.undone_moves.append((last_move, None))

            def after_undo():
                board.pop()
                self.board_widget.set_board(board)
                self.board_widget.set_last_move(
                    board.peek() if board.move_stack else None
                )
                self._update_game_info()
                self.status_label.configure(
                    text="Try a different move!",
                    fg=COLORS["accent"],
                )

            self.board_widget.animate_move(
                last_move, duration_ms=anim_duration,
                on_complete=after_undo, reverse=True
            )

    def _forward_one_move(self) -> None:
        """Go forward (replay the undone moves)."""
        if self.ai_thinking or self.board_widget.is_animating:
            return

        if not self.undone_moves:
            return

        board = self.board_widget.get_board()
        human_move, ai_move = self.undone_moves.pop()

        # Check if the human move is still legal
        if human_move not in board.legal_moves:
            self.undone_moves.clear()
            self.status_label.configure(
                text="Move no longer valid",
                fg=COLORS["warning"],
            )
            return

        # Animation duration for back/forward (2x faster than normal)
        anim_duration = 100

        # Animate the human move first
        def after_human_move():
            board.push(human_move)
            self.board_widget.set_board(board)

            # Play the AI move if there was one
            if ai_move and ai_move in board.legal_moves:
                def after_ai_move():
                    board.push(ai_move)
                    self.board_widget.set_board(board)
                    self.board_widget.set_last_move(ai_move)
                    self._update_game_info()

                    # If we're at the end and it's AI's turn, trigger AI
                    if (
                        self.game_mode == "human_vs_ai"
                        and board.turn != self.human_color
                        and not board.is_game_over()
                        and not self.undone_moves
                    ):
                        self._trigger_ai_move()

                self.board_widget.animate_move(
                    ai_move, duration_ms=anim_duration,
                    on_complete=after_ai_move
                )
            else:
                self.board_widget.set_last_move(human_move)
                self._update_game_info()

                # If there was supposed to be an AI move but it's not valid, trigger AI
                if ai_move and self.game_mode == "human_vs_ai":
                    self._trigger_ai_move()
                # If we're at the end and it's AI's turn, trigger AI
                elif (
                    self.game_mode == "human_vs_ai"
                    and board.turn != self.human_color
                    and not board.is_game_over()
                    and not self.undone_moves
                ):
                    self._trigger_ai_move()

        self.board_widget.animate_move(
            human_move, duration_ms=anim_duration,
            on_complete=after_human_move
        )

    def _flip_board(self) -> None:
        """Flip the board orientation."""
        self.board_widget.flip()

    def _on_human_move(self, move: chess.Move) -> None:
        """Handle human move."""
        # Clear forward history when making a new move
        self.undone_moves.clear()

        self._update_game_info()

        board = self.board_widget.get_board()
        if board.is_game_over():
            self._show_game_over()
        elif self.game_mode == "human_vs_ai":
            self._trigger_ai_move()
        # In human_vs_human mode, just wait for the next player's move

    def _trigger_ai_move(self) -> None:
        """Start AI move calculation in background thread."""
        if self.mcts is None and self.pure_mcts_player is None:
            return

        self.ai_thinking = True
        self.ai_cancelled = False

        # Show animated thinking indicator
        self.thinking_indicator.pack()
        self.thinking_indicator.start()

        self.ai_thread = threading.Thread(target=self._compute_ai_move, daemon=True)
        self.ai_thread.start()

    def _compute_ai_move(self) -> None:
        """Compute AI move in background thread."""
        try:
            board = self.board_widget.get_board()

            if board.is_game_over():
                return

            if self.use_pure_mcts and self.pure_mcts_player is not None:
                # Pure MCTS mode (no neural network)
                # Verbose output is handled inside select_move if verbose=True
                move = self.pure_mcts_player.select_move(board)
            else:
                # Neural network MCTS mode
                move = self.mcts.get_best_move(board, add_noise=False)

                # Print search tree if verbose mode is enabled
                if self.verbose and not self.ai_cancelled:
                    self.mcts.print_search_tree(board, top_n=5, max_depth=3)

            if not self.ai_cancelled and move:
                self.update_queue.put(("ai_move", move))

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
                    self.training_panel.update_progress(data)
                    # Show moves on board during self-play
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

    def _apply_ai_move(self, move: chess.Move) -> None:
        """Apply AI move to the board with animation."""
        self.ai_thinking = False

        # Hide thinking indicator
        self.thinking_indicator.stop()
        self.thinking_indicator.pack_forget()

        board = self.board_widget.get_board()
        if move in board.legal_moves:
            # Animate the move, then apply it after animation completes
            def on_animation_complete():
                board.push(move)
                self.board_widget.set_board(board)
                self.board_widget.set_last_move(move)
                self._update_game_info()

                updated_board = self.board_widget.get_board()
                if updated_board.is_game_over():
                    self._show_game_over()
                elif self.game_mode == "ai_vs_ai":
                    # Continue AI vs AI game
                    self.root.after(300, self._trigger_ai_move)

            # Start animation (300ms duration)
            self.board_widget.animate_move(move, duration_ms=300, on_complete=on_animation_complete)
        else:
            self._update_game_info()

    def _calculate_material(self, board: chess.Board) -> tuple[int, int]:
        """
        Calculate material for each side.

        Returns:
            Tuple of (white_material, black_material).
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

        white_material = 0
        black_material = 0

        for piece_type, value in piece_values.items():
            white_material += len(board.pieces(piece_type, chess.WHITE)) * value
            black_material += len(board.pieces(piece_type, chess.BLACK)) * value

        return white_material, black_material

    def _update_game_info(self) -> None:
        """Update game information display."""
        board = self.board_widget.get_board()

        # Turn
        turn_text = "White" if board.turn == chess.WHITE else "Black"
        self.turn_label.configure(text=turn_text)

        # Move count
        move_num = board.fullmove_number
        self.move_label.configure(text=str(move_num))

        # Material
        white_mat, black_mat = self._calculate_material(board)
        self.white_material_label.configure(text=str(white_mat))
        self.black_material_label.configure(text=str(black_mat))

        # Material advantage
        diff = white_mat - black_mat
        if diff > 0:
            self.material_advantage_label.configure(
                text=f"+{diff} White",
                fg=COLORS["text_primary"],
            )
        elif diff < 0:
            self.material_advantage_label.configure(
                text=f"+{abs(diff)} Black",
                fg=COLORS["text_secondary"],
            )
        else:
            self.material_advantage_label.configure(
                text="=",
                fg=COLORS["accent"],
            )

        # Status
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            self.status_label.configure(
                text=f"Checkmate! {winner} wins", fg=COLORS["accent"]
            )
        elif board.is_stalemate():
            self.status_label.configure(text="Stalemate!", fg=COLORS["warning"])
        elif board.is_insufficient_material():
            self.status_label.configure(
                text="Draw - Insufficient material", fg=COLORS["warning"]
            )
        elif board.is_check():
            self.status_label.configure(text="Check!", fg=COLORS["error"])
        elif self.ai_thinking:
            self.status_label.configure(text="AI thinking...", fg=COLORS["accent"])
        elif self.game_mode == "human_vs_ai" and board.turn == self.human_color:
            self.status_label.configure(text="Your turn", fg=COLORS["success"])
        elif self.game_mode == "human_vs_human":
            turn_text = "White" if board.turn == chess.WHITE else "Black"
            self.status_label.configure(text=f"{turn_text} to play", fg=COLORS["success"])
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
        if self.training_thread and self.training_thread.is_alive():
            return

        self.training_cancelled = False

        # Get config from training panel
        self.training_iterations = self.training_panel.get_iterations()
        self.training_output_file = self.training_panel.get_output_file()

        # Create trainer
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
            self.training_panel.reset()

    def _run_training(self) -> None:
        """Run training loop in background thread."""
        try:

            def callback(data):
                if not self.training_cancelled:
                    self.update_queue.put(("training", data))

            # Run all iterations
            for i in range(self.training_iterations):
                if self.training_cancelled:
                    break

                # Update iteration in callback data
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

                # Signal iteration complete
                self.update_queue.put(
                    (
                        "training",
                        {
                            "phase": "iteration_complete",
                            "iteration": i + 1,
                        },
                    )
                )

            # Save network to file
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

            # Update MCTS with trained network (using play mode settings)
            from alphazero import MCTS

            self.mcts = MCTS(
                network=self.network,
                num_simulations=self.num_simulations,
                batch_size=self.mcts_batch_size,
                history_length=self.history_length,
            )

        except Exception as e:
            self.update_queue.put(("error", f"Training error: {e}"))

    def _stop_training(self) -> None:
        """Stop training."""
        self.training_cancelled = True

    def run(self) -> None:
        """Run the application main loop."""
        # Center window on screen
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
        "--verbose", "-v", action="store_true", help="Print MCTS search tree after each AI move"
    )

    args = parser.parse_args()

    app = ChessGameApp(
        network_path=args.network,
        num_simulations=args.simulations,
        verbose=args.verbose,
    )
    app.run()


if __name__ == "__main__":
    main()
