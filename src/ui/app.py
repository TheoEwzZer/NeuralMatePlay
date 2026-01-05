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

        # Evaluation history for graph
        self.eval_history: List[float] = []

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

        # Black player info (top, opponent when playing white)
        self.black_player_info = PlayerInfo(
            left_panel,
            color=chess.BLACK,
            name="AI",
        )
        self.black_player_info.pack(fill="x", pady=(0, 10))

        # Opening display
        self.opening_display = OpeningDisplay(left_panel)
        self.opening_display.pack(fill="x", pady=(0, 10))

        # Phase indicator
        self.phase_indicator = PhaseIndicator(left_panel)
        self.phase_indicator.pack(fill="x", pady=(0, 10))

        # Spacer
        spacer = tk.Frame(left_panel, bg=COLORS["bg_primary"])
        spacer.pack(fill="both", expand=True)

        # White player info (bottom, you when playing white)
        self.white_player_info = PlayerInfo(
            left_panel,
            color=chess.WHITE,
            name="You",
        )
        self.white_player_info.pack(fill="x", pady=(10, 0))

        # Game mode selection (collapsed panel)
        self._create_mode_panel(left_panel)

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
            if planes == 18:
                return 0
            return (planes - 6) // 12 - 1

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
                )
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
                )
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

        self.board_widget.reset()

        # Reset all analysis panels
        self.mcts_panel.clear()
        self.search_tree_panel.clear()
        self.eval_bar.clear()
        self.eval_graph.clear()
        self.value_label.configure(text="0.00", fg=COLORS["text_primary"])
        self.opening_display.clear()
        self.move_list.clear()

        self._update_game_info()

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

        self.board_widget.animate_move(
            human_move, duration_ms=anim_duration, on_complete=after_human_move
        )

    def _flip_board(self) -> None:
        """Flip the board orientation."""
        self.board_widget.flip()

    def _on_human_move(self, move: chess.Move) -> None:
        """Handle human move."""
        self.undone_moves.clear()
        self._update_game_info()

        # Tree reuse: preserve subtree for next AI search
        if self.mcts is not None:
            board = self.board_widget.get_board()
            self.mcts.advance_root(board)

        # Update evaluation
        self._update_eval_after_move()

        board = self.board_widget.get_board()
        if board.is_game_over():
            self._show_game_over()
        elif self.game_mode == "human_vs_ai":
            self._trigger_ai_move()

    def _trigger_ai_move(self) -> None:
        """Start AI move calculation in background thread."""
        if self.mcts is None and self.pure_mcts_player is None:
            return

        self.ai_thinking = True
        self.ai_cancelled = False

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

            root_value = 0.0
            mcts_stats = None
            tree_data = None
            tree_depth = 0
            tree_branching = 0
            mate_in = None

            if self.use_pure_mcts and self.pure_mcts_player is not None:
                move = self.pure_mcts_player.select_move(board)
            else:
                move = self.mcts.get_best_move(board, add_noise=False)

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

        # Update Search Tree panel
        if tree_data:
            self.search_tree_panel.update_tree(tree_data, tree_depth, tree_branching, mate_in)

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

            def on_animation_complete():
                board.push(move)
                self.board_widget.set_board(board)
                self.board_widget.set_last_move(move)
                self._update_game_info()

                # Tree reuse: preserve subtree for next search
                if self.mcts is not None:
                    self.mcts.advance_root(board)

                # Update evaluation after AI move
                self._update_eval_after_move(ai_value=root_value)

                updated_board = self.board_widget.get_board()
                if updated_board.is_game_over():
                    self._show_game_over()
                elif self.game_mode == "ai_vs_ai":
                    self.root.after(300, self._trigger_ai_move)

            self.board_widget.animate_move(
                move, duration_ms=300, on_complete=on_animation_complete
            )
        else:
            self._update_game_info()

    def _update_eval_after_move(self, ai_value: float = None) -> None:
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
        self.eval_graph.add_evaluation(display_eval)

    def _update_eval_for_position(self) -> None:
        """Update evaluation for current position (after back/forward)."""
        board = self.board_widget.get_board()
        move_index = len(board.move_stack)

        # Update eval bar and graph to current position
        if move_index < len(self.eval_history):
            # Trim eval history to current position
            self.eval_history = self.eval_history[:move_index]
            self.eval_graph.set_history(self.eval_history)

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
            )

        except Exception as e:
            self.update_queue.put(("error", f"Training error: {e}"))

    def _stop_training(self) -> None:
        """Stop training."""
        self.training_cancelled = True

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

    args = parser.parse_args()

    app = ChessGameApp(
        network_path=args.network,
        num_simulations=args.simulations,
        verbose=args.verbose,
    )
    app.run()


if __name__ == "__main__":
    main()
