"""
Visual match application for watching two networks play against each other.
With full MCTS analysis stats for both AIs.
"""

import tkinter as tk
import threading
import queue
import time
from typing import Any

import torch

try:
    import chess
except ImportError:
    raise ImportError("python-chess is required")

from .board_widget import ChessBoardWidget
from .styles import COLORS, FONTS, create_tooltip
from .components.mcts_panel import MCTSPanel
from .components.search_tree_panel import SearchTreePanel
from .components.eval_graph import EvalGraph
from src.chess_encoding.board_utils import get_material_count


class MatchApp(object):
    """Application for watching network vs network matches with full MCTS stats."""

    def __init__(
        self,
        network1_path: str,
        network2_path: str,
        num_games: int = 10,
        num_simulations: int = 200,
        move_delay: float = 0.0,
    ) -> None:
        self.network1_path: str = network1_path
        self.network2_path: str = network2_path
        self.num_games: int = num_games
        self.num_simulations: int = num_simulations
        self.move_delay: float = move_delay

        # Results tracking
        self.results: dict[str, int] = {"player1": 0, "player2": 0, "draws": 0}
        self.current_game: int = 0
        self.game_log: list[str] = []

        # Eval history for graphs (one per player perspective)
        self.eval_history_p1: list[float] = []
        self.eval_history_p2: list[float] = []

        # Threading
        self.move_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.running: bool = False
        self.paused: bool = False
        self.match_thread: threading.Thread | None = None

        # Names (extracted from paths, or special player names)
        import os

        def get_player_name(path: str) -> str:
            """Extract a display name from a player path."""
            if path.lower() == "random":
                return "Random"
            elif path.lower() == "mcts":
                return "PureMCTS"
            else:
                return os.path.splitext(os.path.basename(path))[0]

        self.name1: str = get_player_name(network1_path)
        self.name2: str = get_player_name(network2_path)

        # Stats panel components (set by _create_stats_panel via _setup_ui)
        self.player1_color_indicator: tk.Canvas
        self.player1_score_label: tk.Label
        self.player1_mcts_panel: MCTSPanel
        self.player1_search_tree: SearchTreePanel
        self.player1_eval_graph: EvalGraph
        self.player1_value_label: tk.Label
        self.player2_color_indicator: tk.Canvas
        self.player2_score_label: tk.Label
        self.player2_mcts_panel: MCTSPanel
        self.player2_search_tree: SearchTreePanel
        self.player2_eval_graph: EvalGraph
        self.player2_value_label: tk.Label

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the main UI with 3-column layout."""
        self.root: tk.Tk = tk.Tk()
        self.root.title(f"NeuralMate Match: {self.name1} vs {self.name2}")
        self.root.configure(bg=COLORS["bg_primary"])
        self.root.resizable(True, True)
        self.root.geometry("1920x1080")
        self.root.minsize(1600, 900)

        # Main container with 3 columns
        main_frame: tk.Frame = tk.Frame(self.root, bg=COLORS["bg_primary"])
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Configure grid columns
        # Left stats
        main_frame.columnconfigure(0, weight=1, minsize=450)
        # Center board
        main_frame.columnconfigure(1, weight=0, minsize=750)
        # Right stats
        main_frame.columnconfigure(2, weight=1, minsize=450)
        main_frame.rowconfigure(0, weight=1)

        # Left panel: Player 1 stats
        left_panel: tk.Frame = tk.Frame(main_frame, bg=COLORS["bg_secondary"])
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self._create_stats_panel(left_panel, "player1", self.name1)

        # Center panel: Board + Controls
        center_panel: tk.Frame = tk.Frame(main_frame, bg=COLORS["bg_primary"])
        center_panel.grid(row=0, column=1, sticky="ns", padx=10)
        self._create_center_panel(center_panel)

        # Right panel: Player 2 stats
        right_panel: tk.Frame = tk.Frame(main_frame, bg=COLORS["bg_secondary"])
        right_panel.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        self._create_stats_panel(right_panel, "player2", self.name2)

        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_stats_panel(self, parent: tk.Frame, player_id: str, name: str) -> None:
        """Create a stats panel for one player."""
        # Header with name and score
        header_frame: tk.Frame = tk.Frame(parent, bg=COLORS["bg_tertiary"])
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        # Color indicator
        color_indicator: tk.Canvas = tk.Canvas(
            header_frame,
            width=24,
            height=24,
            bg=COLORS["bg_tertiary"],
            highlightthickness=0,
        )
        color_indicator.pack(side=tk.LEFT, padx=(10, 5))
        setattr(self, f"{player_id}_color_indicator", color_indicator)

        # Name
        name_label: tk.Label = tk.Label(
            header_frame,
            text=name,
            font=FONTS["title"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_tertiary"],
        )
        name_label.pack(side=tk.LEFT, padx=5)

        # Score (W-D-L format)
        score_label: tk.Label = tk.Label(
            header_frame,
            text="W:0 D:0 L:0",
            font=FONTS["body_bold"],
            fg=COLORS["accent"],
            bg=COLORS["bg_tertiary"],
        )
        score_label.pack(side=tk.RIGHT, padx=10)
        setattr(self, f"{player_id}_score_label", score_label)

        # MCTS Analysis Panel
        mcts_frame: tk.Frame = tk.Frame(parent, bg=COLORS["bg_secondary"])
        mcts_frame.pack(fill=tk.X, padx=10, pady=5)

        mcts_panel: MCTSPanel = MCTSPanel(mcts_frame)
        mcts_panel.pack(fill=tk.X)
        setattr(self, f"{player_id}_mcts_panel", mcts_panel)

        # Search Tree Panel
        tree_frame: tk.Frame = tk.Frame(parent, bg=COLORS["bg_secondary"])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        search_tree: SearchTreePanel = SearchTreePanel(tree_frame)
        search_tree.pack(fill=tk.BOTH, expand=True)
        setattr(self, f"{player_id}_search_tree", search_tree)

        # Eval Graph
        eval_frame: tk.Frame = tk.Frame(parent, bg=COLORS["bg_secondary"])
        eval_frame.pack(fill=tk.X, padx=10, pady=5)

        eval_graph: EvalGraph = EvalGraph(eval_frame, height=120)
        eval_graph.pack(fill=tk.X)
        setattr(self, f"{player_id}_eval_graph", eval_graph)

        # Value label
        value_frame: tk.Frame = tk.Frame(parent, bg=COLORS["bg_secondary"])
        value_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        value_label: tk.Label = tk.Label(
            value_frame,
            text="Value: --",
            font=FONTS["mono"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_secondary"],
        )
        value_label.pack()
        setattr(self, f"{player_id}_value_label", value_label)

    def _create_center_panel(self, parent: tk.Frame) -> None:
        """Create the center panel with board and controls."""
        # Board
        board_frame: tk.Frame = tk.Frame(parent, bg=COLORS["bg_primary"])
        board_frame.pack(pady=(0, 10))

        self.board_widget: ChessBoardWidget = ChessBoardWidget(board_frame, size=700)
        self.board_widget.interactive = False
        self.board_widget.pack()

        # Game info
        info_frame: tk.Frame = tk.Frame(
            parent, bg=COLORS["bg_tertiary"], padx=15, pady=10
        )
        info_frame.pack(fill=tk.X, pady=5)

        # Game counter
        self.game_label: tk.Label = tk.Label(
            info_frame,
            text=f"Game 0/{self.num_games}",
            font=FONTS["body_bold"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_tertiary"],
        )
        self.game_label.pack()

        # Move counter
        self.move_label: tk.Label = tk.Label(
            info_frame,
            text="Move: 0",
            font=FONTS["body"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_tertiary"],
        )
        self.move_label.pack()

        # Status
        self.status_label: tk.Label = tk.Label(
            info_frame,
            text="Ready",
            font=FONTS["body_bold"],
            fg=COLORS["warning"],
            bg=COLORS["bg_tertiary"],
        )
        self.status_label.pack(pady=5)

        # Material display
        self.material_label: tk.Label = tk.Label(
            info_frame,
            text="Material: 39 - 39 (=)",
            font=FONTS["mono"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_tertiary"],
        )
        self.material_label.pack()

        # Controls
        controls_frame: tk.Frame = tk.Frame(parent, bg=COLORS["bg_primary"])
        controls_frame.pack(fill=tk.X, pady=10)

        self.start_btn: tk.Button = tk.Button(
            controls_frame,
            text="Start Match",
            font=FONTS["body"],
            command=self._start_match,
            bg=COLORS["success"],
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=8,
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.pause_btn: tk.Button = tk.Button(
            controls_frame,
            text="Pause",
            font=FONTS["body"],
            command=self._toggle_pause,
            bg=COLORS["warning"],
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=8,
            state=tk.DISABLED,
        )
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn: tk.Button = tk.Button(
            controls_frame,
            text="Stop",
            font=FONTS["body"],
            command=self._stop_match,
            bg=COLORS["error"],
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=8,
            state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Speed control
        speed_frame: tk.Frame = tk.Frame(parent, bg=COLORS["bg_primary"])
        speed_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            speed_frame,
            text="Delay:",
            font=FONTS["body"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_primary"],
        ).pack(side=tk.LEFT)

        self.speed_var: tk.DoubleVar = tk.DoubleVar(value=0.0)
        speed_scale: tk.Scale = tk.Scale(
            speed_frame,
            from_=0.0,
            to=2.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            bg=COLORS["bg_primary"],
            fg=COLORS["text_primary"],
            highlightthickness=0,
            length=200,
        )
        speed_scale.pack(side=tk.LEFT, padx=10)

        tk.Label(
            speed_frame,
            text="sec",
            font=FONTS["body"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_primary"],
        ).pack(side=tk.LEFT)

        # Tooltips
        create_tooltip(self.start_btn, "Start the match between networks")
        create_tooltip(self.pause_btn, "Pause/Resume the match")
        create_tooltip(self.stop_btn, "Stop the match")

    def _update_color_indicator(self, player_id: str, color: str) -> None:
        """Update the color indicator for a player."""
        indicator: tk.Canvas = getattr(self, f"{player_id}_color_indicator")
        indicator.delete("all")

        if color == "W":
            indicator.create_oval(3, 3, 21, 21, fill="white", outline="black", width=1)
        else:
            indicator.create_oval(3, 3, 21, 21, fill="black", outline="white", width=1)

    def _update_scores(self) -> None:
        """Update score labels for both players."""
        w1: int = self.results["player1"]
        l1: int = self.results["player2"]
        w2: int = self.results["player2"]
        l2: int = self.results["player1"]
        d: int = self.results["draws"]

        self.player1_score_label.configure(text=f"W:{w1} D:{d} L:{l1}")
        self.player2_score_label.configure(text=f"W:{w2} D:{d} L:{l2}")

    def _update_stats_panel(
        self, player_id: str, stats: dict[str, Any], board: chess.Board
    ) -> None:
        """Update the stats panel for a player."""
        if not stats:
            return

        # Get components
        mcts_panel: MCTSPanel = getattr(self, f"{player_id}_mcts_panel")
        search_tree: SearchTreePanel = getattr(self, f"{player_id}_search_tree")
        eval_graph: EvalGraph = getattr(self, f"{player_id}_eval_graph")
        value_label: tk.Label = getattr(self, f"{player_id}_value_label")

        # Update MCTS panel
        mcts_stats: list[dict[str, Any]] | None = stats.get("mcts_stats")
        total_visits: int = stats.get("total_visits", 0)
        if mcts_stats:
            mcts_panel.update_stats(board, mcts_stats, total_visits)

        # Update search tree
        tree_data: list[dict[str, Any]] | None = stats.get("tree_data")
        tree_depth: int = stats.get("tree_depth", 0)
        tree_branching: int = stats.get("tree_branching", 0)
        mate_in: int | None = stats.get("mate_in")
        if tree_data:
            search_tree.update_tree(tree_data, tree_depth, tree_branching, mate_in)

        # Update value label
        root_value: float | None = stats.get("root_value")
        if root_value is not None:
            value_color: str = (
                COLORS["q_value_positive"]
                if root_value >= 0
                else COLORS["q_value_negative"]
            )
            value_label.configure(text=f"Value: {root_value:+.3f}", fg=value_color)

            # Add to eval history and update graph
            if player_id == "player1":
                self.eval_history_p1.append(root_value)
                eval_graph.set_history(self.eval_history_p1)
            else:
                self.eval_history_p2.append(root_value)
                eval_graph.set_history(self.eval_history_p2)

    def _clear_stats_panels(self) -> None:
        """Clear all stats panels for a new game."""
        for player_id in ["player1", "player2"]:
            mcts_panel: MCTSPanel = getattr(self, f"{player_id}_mcts_panel")
            search_tree: SearchTreePanel = getattr(self, f"{player_id}_search_tree")
            eval_graph: EvalGraph = getattr(self, f"{player_id}_eval_graph")
            value_label: tk.Label = getattr(self, f"{player_id}_value_label")

            mcts_panel.clear()
            search_tree.clear()
            eval_graph.clear()
            value_label.configure(text="Value: --", fg=COLORS["text_primary"])

        # Clear eval histories
        self.eval_history_p1 = []
        self.eval_history_p2 = []

    @staticmethod
    def _calculate_material(board: chess.Board) -> tuple[int, int]:
        """Calculate material for each side."""
        return get_material_count(board, chess.WHITE), get_material_count(
            board, chess.BLACK
        )

    def _update_material_display(self, board: chess.Board) -> None:
        """Update material display label."""
        white_mat: int
        black_mat: int
        white_mat, black_mat = self._calculate_material(board)
        diff: int = white_mat - black_mat
        adv_str: str
        if diff > 0:
            adv_str = f"+{diff} W"
        elif diff < 0:
            adv_str = f"+{abs(diff)} B"
        else:
            adv_str = "="
        self.material_label.configure(
            text=f"Material: {white_mat} - {black_mat} ({adv_str})"
        )

    def _start_match(self) -> None:
        """Start the match in a background thread."""
        self.running = True
        self.paused = False
        self.start_btn.configure(state=tk.DISABLED)
        self.pause_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_label.configure(text="Loading networks...", fg=COLORS["warning"])

        self.match_thread = threading.Thread(target=self._run_match, daemon=True)
        self.match_thread.start()

        # Start UI update loop
        self._process_queue()

    def _toggle_pause(self) -> None:
        """Toggle pause state."""
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.configure(text="Resume")
            self.status_label.configure(text="Paused", fg=COLORS["warning"])
        else:
            self.pause_btn.configure(text="Pause")
            self.status_label.configure(text="Playing...", fg=COLORS["success"])

    def _stop_match(self) -> None:
        """Stop the match."""
        self.running = False
        self.status_label.configure(text="Stopped", fg=COLORS["error"])
        self.start_btn.configure(state=tk.NORMAL)
        self.pause_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.DISABLED)

    def _on_close(self) -> None:
        """Handle window close."""
        self.running = False
        self.root.destroy()

    def _process_queue(self) -> None:
        """Process updates from the match thread."""
        # Skip processing if animation is in progress
        if self.board_widget.is_animating:
            self.root.after(50, self._process_queue)
            return

        try:
            while True:
                msg: dict[str, Any] = self.move_queue.get_nowait()
                msg_type: str | None = msg.get("type")

                if msg_type == "board":
                    board: chess.Board = msg["board"]
                    last_move: chess.Move | None = msg.get("last_move")

                    if last_move is not None:
                        # Animate the move
                        board_before: chess.Board = board.copy()
                        board_before.pop()
                        self.board_widget.set_board(board_before)

                        def on_animation_complete(
                            _board: chess.Board = board,
                            _move: chess.Move = last_move,
                        ) -> None:
                            """Update board after move animation finishes."""
                            self.board_widget.set_board(_board)
                            self.board_widget.last_move = _move
                            self.board_widget._draw_board()
                            self._update_material_display(_board)

                        self.board_widget.animate_move(
                            last_move,
                            duration_ms=300,
                            on_complete=on_animation_complete,
                        )
                        break
                    else:
                        self.board_widget.set_board(board)
                        self.board_widget._draw_board()
                        self._update_material_display(board)

                elif msg_type == "move":
                    self.move_label.configure(text=f"Move: {msg['move_num']}")

                elif msg_type == "game_start":
                    self.game_label.configure(
                        text=f"Game {msg['game']}/{self.num_games}"
                    )
                    self._update_color_indicator("player1", msg.get("p1_color", "W"))
                    self._update_color_indicator("player2", msg.get("p2_color", "B"))
                    self.status_label.configure(text="Playing...", fg=COLORS["success"])
                    self._clear_stats_panels()

                elif msg_type == "game_end":
                    self._update_scores()

                elif msg_type == "match_end":
                    self.status_label.configure(
                        text="Match Complete", fg=COLORS["accent"]
                    )
                    self.start_btn.configure(state=tk.NORMAL)
                    self.pause_btn.configure(state=tk.DISABLED)
                    self.stop_btn.configure(state=tk.DISABLED)

                elif msg_type == "status":
                    self.status_label.configure(
                        text=msg["text"], fg=msg.get("color", COLORS["text_primary"])
                    )

                elif msg_type == "ai_stats":
                    # Update stats panel for the player who just moved
                    player_id: str = msg["player_id"]
                    stats: dict[str, Any] = msg["stats"]
                    board = msg["board"]
                    self._update_stats_panel(player_id, stats, board)

        except queue.Empty:
            pass

        if self.running:
            self.root.after(50, self._process_queue)

    def _run_match(self) -> None:
        """Run the match in background thread."""
        try:
            from alphazero import DualHeadNetwork
            from alphazero.arena import (
                NetworkPlayer,
                RandomPlayer,
                PureMCTSPlayer,
            )
            from alphazero.device import get_device

            device: torch.device = get_device()

            self.move_queue.put(
                {
                    "type": "status",
                    "text": f"Loading on {device}...",
                    "color": COLORS["warning"],
                }
            )

            # Helper to get history_length from network
            def get_history_length(network: DualHeadNetwork) -> int:
                """Compute history length from the network's input plane count."""
                planes: int = network.num_input_planes
                # 72 planes = (history_length + 1) * 12 + 24 (metadata + semantic + tactical)
                return (planes - 24) // 12 - 1

            # Load player 1
            history_length: int = 0
            player1: NetworkPlayer | RandomPlayer | PureMCTSPlayer
            if self.network1_path.lower() == "random":
                player1 = RandomPlayer(name="Random")
            elif self.network1_path.lower() == "mcts":
                player1 = PureMCTSPlayer(
                    num_simulations=self.num_simulations, name="PureMCTS"
                )
            else:
                network1: DualHeadNetwork = DualHeadNetwork.load(
                    self.network1_path, device=device
                )
                history_length = get_history_length(network1)
                player1 = NetworkPlayer(
                    network1,
                    num_simulations=self.num_simulations,
                    name=self.name1,
                    history_length=history_length,
                )

            # Load player 2
            player2: NetworkPlayer | RandomPlayer | PureMCTSPlayer
            if self.network2_path.lower() == "random":
                player2 = RandomPlayer(name="Random")
            elif self.network2_path.lower() == "mcts":
                player2 = PureMCTSPlayer(
                    num_simulations=self.num_simulations, name="PureMCTS"
                )
            else:
                network2: DualHeadNetwork = DualHeadNetwork.load(
                    self.network2_path, device=device
                )
                hl2: int = get_history_length(network2)
                history_length = max(history_length, hl2)
                player2 = NetworkPlayer(
                    network2,
                    num_simulations=self.num_simulations,
                    name=self.name2,
                    history_length=hl2,
                )

            self.move_queue.put(
                {
                    "type": "status",
                    "text": "Networks loaded",
                    "color": COLORS["success"],
                }
            )

            for game_num in range(self.num_games):
                if not self.running:
                    break

                self.current_game = game_num + 1

                # Alternate colors
                white: NetworkPlayer | RandomPlayer | PureMCTSPlayer
                black: NetworkPlayer | RandomPlayer | PureMCTSPlayer
                white_name: str
                black_name: str
                white_id: str
                black_id: str
                p1_color: str
                p2_color: str
                if game_num % 2 == 0:
                    white, black = player1, player2
                    white_name, black_name = self.name1, self.name2
                    white_id, black_id = "player1", "player2"
                    p1_color, p2_color = "W", "B"
                else:
                    white, black = player2, player1
                    white_name, black_name = self.name2, self.name1
                    white_id, black_id = "player2", "player1"
                    p1_color, p2_color = "B", "W"

                # Reset players
                white.reset()
                black.reset()

                self.move_queue.put(
                    {
                        "type": "game_start",
                        "game": self.current_game,
                        "p1_color": p1_color,
                        "p2_color": p2_color,
                    }
                )

                # Play the game
                board: chess.Board = chess.Board()
                move_count: int = 0

                self.move_queue.put(
                    {"type": "board", "board": board.copy(), "last_move": None}
                )

                while not board.is_game_over() and move_count < 200 and self.running:
                    # Wait while paused
                    while self.paused and self.running:
                        time.sleep(0.1)

                    if not self.running:
                        break

                    # Get current player info
                    current_player: NetworkPlayer | RandomPlayer | PureMCTSPlayer
                    current_id: str
                    if board.turn == chess.WHITE:
                        current_player = white
                        current_id = white_id
                    else:
                        current_player = black
                        current_id = black_id

                    # Get move
                    move: chess.Move | None = current_player.select_move(board)

                    if move is None:
                        break

                    # Send stats if available (NetworkPlayer has get_last_stats)
                    if isinstance(current_player, NetworkPlayer):
                        stats: dict[str, Any] = current_player.get_last_stats()
                        self.move_queue.put(
                            {
                                "type": "ai_stats",
                                "player_id": current_id,
                                "stats": stats,
                                "board": board.copy(),
                            }
                        )

                    board.push(move)
                    move_count += 1

                    self.move_queue.put(
                        {
                            "type": "board",
                            "board": board.copy(),
                            "last_move": move,
                        }
                    )
                    self.move_queue.put({"type": "move", "move_num": move_count})

                    # Delay between moves
                    time.sleep(self.speed_var.get())

                if not self.running:
                    break

                # Determine result
                winner: str
                termination: str
                if board.is_checkmate():
                    winner = "black" if board.turn == chess.WHITE else "white"
                    termination = "checkmate"
                elif board.is_stalemate():
                    winner = "draw"
                    termination = "stalemate"
                elif board.is_insufficient_material():
                    winner = "draw"
                    termination = "insufficient"
                elif board.can_claim_fifty_moves():
                    winner = "draw"
                    termination = "50 moves"
                elif board.can_claim_threefold_repetition():
                    winner = "draw"
                    termination = "repetition"
                elif move_count >= 200:
                    winner = "draw"
                    termination = "max moves"
                else:
                    winner = "draw"
                    termination = "unknown"

                # Update results
                winner_str: str
                if winner == "white":
                    if white_name == self.name1:
                        self.results["player1"] += 1
                    else:
                        self.results["player2"] += 1
                    winner_str = white_name
                elif winner == "black":
                    if black_name == self.name1:
                        self.results["player1"] += 1
                    else:
                        self.results["player2"] += 1
                    winner_str = black_name
                else:
                    self.results["draws"] += 1
                    winner_str = "Draw"

                self.move_queue.put({"type": "game_end"})

                # Console output
                print(
                    f"Game {self.current_game}/{self.num_games}: "
                    f"{white_name} (W) vs {black_name} (B) | "
                    f"{move_count} moves | {termination} | {winner_str}"
                )

            # Match complete
            if self.running:
                self.move_queue.put({"type": "match_end"})
                self.running = False

                # Print final summary
                print("\n" + "=" * 60)
                print("MATCH RESULTS")
                print("=" * 60)
                print(f"{self.name1}: {self.results['player1']} wins")
                print(f"{self.name2}: {self.results['player2']} wins")
                print(f"Draws: {self.results['draws']}")
                print("=" * 60)

        except Exception as e:
            self.move_queue.put(
                {"type": "status", "text": f"Error: {e}", "color": COLORS["error"]}
            )
            import traceback

            traceback.print_exc()

    def run(self) -> None:
        """Run the application."""
        self.root.mainloop()
