"""
Visual match application for watching two networks play against each other.
"""

import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
from typing import Any

try:
    import chess
except ImportError:
    raise ImportError("python-chess is required")

from .board_widget import ChessBoardWidget
from .styles import COLORS, FONTS, STATUS_ICONS, create_tooltip


class MatchApp:
    """Application for watching network vs network matches with visual board."""

    def __init__(
        self,
        network1_path: str,
        network2_path: str,
        num_games: int = 10,
        num_simulations: int = 200,
        move_delay: float = 0.0,
        no_adjudication: bool = False,
    ):
        self.network1_path = network1_path
        self.network2_path = network2_path
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.move_delay = move_delay
        self.no_adjudication = no_adjudication

        # Results tracking
        self.results = {"player1": 0, "player2": 0, "draws": 0}
        self.current_game = 0
        self.game_log = []

        # Threading
        self.move_queue = queue.Queue()
        self.running = False
        self.paused = False
        self.match_thread = None

        # Names (extracted from paths, or special player names)
        import os
        def get_player_name(path):
            if path.lower() == "random":
                return "Random"
            elif path.lower() == "mcts":
                return "PureMCTS"
            else:
                return os.path.splitext(os.path.basename(path))[0]
        self.name1 = get_player_name(network1_path)
        self.name2 = get_player_name(network2_path)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the main UI."""
        self.root = tk.Tk()
        self.root.title(f"NeuralMate Match: {self.name1} vs {self.name2}")
        self.root.configure(bg=COLORS["bg_primary"])
        self.root.resizable(True, True)
        self.root.geometry("1400x850")  # Default size for 1080p
        self.root.minsize(1100, 700)

        # Main container
        main_frame = tk.Frame(self.root, bg=COLORS["bg_primary"])
        main_frame.pack(padx=40, pady=30, fill="both", expand=True)

        # Left side: Board (larger for 1080p)
        board_frame = tk.Frame(main_frame, bg=COLORS["bg_primary"])
        board_frame.pack(side=tk.LEFT, padx=(0, 40))

        self.board_widget = ChessBoardWidget(board_frame, size=720)
        self.board_widget.interactive = False  # View only
        self.board_widget.pack()

        # Right side: Info panel (wider)
        info_frame = tk.Frame(main_frame, bg=COLORS["bg_secondary"], width=450)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_frame.pack_propagate(False)

        # Title
        title = tk.Label(
            info_frame,
            text="Network Match",
            font=FONTS["title"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_secondary"],
        )
        title.pack(pady=(15, 10))

        # Players frame
        players_frame = tk.Frame(info_frame, bg=COLORS["bg_secondary"])
        players_frame.pack(fill=tk.X, padx=15, pady=5)

        # Player 1
        p1_frame = tk.Frame(players_frame, bg=COLORS["bg_tertiary"], padx=10, pady=8)
        p1_frame.pack(fill=tk.X, pady=2)

        # Color indicator (canvas circle)
        self.p1_color_indicator = tk.Canvas(
            p1_frame, width=20, height=20,
            bg=COLORS["bg_tertiary"], highlightthickness=0
        )
        self.p1_color_indicator.pack(side=tk.LEFT)

        tk.Label(
            p1_frame, text=self.name1, font=FONTS["body_bold"],
            fg=COLORS["text_primary"], bg=COLORS["bg_tertiary"]
        ).pack(side=tk.LEFT, padx=5)

        self.p1_score_label = tk.Label(
            p1_frame, text="0", font=FONTS["body_bold"],
            fg=COLORS["accent"], bg=COLORS["bg_tertiary"]
        )
        self.p1_score_label.pack(side=tk.RIGHT)

        # Player 2
        p2_frame = tk.Frame(players_frame, bg=COLORS["bg_tertiary"], padx=10, pady=8)
        p2_frame.pack(fill=tk.X, pady=2)

        # Color indicator (canvas circle)
        self.p2_color_indicator = tk.Canvas(
            p2_frame, width=20, height=20,
            bg=COLORS["bg_tertiary"], highlightthickness=0
        )
        self.p2_color_indicator.pack(side=tk.LEFT)

        tk.Label(
            p2_frame, text=self.name2, font=FONTS["body_bold"],
            fg=COLORS["text_primary"], bg=COLORS["bg_tertiary"]
        ).pack(side=tk.LEFT, padx=5)

        self.p2_score_label = tk.Label(
            p2_frame, text="0", font=FONTS["body_bold"],
            fg=COLORS["accent"], bg=COLORS["bg_tertiary"]
        )
        self.p2_score_label.pack(side=tk.RIGHT)

        # Draws
        draws_frame = tk.Frame(players_frame, bg=COLORS["bg_tertiary"], padx=10, pady=8)
        draws_frame.pack(fill=tk.X, pady=2)

        tk.Label(
            draws_frame, text="Draws", font=FONTS["body"],
            fg=COLORS["text_secondary"], bg=COLORS["bg_tertiary"]
        ).pack(side=tk.LEFT, padx=(28, 5))

        self.draws_label = tk.Label(
            draws_frame, text="0", font=FONTS["body_bold"],
            fg=COLORS["text_secondary"], bg=COLORS["bg_tertiary"]
        )
        self.draws_label.pack(side=tk.RIGHT)

        # Game info
        game_frame = tk.Frame(info_frame, bg=COLORS["bg_secondary"])
        game_frame.pack(fill=tk.X, padx=15, pady=15)

        self.game_label = tk.Label(
            game_frame, text=f"Game 0/{self.num_games}",
            font=FONTS["body_bold"], fg=COLORS["text_primary"], bg=COLORS["bg_secondary"]
        )
        self.game_label.pack()

        self.move_label = tk.Label(
            game_frame, text="Move: 0",
            font=FONTS["body"], fg=COLORS["text_secondary"], bg=COLORS["bg_secondary"]
        )
        self.move_label.pack()

        self.status_label = tk.Label(
            game_frame, text="Ready",
            font=FONTS["body"], fg=COLORS["warning"], bg=COLORS["bg_secondary"]
        )
        self.status_label.pack(pady=5)

        # Material display
        material_frame = tk.Frame(game_frame, bg=COLORS["bg_secondary"])
        material_frame.pack(fill=tk.X, pady=5)

        self.material_label = tk.Label(
            material_frame, text="Material: 39 - 39 (=)",
            font=FONTS["mono"], fg=COLORS["text_secondary"], bg=COLORS["bg_secondary"]
        )
        self.material_label.pack()

        # Game log
        log_frame = tk.Frame(info_frame, bg=COLORS["bg_secondary"])
        log_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        tk.Label(
            log_frame, text="Game Log", font=FONTS["body_bold"],
            fg=COLORS["text_primary"], bg=COLORS["bg_secondary"]
        ).pack(anchor=tk.W)

        self.log_text = tk.Text(
            log_frame, height=12, width=50, font=("Consolas", 10),
            bg=COLORS["bg_tertiary"], fg=COLORS["text_primary"],
            relief=tk.FLAT, state=tk.DISABLED, wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Add scrollbar
        log_scroll = tk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=log_scroll.set)

        # Controls
        controls_frame = tk.Frame(info_frame, bg=COLORS["bg_secondary"])
        controls_frame.pack(fill=tk.X, padx=15, pady=15)

        self.start_btn = tk.Button(
            controls_frame, text="Start Match", font=FONTS["body"],
            command=self._start_match, bg=COLORS["success"], fg="white",
            relief=tk.FLAT, padx=15, pady=5
        )
        self.start_btn.pack(side=tk.LEFT, padx=2)

        self.pause_btn = tk.Button(
            controls_frame, text="Pause", font=FONTS["body"],
            command=self._toggle_pause, bg=COLORS["warning"], fg="white",
            relief=tk.FLAT, padx=15, pady=5, state=tk.DISABLED
        )
        self.pause_btn.pack(side=tk.LEFT, padx=2)

        self.stop_btn = tk.Button(
            controls_frame, text="Stop", font=FONTS["body"],
            command=self._stop_match, bg=COLORS["error"], fg="white",
            relief=tk.FLAT, padx=15, pady=5, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        # Add tooltips
        create_tooltip(self.start_btn, "Start the match between networks")
        create_tooltip(self.pause_btn, "Pause/Resume the match")
        create_tooltip(self.stop_btn, "Stop the match")

        # Speed control
        speed_frame = tk.Frame(info_frame, bg=COLORS["bg_secondary"])
        speed_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        tk.Label(
            speed_frame, text="Speed:", font=FONTS["body"],
            fg=COLORS["text_secondary"], bg=COLORS["bg_secondary"]
        ).pack(side=tk.LEFT)

        self.speed_var = tk.DoubleVar(value=0.0)
        speed_scale = tk.Scale(
            speed_frame, from_=0.0, to=1.0, resolution=0.1,
            orient=tk.HORIZONTAL, variable=self.speed_var,
            bg=COLORS["bg_secondary"], fg=COLORS["text_primary"],
            highlightthickness=0, length=150
        )
        speed_scale.pack(side=tk.LEFT, padx=10)

        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _log(self, message: str):
        """Add message to log."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _update_scores(self):
        """Update score labels."""
        self.p1_score_label.configure(text=str(self.results["player1"]))
        self.p2_score_label.configure(text=str(self.results["player2"]))
        self.draws_label.configure(text=str(self.results["draws"]))

    def _update_color_indicators(self, p1_color: str, p2_color: str):
        """Update the visual color indicators for each player."""
        # Clear previous indicators
        self.p1_color_indicator.delete("all")
        self.p2_color_indicator.delete("all")

        # Draw player 1 color indicator
        if p1_color == "W":
            # White circle with black border
            self.p1_color_indicator.create_oval(
                3, 3, 17, 17, fill="white", outline="black", width=1
            )
        else:
            # Black circle with white border
            self.p1_color_indicator.create_oval(
                3, 3, 17, 17, fill="black", outline="white", width=1
            )

        # Draw player 2 color indicator
        if p2_color == "W":
            self.p2_color_indicator.create_oval(
                3, 3, 17, 17, fill="white", outline="black", width=1
            )
        else:
            self.p2_color_indicator.create_oval(
                3, 3, 17, 17, fill="black", outline="white", width=1
            )

    def _start_match(self):
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

    def _toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.configure(text="Resume")
            self.status_label.configure(text="Paused", fg=COLORS["warning"])
        else:
            self.pause_btn.configure(text="Pause")
            self.status_label.configure(text="Playing...", fg=COLORS["success"])

    def _stop_match(self):
        """Stop the match."""
        self.running = False
        self.status_label.configure(text="Stopped", fg=COLORS["error"])
        self._log("Match stopped by user")
        self.start_btn.configure(state=tk.NORMAL)
        self.pause_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.DISABLED)

    def _on_close(self):
        """Handle window close."""
        self.running = False
        self.root.destroy()

    def _calculate_material(self, board) -> tuple[int, int]:
        """Calculate material for each side."""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }
        white_mat = sum(
            len(board.pieces(pt, chess.WHITE)) * val
            for pt, val in piece_values.items()
        )
        black_mat = sum(
            len(board.pieces(pt, chess.BLACK)) * val
            for pt, val in piece_values.items()
        )
        return white_mat, black_mat

    def _update_material_display(self, board):
        """Update material display label."""
        white_mat, black_mat = self._calculate_material(board)
        diff = white_mat - black_mat
        if diff > 0:
            adv_str = f"+{diff} W"
        elif diff < 0:
            adv_str = f"+{abs(diff)} B"
        else:
            adv_str = "="
        self.material_label.configure(text=f"Material: {white_mat} - {black_mat} ({adv_str})")

    def _process_queue(self):
        """Process updates from the match thread."""
        try:
            while True:
                msg = self.move_queue.get_nowait()
                msg_type = msg.get("type")

                if msg_type == "board":
                    board = msg["board"]
                    self.board_widget.set_board(board)
                    self.board_widget.last_move = msg.get("last_move")
                    self.board_widget._draw_board()
                    self._update_material_display(board)
                elif msg_type == "move":
                    self.move_label.configure(text=f"Move: {msg['move_num']}")
                elif msg_type == "game_start":
                    self.game_label.configure(text=f"Game {msg['game']}/{self.num_games}")
                    # Update visual color indicators
                    p1_color = msg.get("p1_color", "W")
                    p2_color = msg.get("p2_color", "B")
                    self._update_color_indicators(p1_color, p2_color)
                    self.status_label.configure(text="Playing...", fg=COLORS["success"])
                elif msg_type == "game_end":
                    self._log(msg["log"])
                    self._update_scores()
                elif msg_type == "match_end":
                    self.status_label.configure(text="Match Complete", fg=COLORS["accent"])
                    self._log(f"\n{'='*30}")
                    self._log(f"FINAL: {self.name1} {self.results['player1']} - {self.results['player2']} {self.name2}")
                    self._log(f"Draws: {self.results['draws']}")
                    self.start_btn.configure(state=tk.NORMAL)
                    self.pause_btn.configure(state=tk.DISABLED)
                    self.stop_btn.configure(state=tk.DISABLED)
                elif msg_type == "status":
                    self.status_label.configure(text=msg["text"], fg=msg.get("color", COLORS["text_primary"]))
                elif msg_type == "log":
                    self._log(msg["text"])

        except queue.Empty:
            pass

        if self.running:
            self.root.after(50, self._process_queue)

    def _run_match(self):
        """Run the match in background thread."""
        try:
            from alphazero import DualHeadResNet
            from alphazero.arena import Arena, NetworkPlayer, RandomPlayer, PureMCTSPlayer
            from alphazero.device import get_device

            device = get_device()

            self.move_queue.put({"type": "log", "text": f"Device: {device}"})

            # Helper to get history_length from network
            def get_history_length(network):
                planes = network.num_input_planes
                if planes == 18:
                    return 0
                return (planes - 6) // 12 - 1  # 54 planes = 3 history

            # Load player 1
            history_length = 0
            if self.network1_path.lower() == "random":
                player1 = RandomPlayer(name="Random")
                self.move_queue.put({"type": "log", "text": "Player 1: Random"})
            elif self.network1_path.lower() == "mcts":
                player1 = PureMCTSPlayer(num_simulations=self.num_simulations, name="PureMCTS")
                self.move_queue.put({"type": "log", "text": f"Player 1: Pure MCTS ({self.num_simulations} sims)"})
            else:
                self.move_queue.put({"type": "log", "text": f"Loading {self.name1}..."})
                network1 = DualHeadResNet.load(self.network1_path, device=device)
                history_length = get_history_length(network1)
                player1 = NetworkPlayer(
                    network1, num_simulations=self.num_simulations, name=self.name1,
                    history_length=history_length
                )

            # Load player 2
            if self.network2_path.lower() == "random":
                player2 = RandomPlayer(name="Random")
                self.move_queue.put({"type": "log", "text": "Player 2: Random"})
            elif self.network2_path.lower() == "mcts":
                player2 = PureMCTSPlayer(num_simulations=self.num_simulations, name="PureMCTS")
                self.move_queue.put({"type": "log", "text": f"Player 2: Pure MCTS ({self.num_simulations} sims)"})
            else:
                self.move_queue.put({"type": "log", "text": f"Loading {self.name2}..."})
                network2 = DualHeadResNet.load(self.network2_path, device=device)
                hl2 = get_history_length(network2)
                history_length = max(history_length, hl2)  # Use max of both
                player2 = NetworkPlayer(
                    network2, num_simulations=self.num_simulations, name=self.name2,
                    history_length=hl2
                )

            arena = Arena(
                num_games=self.num_games, num_simulations=self.num_simulations,
                max_moves=200, history_length=history_length
            )

            self.move_queue.put({"type": "log", "text": "Networks loaded. Starting match...\n"})

            for game_num in range(self.num_games):
                if not self.running:
                    break

                self.current_game = game_num + 1

                # Alternate colors
                if game_num % 2 == 0:
                    white, black = player1, player2
                    white_name, black_name = self.name1, self.name2
                    p1_color, p2_color = "W", "B"
                else:
                    white, black = player2, player1
                    white_name, black_name = self.name2, self.name1
                    p1_color, p2_color = "B", "W"

                self.move_queue.put({
                    "type": "game_start",
                    "game": self.current_game,
                    "p1_color": p1_color,
                    "p2_color": p2_color,
                })

                # Play the game
                board = chess.Board()
                move_count = 0

                self.move_queue.put({"type": "board", "board": board.copy(), "last_move": None})

                while not board.is_game_over() and move_count < 200 and self.running:
                    # Wait while paused
                    while self.paused and self.running:
                        time.sleep(0.1)

                    if not self.running:
                        break

                    # Get move
                    current_player = white if board.turn == chess.WHITE else black
                    move = current_player.select_move(board)

                    if move is None:
                        break

                    board.push(move)
                    move_count += 1

                    self.move_queue.put({
                        "type": "board",
                        "board": board.copy(),
                        "last_move": move,
                    })
                    self.move_queue.put({"type": "move", "move_num": move_count})

                    # Check adjudication (unless disabled)
                    adjudicated = False
                    adj_reason = ""
                    if not self.no_adjudication:
                        adjudicated, winner, adj_reason = arena._check_adjudication(board, move_count)
                        if adjudicated:
                            break

                    # Delay between moves
                    time.sleep(self.speed_var.get())

                if not self.running:
                    break

                # Determine result
                if board.is_checkmate():
                    winner = "black" if board.turn == chess.WHITE else "white"
                    termination = "checkmate"
                elif adjudicated:
                    termination = f"adj: {adj_reason}"
                elif board.is_stalemate():
                    winner = "draw"
                    termination = "stalemate"
                elif board.is_insufficient_material():
                    winner = "draw"
                    termination = "insufficient_material"
                elif board.can_claim_fifty_moves():
                    winner = "draw"
                    termination = "fifty_moves"
                elif board.can_claim_threefold_repetition():
                    winner = "draw"
                    termination = "threefold_repetition"
                elif move_count >= 200:
                    winner = "draw"
                    termination = "max_moves"
                else:
                    winner = "draw"
                    termination = "unknown"

                # Update results
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

                # Calculate material advantage for draws
                material_info = ""
                if winner_str == "Draw":
                    white_mat, black_mat = self._calculate_material(board)
                    diff = white_mat - black_mat
                    if diff > 0:
                        material_info = f" [W+{diff}]"
                    elif diff < 0:
                        material_info = f" [B+{abs(diff)}]"
                    else:
                        material_info = " [=]"

                log_msg = f"G{self.current_game}: {winner_str} ({termination}, {move_count} moves){material_info}"
                self.move_queue.put({"type": "game_end", "log": log_msg})

                # Console output
                print(f"Game {self.current_game}/{self.num_games}: {white_name} (W) vs {black_name} (B) | {move_count} moves | {termination} | {winner_str}{material_info}")

            # Match complete
            if self.running:
                self.move_queue.put({"type": "match_end"})
                self.running = False

                # Print final summary to console
                print("\n" + "=" * 60)
                print("MATCH RESULTS")
                print("=" * 60)
                print(f"{self.name1}: {self.results['player1']} wins")
                print(f"{self.name2}: {self.results['player2']} wins")
                print(f"Draws: {self.results['draws']}")
                print("=" * 60)

        except Exception as e:
            self.move_queue.put({"type": "log", "text": f"Error: {e}"})
            self.move_queue.put({"type": "status", "text": "Error", "color": COLORS["error"]})
            import traceback
            traceback.print_exc()

    def run(self):
        """Run the application."""
        # Auto-start the match after UI is ready
        self.root.after(500, self._start_match)
        self.root.mainloop()
