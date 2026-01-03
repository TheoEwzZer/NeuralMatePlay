"""
Visual training application for AlphaZero.

Provides a UI for watching self-play games in real-time during training.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import os
import json
import re

try:
    import chess
except ImportError:
    raise ImportError("python-chess is required. Install with: pip install python-chess")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .styles import COLORS, FONTS, apply_theme, create_styled_button, create_styled_label, create_tooltip
from .board_widget import ChessBoardWidget


class TrainingApp:
    """
    Visual training application.

    Shows self-play games on a chess board while training progresses.
    """

    def __init__(
        self,
        config_path: str | None = None,
        resume_checkpoint: str | None = None,
        iterations: int = 10,
        output_path: str = "alphazero_trained.pt",
    ):
        """
        Initialize the training app.

        Args:
            config_path: Path to training config JSON
            resume_checkpoint: Checkpoint to resume from
            iterations: Number of training iterations
            output_path: Where to save the trained network
        """
        self.config_path = config_path
        self.resume_checkpoint = resume_checkpoint
        self.iterations = iterations
        self.output_path = output_path

        self.root = tk.Tk()
        self.root.title("NeuralMate - Visual Training")
        self.root.resizable(True, True)
        self.root.geometry("1500x900")  # Default size for 1080p
        self.root.minsize(1200, 800)

        apply_theme(self.root)

        # Training state
        self.trainer = None
        self.network = None
        self.training_thread = None
        self.training_cancelled = False
        self.is_training = False

        # Update queue for thread-safe UI updates
        self.update_queue = queue.Queue()

        self._create_ui()
        self._start_update_loop()

    def _create_ui(self) -> None:
        """Create the UI layout."""
        # Main container (larger padding for 1080p)
        main_frame = tk.Frame(self.root, bg=COLORS["bg_primary"])
        main_frame.pack(padx=40, pady=30, fill="both", expand=True)

        # Title
        title_frame = tk.Frame(main_frame, bg=COLORS["bg_primary"])
        title_frame.pack(fill="x", pady=(0, 25))

        title = create_styled_label(
            title_frame,
            "Visual Training",
            style="title",
        )
        title.pack(side="left")

        subtitle = create_styled_label(
            title_frame,
            "AlphaZero Self-Play",
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
            size=680,  # Larger board for 1080p
            on_move=None,  # No interaction during training
        )
        self.board_widget.pack()
        self.board_widget.set_interactive(False)

        # Game info under board
        game_info_frame = tk.Frame(board_frame, bg=COLORS["bg_secondary"])
        game_info_frame.pack(fill="x", pady=(10, 0))

        self.game_label = create_styled_label(
            game_info_frame,
            "Game: - | Move: -",
            style="body",
            bg=COLORS["bg_secondary"],
        )
        self.game_label.pack(pady=(8, 4))

        # Game result label
        self.result_label = create_styled_label(
            game_info_frame,
            "",
            style="body_bold",
            bg=COLORS["bg_secondary"],
            fg=COLORS["accent"],
        )
        self.result_label.pack(pady=(0, 8))

        # Right side - Training info
        info_panel = tk.Frame(content_frame, bg=COLORS["bg_primary"], width=450)
        info_panel.pack(side="left", fill="both", expand=True)
        info_panel.pack_propagate(False)

        # Status section
        status_frame = tk.Frame(info_panel, bg=COLORS["bg_secondary"])
        status_frame.pack(fill="x", pady=(0, 10))

        status_title = create_styled_label(
            status_frame,
            "Training Status",
            style="heading",
            bg=COLORS["bg_secondary"],
        )
        status_title.pack(anchor="w", padx=15, pady=(15, 10))

        self.status_label = create_styled_label(
            status_frame,
            "Ready to start",
            style="body",
            bg=COLORS["bg_secondary"],
            fg=COLORS["success"],
        )
        self.status_label.pack(anchor="w", padx=15, pady=(0, 15))

        # Progress section
        progress_frame = tk.Frame(info_panel, bg=COLORS["bg_secondary"])
        progress_frame.pack(fill="x", pady=(0, 10))

        progress_title = create_styled_label(
            progress_frame,
            "Progress",
            style="heading",
            bg=COLORS["bg_secondary"],
        )
        progress_title.pack(anchor="w", padx=15, pady=(15, 10))

        # Iteration progress
        iter_frame = tk.Frame(progress_frame, bg=COLORS["bg_secondary"])
        iter_frame.pack(fill="x", padx=15, pady=5)

        tk.Label(
            iter_frame, text="Iteration:",
            bg=COLORS["bg_secondary"], fg=COLORS["text_secondary"],
            font=FONTS["body"],
        ).pack(side="left")

        self.iteration_label = tk.Label(
            iter_frame, text="0 / 0",
            bg=COLORS["bg_secondary"], fg=COLORS["text_primary"],
            font=FONTS["body_bold"],
        )
        self.iteration_label.pack(side="right")

        # Games progress
        games_frame = tk.Frame(progress_frame, bg=COLORS["bg_secondary"])
        games_frame.pack(fill="x", padx=15, pady=5)

        tk.Label(
            games_frame, text="Games:",
            bg=COLORS["bg_secondary"], fg=COLORS["text_secondary"],
            font=FONTS["body"],
        ).pack(side="left")

        self.games_label = tk.Label(
            games_frame, text="0 / 0",
            bg=COLORS["bg_secondary"], fg=COLORS["text_primary"],
            font=FONTS["body_bold"],
        )
        self.games_label.pack(side="right")

        # Examples count
        examples_frame = tk.Frame(progress_frame, bg=COLORS["bg_secondary"])
        examples_frame.pack(fill="x", padx=15, pady=5)

        tk.Label(
            examples_frame, text="Examples:",
            bg=COLORS["bg_secondary"], fg=COLORS["text_secondary"],
            font=FONTS["body"],
        ).pack(side="left")

        self.examples_label = tk.Label(
            examples_frame, text="0",
            bg=COLORS["bg_secondary"], fg=COLORS["text_primary"],
            font=FONTS["body_bold"],
        )
        self.examples_label.pack(side="right")

        # Buffer size
        buffer_frame = tk.Frame(progress_frame, bg=COLORS["bg_secondary"])
        buffer_frame.pack(fill="x", padx=15, pady=(5, 15))

        tk.Label(
            buffer_frame, text="Buffer:",
            bg=COLORS["bg_secondary"], fg=COLORS["text_secondary"],
            font=FONTS["body"],
        ).pack(side="left")

        self.buffer_label = tk.Label(
            buffer_frame, text="0",
            bg=COLORS["bg_secondary"], fg=COLORS["text_primary"],
            font=FONTS["body_bold"],
        )
        self.buffer_label.pack(side="right")

        # Statistics section
        stats_frame = tk.Frame(info_panel, bg=COLORS["bg_secondary"])
        stats_frame.pack(fill="x", pady=(0, 10))

        stats_title = create_styled_label(
            stats_frame,
            "Statistics",
            style="heading",
            bg=COLORS["bg_secondary"],
        )
        stats_title.pack(anchor="w", padx=15, pady=(15, 10))

        # Loss
        loss_frame = tk.Frame(stats_frame, bg=COLORS["bg_secondary"])
        loss_frame.pack(fill="x", padx=15, pady=5)

        tk.Label(
            loss_frame, text="Loss:",
            bg=COLORS["bg_secondary"], fg=COLORS["text_secondary"],
            font=FONTS["body"],
        ).pack(side="left")

        self.loss_label = tk.Label(
            loss_frame, text="-",
            bg=COLORS["bg_secondary"], fg=COLORS["text_primary"],
            font=FONTS["mono"],
        )
        self.loss_label.pack(side="right")

        # Win rates
        winrate_frame = tk.Frame(stats_frame, bg=COLORS["bg_secondary"])
        winrate_frame.pack(fill="x", padx=15, pady=5)

        tk.Label(
            winrate_frame, text="W/B/D:",
            bg=COLORS["bg_secondary"], fg=COLORS["text_secondary"],
            font=FONTS["body"],
        ).pack(side="left")

        self.winrate_label = tk.Label(
            winrate_frame, text="- / - / -",
            bg=COLORS["bg_secondary"], fg=COLORS["text_primary"],
            font=FONTS["mono"],
        )
        self.winrate_label.pack(side="right")

        # Elo
        elo_frame = tk.Frame(stats_frame, bg=COLORS["bg_secondary"])
        elo_frame.pack(fill="x", padx=15, pady=(5, 15))

        tk.Label(
            elo_frame, text="Elo:",
            bg=COLORS["bg_secondary"], fg=COLORS["text_secondary"],
            font=FONTS["body"],
        ).pack(side="left")

        self.elo_label = tk.Label(
            elo_frame, text="1500",
            bg=COLORS["bg_secondary"], fg=COLORS["accent"],
            font=FONTS["body_bold"],
        )
        self.elo_label.pack(side="right")

        # Control buttons
        btn_frame = tk.Frame(info_panel, bg=COLORS["bg_primary"])
        btn_frame.pack(fill="x", pady=(10, 0))

        self.start_btn = create_styled_button(
            btn_frame,
            "Start Training",
            command=self._start_training,
            style="accent",
        )
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.stop_btn = create_styled_button(
            btn_frame,
            "Stop",
            command=self._stop_training,
            style="outline",
        )
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=(5, 0))
        self.stop_btn.configure(state="disabled")

        # Add tooltips
        create_tooltip(self.start_btn, "Start AlphaZero self-play training")
        create_tooltip(self.stop_btn, "Stop training (network will be saved)")

        # Output path display
        output_frame = tk.Frame(info_panel, bg=COLORS["bg_primary"])
        output_frame.pack(fill="x", pady=(10, 0))

        output_label = create_styled_label(
            output_frame,
            f"Output: {self.output_path}",
            style="small",
            fg=COLORS["text_muted"],
        )
        output_label.pack(anchor="w")

    def _start_training(self) -> None:
        """Start training."""
        if self.is_training:
            return

        self.is_training = True
        self.training_cancelled = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="Initializing...", fg=COLORS["warning"])

        self.training_thread = threading.Thread(
            target=self._run_training,
            daemon=True,
        )
        self.training_thread.start()

    def _stop_training(self) -> None:
        """Stop training."""
        self.training_cancelled = True
        self.status_label.configure(text="Stopping...", fg=COLORS["warning"])

    def _run_training(self) -> None:
        """Run training in background thread."""
        try:
            from alphazero import DualHeadNetwork, AlphaZeroTrainer, TrainingConfig

            # Load or create network
            self.update_queue.put(("status", "Loading network..."))

            self.network = DualHeadNetwork()

            # Load config
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path) as f:
                    config_dict = json.load(f)
                training_config = config_dict.get("training", {})
                config = TrainingConfig(
                    games_per_iteration=training_config.get("games_per_iteration", 25),
                    num_simulations=training_config.get("num_simulations", 200),
                    max_moves=training_config.get("max_moves", 150),
                    epochs_per_iteration=training_config.get("epochs_per_iteration", 5),
                    batch_size=training_config.get("batch_size", 256),
                    learning_rate=training_config.get("learning_rate", 0.001),
                    lr_decay=training_config.get("lr_decay", 0.995),
                    min_learning_rate=training_config.get("min_learning_rate", 0.0001),
                    min_buffer_size=training_config.get("min_buffer_size", 2500),
                    recent_weight=training_config.get("recent_weight", 2.0),
                    arena_interval=training_config.get("arena_interval", 5),
                    arena_games=training_config.get("arena_games", 10),
                    arena_simulations=training_config.get("arena_simulations", 30),
                )
            else:
                config = TrainingConfig(
                    games_per_iteration=25,
                    num_simulations=200,
                    max_moves=150,
                    epochs_per_iteration=5,
                    batch_size=256,
                    min_buffer_size=2200,
                    arena_interval=5,
                )

            self.trainer = AlphaZeroTrainer(self.network, config)

            # Resume if requested
            if self.resume_checkpoint:
                self.update_queue.put(("status", f"Resuming from {self.resume_checkpoint}..."))
                checkpoint_dir = config.checkpoint_path

                if self.resume_checkpoint == "latest":
                    pattern = re.compile(r"iteration_(\d+)_network\.pt")
                    iterations = []
                    if os.path.exists(checkpoint_dir):
                        for filename in os.listdir(checkpoint_dir):
                            match = pattern.match(filename)
                            if match:
                                iterations.append(int(match.group(1)))
                    if iterations:
                        latest = max(iterations)
                        self.trainer.load_checkpoint(f"iteration_{latest}")
                else:
                    self.trainer.load_checkpoint(self.resume_checkpoint)

            # Training callback
            def callback(data):
                if self.training_cancelled:
                    return
                self.update_queue.put(("training", data))

            # Run training
            self.update_queue.put(("status", "Training..."))

            # Console header
            print("\n" + "=" * 50)
            print("NeuralMate - Visual Training")
            print("=" * 50)
            print(f"Config: {self.config_path or 'default'}")
            print(f"Iterations: {self.iterations}")
            print(f"Output: {self.output_path}")
            print(f"Games/iter: {config.games_per_iteration}")
            print(f"MCTS sims: {config.num_simulations}")
            print("=" * 50 + "\n")

            for i in range(self.iterations):
                if self.training_cancelled:
                    break

                print(f"\n--- Iteration {i + 1}/{self.iterations} ---")
                self.update_queue.put(("iteration", (i + 1, self.iterations)))
                self.trainer.train_iteration(callback=callback)

            # Save network
            if not self.training_cancelled:
                self.update_queue.put(("status", f"Saving to {self.output_path}..."))
                self.network.save(self.output_path)
                print(f"\n{'=' * 50}")
                print(f"Training complete! Saved to {self.output_path}")
                print(f"{'=' * 50}\n")
                self.update_queue.put(("complete", self.output_path))
            else:
                print("\nTraining stopped by user.")
                self.update_queue.put(("stopped", None))

        except Exception as e:
            self.update_queue.put(("error", str(e)))

    def _start_update_loop(self) -> None:
        """Start the UI update loop."""
        self._process_updates()

    def _process_updates(self) -> None:
        """Process queued updates from training thread."""
        try:
            while True:
                update_type, data = self.update_queue.get_nowait()

                if update_type == "status":
                    self.status_label.configure(text=data, fg=COLORS["warning"])

                elif update_type == "iteration":
                    current, total = data
                    self.iteration_label.configure(text=f"{current} / {total}")

                elif update_type == "training":
                    self._apply_training_update(data)

                elif update_type == "complete":
                    self.status_label.configure(
                        text=f"Complete! Saved to {data}",
                        fg=COLORS["success"]
                    )
                    self.is_training = False
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")

                elif update_type == "stopped":
                    self.status_label.configure(text="Stopped", fg=COLORS["text_muted"])
                    self.is_training = False
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")

                elif update_type == "error":
                    self.status_label.configure(text=f"Error: {data}", fg=COLORS["error"])
                    self.is_training = False
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")

        except queue.Empty:
            pass

        self.root.after(30, self._process_updates)

    def _apply_training_update(self, data: dict) -> None:
        """Apply training progress update."""
        phase = data.get("phase", "")

        if phase == "move":
            # Update board with current position
            fen = data.get("fen")
            game = data.get("game", 0)
            move = data.get("move", 0)
            if fen:
                try:
                    board = chess.Board(fen)
                    self.board_widget.set_board(board)
                    self.game_label.configure(text=f"Game: {game} | Move: {move}")
                    self.result_label.configure(text="")  # Clear previous result
                except Exception:
                    pass

        elif phase == "game_end":
            # Display game result
            winner = data.get("winner", "draw")
            termination = data.get("termination", "unknown")
            moves = data.get("moves", 0)
            game = data.get("game", 0)

            # Format result message
            term_names = {
                "checkmate": "checkmate",
                "stalemate": "stalemate",
                "insufficient": "insufficient material",
                "fifty_moves": "50-move rule",
                "repetition": "repetition",
                "max_moves": "max moves",
            }
            term_str = term_names.get(termination, termination)

            if winner == "white":
                result_text = f"White wins by {term_str}"
                color = COLORS["text_primary"]
            elif winner == "black":
                result_text = f"Black wins by {term_str}"
                color = COLORS["text_secondary"]
            else:
                result_text = f"Draw ({term_str})"
                color = COLORS["text_muted"]

            self.result_label.configure(text=result_text, fg=color)

            # Console log
            print(f"  Game {game}: {result_text} in {moves} moves")

        elif phase == "self_play":
            self.status_label.configure(text="Self-play...", fg=COLORS["warning"])
            games = data.get("games_played", 0)
            total = data.get("total_games", 0)
            examples = data.get("examples", 0)
            self.games_label.configure(text=f"{games} / {total}")
            self.examples_label.configure(text=str(examples))

            w = data.get("white_wins", 0)
            b = data.get("black_wins", 0)
            d = data.get("draws", 0)
            total_games = w + b + d
            if total_games > 0:
                self.winrate_label.configure(text=f"{w} / {b} / {d}")

        elif phase == "training":
            self.status_label.configure(text="Training network...", fg=COLORS["accent"])
            epoch = data.get("epoch", 0)
            epochs = data.get("epochs", 0)
            loss = data.get("total_loss", 0)
            if loss > 0:
                self.loss_label.configure(text=f"{loss:.4f}")
            if epoch == epochs and epochs > 0:
                print(f"  Training: Epoch {epoch}/{epochs} | Loss: {loss:.4f}")

        elif phase == "iteration_complete":
            stats = data.get("stats", {})
            buffer_size = stats.get("buffer_size", 0)
            self.buffer_label.configure(text=str(buffer_size))

            loss = stats.get("avg_total_loss", 0)
            if loss > 0:
                self.loss_label.configure(text=f"{loss:.4f}")

            # Console summary
            iteration = data.get("iteration", 0)
            sp_stats = stats.get("selfplay_stats", {})
            w = sp_stats.get("white_wins", 0)
            b = sp_stats.get("black_wins", 0)
            d = sp_stats.get("draws", 0)
            print(f"\n  Iteration {iteration} complete: Loss={loss:.4f} | W:{w} B:{b} D:{d} | Buffer: {buffer_size}\n")

        elif phase == "arena_complete":
            if self.trainer:
                elo = self.trainer.arena.get_elo()
                self.elo_label.configure(text=f"{elo:.0f}")
                print(f"  Arena: Elo = {elo:.0f}")

    def run(self) -> None:
        """Run the application."""
        # Center window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"+{x}+{y}")

        self.root.mainloop()
