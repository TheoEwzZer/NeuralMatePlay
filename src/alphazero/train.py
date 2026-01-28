"""
Self-play training script for AlphaZero.

Usage:
    ./neural_mate_train --config config.json
    ./neural_mate_train --iterations 100 --games 50
    ./neural_mate_train --checkpoint models/pretrained.pt --iterations 100
    ./neural_mate_train --resume-trained latest
    ./neural_mate_train --resume-trained 10 --iterations 50
    ./neural_mate_train --gui  # With GUI monitoring
"""

import argparse
import os
import sys
import threading
import queue
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphazero import DualHeadNetwork, AlphaZeroTrainer
from config import Config, TrainingConfig, generate_default_config


class TrainingGUI:
    """Non-blocking GUI for monitoring training progress with chess board."""

    def __init__(self):
        self.data_queue: queue.Queue = queue.Queue()
        self.running = True
        self.thread: Optional[threading.Thread] = None
        self.root = None
        self.board_widget = None

        # Training state
        self.iteration = 0
        self.total_iterations = 0
        self.current_game = 0
        self.move_number = 0

    def start(self, total_iterations: int):
        """Start GUI in a separate thread."""
        self.total_iterations = total_iterations
        self.thread = threading.Thread(target=self._run_gui, daemon=True)
        self.thread.start()

    def _run_gui(self):
        """Run the GUI main loop."""
        try:
            import tkinter as tk
            from tkinter import ttk
            import chess
        except ImportError:
            print("Warning: tkinter not available, GUI disabled")
            return

        # Import UI components
        try:
            from ui.styles import COLORS, FONTS, apply_theme, create_panel
            from ui.board_widget import ChessBoardWidget

            use_ui_module = True
        except ImportError:
            use_ui_module = False
            # Fallback colors
            COLORS = {
                "bg_primary": "#1a1a2e",
                "bg_secondary": "#16213e",
                "text_primary": "#ffffff",
                "text_secondary": "#a0a0a0",
                "accent": "#e94560",
                "success": "#0ead69",
                "warning": "#ffc107",
            }
            FONTS = {
                "title": ("Segoe UI", 20, "bold"),
                "heading": ("Segoe UI", 14, "bold"),
                "body": ("Segoe UI", 11),
                "body_bold": ("Segoe UI", 11, "bold"),
                "mono": ("Consolas", 11),
            }

        self.root = tk.Tk()
        self.root.title("NeuralMate2 Training Monitor")
        self.root.geometry("1100x700")
        self.root.configure(bg=COLORS["bg_primary"])
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if use_ui_module:
            apply_theme(self.root)

        # Main container
        main_frame = tk.Frame(self.root, bg=COLORS["bg_primary"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left side - Chess board
        left_frame = tk.Frame(main_frame, bg=COLORS["bg_primary"])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 20))

        board_title = tk.Label(
            left_frame,
            text="Self-Play",
            font=FONTS["heading"],
            bg=COLORS["bg_primary"],
            fg=COLORS["text_primary"],
        )
        board_title.pack(pady=(0, 10))

        if use_ui_module:
            self.board_widget = ChessBoardWidget(left_frame, size=520)
            self.board_widget.pack()
            self.board_widget.set_interactive(False)
        else:
            # Fallback canvas
            self.canvas = tk.Canvas(
                left_frame,
                width=520,
                height=520,
                bg=COLORS["bg_secondary"],
                highlightthickness=0,
            )
            self.canvas.pack()
            self._draw_fallback_board()

        # Game info
        info_frame = tk.Frame(left_frame, bg=COLORS["bg_secondary"])
        info_frame.pack(fill=tk.X, pady=(10, 0))

        self.game_label = tk.Label(
            info_frame,
            text="Game: 0 | Move: 0",
            font=FONTS["body"],
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
        )
        self.game_label.pack(pady=8)

        self.move_label = tk.Label(
            info_frame,
            text="",
            font=FONTS["body"],
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
        )
        self.move_label.pack(pady=(0, 8))

        # Right side - Stats
        right_frame = tk.Frame(main_frame, bg=COLORS["bg_primary"], width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right_frame.pack_propagate(False)

        title = tk.Label(
            right_frame,
            text="AlphaZero Training",
            font=FONTS["title"],
            bg=COLORS["bg_primary"],
            fg=COLORS["text_primary"],
        )
        title.pack(pady=(0, 20))

        # Progress section
        progress_panel = tk.Frame(right_frame, bg=COLORS["bg_secondary"])
        progress_panel.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            progress_panel,
            text="Progress",
            font=FONTS["heading"],
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
        ).pack(anchor=tk.W, padx=15, pady=(15, 10))

        self.iteration_label = tk.Label(
            progress_panel,
            text="Iteration: 0/0",
            font=FONTS["body_bold"],
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
        )
        self.iteration_label.pack(anchor=tk.W, padx=15, pady=5)

        self.progress_bar = ttk.Progressbar(
            progress_panel, length=350, mode="determinate"
        )
        self.progress_bar.pack(padx=15, pady=(5, 15))

        # Self-play section
        selfplay_panel = tk.Frame(right_frame, bg=COLORS["bg_secondary"])
        selfplay_panel.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            selfplay_panel,
            text="Self-Play",
            font=FONTS["heading"],
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
        ).pack(anchor=tk.W, padx=15, pady=(15, 10))

        self.selfplay_label = tk.Label(
            selfplay_panel,
            text="Games: 0/0",
            font=FONTS["body"],
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
        )
        self.selfplay_label.pack(anchor=tk.W, padx=15, pady=2)

        self.selfplay_stats = tk.Label(
            selfplay_panel,
            text="W: 0  B: 0  D: 0",
            font=FONTS["mono"],
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
        )
        self.selfplay_stats.pack(anchor=tk.W, padx=15, pady=(2, 15))

        # Training section
        training_panel = tk.Frame(right_frame, bg=COLORS["bg_secondary"])
        training_panel.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            training_panel,
            text="Training",
            font=FONTS["heading"],
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
        ).pack(anchor=tk.W, padx=15, pady=(15, 10))

        self.loss_label = tk.Label(
            training_panel,
            text="Loss: --",
            font=FONTS["body_bold"],
            bg=COLORS["bg_secondary"],
            fg=COLORS["accent"],
        )
        self.loss_label.pack(anchor=tk.W, padx=15, pady=2)

        self.policy_label = tk.Label(
            training_panel,
            text="Policy: --  |  Value: --",
            font=FONTS["mono"],
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
        )
        self.policy_label.pack(anchor=tk.W, padx=15, pady=2)

        self.lr_label = tk.Label(
            training_panel,
            text="LR: --",
            font=FONTS["mono"],
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
        )
        self.lr_label.pack(anchor=tk.W, padx=15, pady=(2, 15))

        # Arena section
        arena_panel = tk.Frame(right_frame, bg=COLORS["bg_secondary"])
        arena_panel.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            arena_panel,
            text="Arena",
            font=FONTS["heading"],
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
        ).pack(anchor=tk.W, padx=15, pady=(15, 10))

        self.elo_label = tk.Label(
            arena_panel,
            text="Elo: 1500",
            font=("Segoe UI", 16, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["success"],
        )
        self.elo_label.pack(anchor=tk.W, padx=15, pady=2)

        self.arena_status = tk.Label(
            arena_panel,
            text="",
            font=FONTS["body"],
            bg=COLORS["bg_secondary"],
            fg=COLORS["warning"],
        )
        self.arena_status.pack(anchor=tk.W, padx=15, pady=(2, 15))

        # Status at bottom
        status_frame = tk.Frame(right_frame, bg=COLORS["bg_primary"])
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)

        self.status_label = tk.Label(
            status_frame,
            text="Status: Initializing...",
            font=FONTS["body"],
            bg=COLORS["bg_primary"],
            fg=COLORS["text_secondary"],
        )
        self.status_label.pack(anchor=tk.W)

        note = tk.Label(
            status_frame,
            text="(Closing window won't stop training)",
            font=("Segoe UI", 9),
            bg=COLORS["bg_primary"],
            fg="#555555",
        )
        note.pack(anchor=tk.W)

        # Store chess module for later use
        self._chess = chess
        self._use_ui_module = use_ui_module

        # Schedule queue processing
        self.root.after(50, self._process_queue)
        self.root.mainloop()

    def _draw_fallback_board(self):
        """Draw a simple fallback board when UI module not available."""
        if not hasattr(self, "canvas"):
            return
        self.canvas.delete("all")
        size = 65
        light = "#ebecd0"
        dark = "#739552"
        for r in range(8):
            for c in range(8):
                color = light if (r + c) % 2 == 0 else dark
                self.canvas.create_rectangle(
                    c * size,
                    r * size,
                    (c + 1) * size,
                    (r + 1) * size,
                    fill=color,
                    outline="",
                )

    def _process_queue(self):
        """Process incoming data from training thread."""
        if not self.running or self.root is None:
            return

        try:
            while True:
                data = self.data_queue.get_nowait()
                self._update_display(data)
        except queue.Empty:
            pass

        if self.running and self.root:
            self.root.after(50, self._process_queue)

    def _update_display(self, data: dict):
        """Update GUI with new data."""
        phase = data.get("phase", "")

        if phase == "iteration_start":
            self.iteration = data.get("iteration", 0)
            self.iteration_label.config(
                text=f"Iteration: {self.iteration}/{self.total_iterations}"
            )
            progress = (
                (self.iteration / self.total_iterations) * 100
                if self.total_iterations > 0
                else 0
            )
            self.progress_bar["value"] = progress
            self.status_label.config(text="Status: Self-play...")

        elif phase == "self_play":
            games = data.get("games_played", 0)
            total = data.get("total_games", 0)
            w = data.get("white_wins", 0)
            b = data.get("black_wins", 0)
            d = data.get("draws", 0)
            self.selfplay_label.config(text=f"Games: {games}/{total}")
            self.selfplay_stats.config(text=f"W: {w}  B: {b}  D: {d}")
            self.current_game = games

        elif phase == "board_update":
            fen = data.get("fen", "")
            self.move_number = data.get("move_number", 0)
            move_san = data.get("move_san", "")

            if self._use_ui_module and self.board_widget and fen:
                try:
                    # Reconstruct full FEN for chess.Board
                    full_fen = fen + " w - - 0 1"
                    board = self._chess.Board(full_fen)
                    self.board_widget.set_board(board)

                    # Set last move highlight
                    last_move = data.get("last_move")
                    if last_move:
                        (from_row, from_col), (to_row, to_col) = last_move
                        from_sq = self._chess.square(from_col, 7 - from_row)
                        to_sq = self._chess.square(to_col, 7 - to_row)
                        move = self._chess.Move(from_sq, to_sq)
                        self.board_widget.set_last_move(move)
                except Exception:
                    pass

            self.game_label.config(
                text=f"Game: {self.current_game} | Move: {self.move_number}"
            )
            if move_san:
                self.move_label.config(text=f"Last: {move_san}")

        elif phase == "training":
            total_loss = data.get("total_loss", 0)
            policy_loss = data.get("policy_loss", 0)
            value_loss = data.get("value_loss", 0)
            self.loss_label.config(text=f"Loss: {total_loss:.4f}")
            self.policy_label.config(
                text=f"Policy: {policy_loss:.4f}  |  Value: {value_loss:.4f}"
            )
            self.status_label.config(text="Status: Training...")

        elif phase == "training_start":
            self.status_label.config(text="Status: Training...")

        elif phase == "iteration_complete":
            stats = data.get("stats", {})
            training_stats = stats.get("training_stats", {})
            final_loss = training_stats.get("avg_total_loss", 0)
            if final_loss > 0:
                self.loss_label.config(text=f"Loss: {final_loss:.4f}")
            lr = stats.get("learning_rate", 0)
            self.lr_label.config(text=f"LR: {lr:.6f}")
            self.status_label.config(text="Status: Iteration complete")

        elif phase == "arena_start":
            self.status_label.config(text="Status: Arena evaluation...")

        elif phase == "arena_complete":
            elo = data.get("elo", 1500)
            best_elo = data.get("best_elo", elo)
            is_new_best = data.get("is_new_best", False)
            self.elo_label.config(text=f"Elo: {elo:.0f} (best: {best_elo:.0f})")
            if is_new_best:
                self.arena_status.config(text="★ NEW BEST MODEL!")
            else:
                self.arena_status.config(text="")

        elif phase == "training_complete":
            self.status_label.config(text="Status: Training complete!")
            self.progress_bar["value"] = 100

    def _on_close(self):
        """Handle window close - don't stop training."""
        self.running = False
        if self.root:
            self.root.destroy()
            self.root = None

    def update(self, data: dict):
        """Send data to GUI (called from training thread)."""
        if self.running:
            self.data_queue.put(data)

    def stop(self):
        """Stop the GUI."""
        self.running = False
        if self.root:
            try:
                self.root.after(0, self.root.destroy)
            except Exception:
                pass


def main() -> int:
    """Main entry point for self-play training."""
    parser = argparse.ArgumentParser(
        description="AlphaZero self-play training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./neural_mate_train --config config.json
  ./neural_mate_train --config config.json --iterations 50 --games 200
  ./neural_mate_train --iterations 100
  ./neural_mate_train --checkpoint models/pretrained.pt --iterations 100
  ./neural_mate_train --resume-trained latest
  ./neural_mate_train --resume-trained 10 --iterations 50
  ./neural_mate_train --generate-config
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file",
    )

    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Print default config and exit",
    )

    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=None,
        help="Number of training iterations",
    )

    parser.add_argument(
        "--games",
        "-g",
        type=int,
        default=None,
        help="Games per iteration",
    )

    parser.add_argument(
        "--simulations",
        "-s",
        type=int,
        default=None,
        help="MCTS simulations per move",
    )

    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for checkpoints",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=None,
        help="Training batch size",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )

    parser.add_argument(
        "--arena-interval",
        type=int,
        default=None,
        help="Evaluate every N iterations",
    )

    parser.add_argument(
        "--arena-games",
        type=int,
        default=None,
        help="Number of arena games per evaluation",
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Replay buffer size",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs per iteration",
    )

    parser.add_argument(
        "--resume-trained",
        type=str,
        default=None,
        help="Resume from checkpoint: 'latest' or iteration number",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show GUI window to monitor training (closing window won't stop training)",
    )

    args = parser.parse_args()

    # Generate default config
    if args.generate_config:
        print(generate_default_config())
        return 0

    # Load config
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config.default()

    cfg = config.training

    # Override with CLI arguments
    if args.iterations:
        cfg.iterations = args.iterations
    if args.games:
        cfg.games_per_iteration = args.games
    if args.simulations:
        cfg.num_simulations = args.simulations
    if args.output:
        cfg.checkpoint_path = args.output
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.arena_interval:
        cfg.arena_interval = args.arena_interval
    if args.arena_games:
        cfg.arena_games = args.arena_games
    if args.buffer_size:
        cfg.buffer_size = args.buffer_size
    if args.epochs:
        cfg.epochs_per_iteration = args.epochs

    # Detect device
    import torch

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_str = f"GPU ({device_name})"
    else:
        device_str = "CPU"

    # Print config
    print("\n[Configuration]")
    print(f"  Device: {device_str}")
    print(f"  Iterations: {cfg.iterations}")
    print(f"  Games per iteration: {cfg.games_per_iteration}")
    print(f"  MCTS simulations: {cfg.num_simulations}")
    print(f"  MCTS batch size: {cfg.mcts_batch_size}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Checkpoint path: {cfg.checkpoint_path}")
    print(f"  Arena interval: {cfg.arena_interval}")
    print(f"  Buffer size: {cfg.buffer_size}")

    # Create or load network
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nLoading network from {args.checkpoint}")
        network = DualHeadNetwork.load(args.checkpoint)
    elif config.network:
        print("\nCreating network with config")
        network = DualHeadNetwork(
            num_filters=config.network.num_filters,
            num_residual_blocks=config.network.num_residual_blocks,
        )
    else:
        print("\nCreating new network")
        network = DualHeadNetwork()

    # Create trainer
    trainer = AlphaZeroTrainer(network, cfg)

    # Handle resume
    start_iteration = 0
    if args.resume_trained is not None:
        if args.resume_trained.lower() == "latest":
            # Load latest checkpoint
            if trainer.load_checkpoint():
                start_iteration = trainer._iteration
                print(f"Resumed from iteration {start_iteration}")
            else:
                print("No checkpoint found to resume from, starting fresh")
        else:
            # Load specific iteration
            try:
                iteration = int(args.resume_trained)
                if trainer.load_checkpoint(f"iteration_{iteration}"):
                    start_iteration = trainer._iteration
                    print(f"Resumed from iteration {start_iteration}")
                else:
                    print(f"Checkpoint iteration_{iteration} not found, starting fresh")
            except ValueError:
                print(f"Invalid resume value: {args.resume_trained}")
                print("Use 'latest' or an iteration number")

    # Initialize GUI if requested
    gui: Optional[TrainingGUI] = None
    if args.gui:
        gui = TrainingGUI()
        gui.start(cfg.iterations)
        print("GUI started (closing window won't stop training)")

    # Training state for callback
    iteration_stats = {"prev_loss": None, "last_phase": None}

    # Training callback
    def callback(data: dict) -> None:
        # Send to GUI if active
        if gui is not None:
            gui.update(data)
        phase = data.get("phase", "")
        last_phase = iteration_stats["last_phase"]
        iteration_stats["last_phase"] = phase

        if phase == "self_play":
            games = data.get("games_played", 0)
            total = data.get("total_games", 0)
            examples = data.get("examples", 0)
            avg_moves = data.get("avg_moves", 0)
            avg_time = data.get("avg_time", 0)
            w = data.get("white_wins", 0)
            b = data.get("black_wins", 0)
            d = data.get("draws", 0)
            # Termination details
            mates = data.get("checkmates", 0)
            stales = data.get("stalemates", 0)
            max_mv = data.get("max_moves_reached", 0)

            # Calculate ETA for current iteration
            eta_str = ""
            if games > 0 and avg_time > 0:
                remaining_games = total - games
                eta_seconds = remaining_games * avg_time
                if eta_seconds > 0:
                    hours, remainder = divmod(int(eta_seconds), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    if hours > 0:
                        eta_str = f" | ETA: {hours}h{minutes:02d}m"
                    else:
                        eta_str = f" | ETA: {minutes}m{seconds:02d}s"

            line = (
                f"  Self-play: {games}/{total} | {examples} ex | "
                f"{avg_moves:.0f} moves | {avg_time:.1f}s | W:{w} B:{b} D:{d} | "
                f"#:{mates} S:{stales} M:{max_mv}{eta_str}"
            )
            # Pad to 140 chars to ensure clean overwrite
            print(f"\r{line:<140}", end="", flush=True)

        elif phase == "training_skip":
            reason = data.get("reason", "")
            print(f"\n  Skipping training: {reason}")

        elif phase == "checkpoint_skip":
            reason = data.get("reason", "")
            print(f"  [Checkpoint] Skipped: {reason}")

        elif phase == "training_start":
            # Print newline to preserve self-play line
            print()

        elif phase == "training":
            epoch = data.get("epoch", 0)
            epochs = data.get("epochs", 0)
            total_loss = data.get("total_loss", 0)
            policy_loss = data.get("policy_loss", 0)
            value_loss = data.get("value_loss", 0)
            kl_loss = data.get("kl_loss", 0)
            kl_weight = data.get("kl_weight", 0.1)
            if epochs > 0:
                if kl_loss > 0:
                    kl_str = f", kl: {kl_loss:.4f} w={kl_weight:.2f}"
                else:
                    kl_str = ""
                line = (
                    f"  Epoch {epoch}/{epochs} | Loss: {total_loss:.4f} "
                    f"(policy: {policy_loss:.4f}, value: {value_loss:.4f}{kl_str})"
                )
                print(f"\r{line:<80}", end="", flush=True)

        elif phase == "iteration_complete":
            stats = data.get("stats", {})
            training_stats = stats.get("training_stats", {})
            final_loss = training_stats.get("avg_total_loss", 0)
            final_policy = training_stats.get("avg_policy_loss", 0)
            final_value = training_stats.get("avg_value_loss", 0)
            final_kl = training_stats.get("avg_kl_loss", 0)
            buffer_size = stats.get("buffer_size", 0)
            lr = stats.get("learning_rate", 0)

            sp_stats = stats.get("selfplay_stats", {})
            avg_moves = sp_stats.get("avg_game_length", 0)
            total_games = sp_stats.get("games_played", 1)
            w_rate = sp_stats.get("white_wins", 0) / total_games * 100
            b_rate = sp_stats.get("black_wins", 0) / total_games * 100
            d_rate = sp_stats.get("draws", 0) / total_games * 100

            # Calculate trend
            trend = ""
            if iteration_stats["prev_loss"] is not None and final_loss > 0:
                diff = final_loss - iteration_stats["prev_loss"]
                if diff > 0.01:
                    trend = " ↑"
                elif diff < -0.01:
                    trend = " ↓"
                else:
                    trend = " ="
            iteration_stats["prev_loss"] = (
                final_loss if final_loss > 0 else iteration_stats["prev_loss"]
            )

            if final_loss > 0:
                kl_weight = training_stats.get("kl_weight", 0.1)
                if final_kl > 0:
                    kl_str = f", kl: {final_kl:.4f} w={kl_weight:.2f}"
                else:
                    kl_str = ""
                print(
                    f"\n  Final: {final_loss:.4f}{trend} "
                    f"(policy: {final_policy:.4f}, value: {final_value:.4f}{kl_str})"
                )
                print(f"  LR: {lr:.6f} | Buffer: {buffer_size}")
                avg_time = sp_stats.get("avg_game_time", 0)
                print(
                    f"  Games: {avg_moves:.0f} moves | {avg_time:.1f}s/game | "
                    f"W:{w_rate:.0f}% B:{b_rate:.0f}% D:{d_rate:.0f}%"
                )
            else:
                print()  # Just newline if no training happened

        elif phase == "arena_start":
            # Newline after training epochs before arena
            print()
            print("  Arena: Evaluating...", end="", flush=True)

        elif phase == "arena_match":
            opponent = data.get("opponent", "")
            # Update progress on same line
            print(
                f"\r  Arena: vs {opponent}...                              ",
                end="",
                flush=True,
            )

        elif phase == "arena_complete":
            elo = data.get("elo", 1500)
            best_elo = data.get("best_elo", elo)
            is_new_best = data.get("is_new_best", False)
            veto = data.get("veto", False)
            veto_reason = data.get("veto_reason")
            vs_random = data.get("vs_random", {})
            vs_best = data.get("vs_best")
            vs_old = data.get("vs_old")
            vs_pretrained = data.get("vs_pretrained")

            print("\n  ========== ARENA EVALUATION ==========")

            # vs Random
            if vs_random:
                w = vs_random.get("wins", 0)
                l = vs_random.get("losses", 0)
                d = vs_random.get("draws", 0)
                sc = vs_random.get("score", 0) * 100
                t = vs_random.get("avg_time", 0)
                print(f"  vs Random:  {w}W-{l}L-{d}D ({sc:.0f}% score) {t:.1f}s/game")

            # vs Best
            if vs_best:
                w = vs_best.get("wins", 0)
                l = vs_best.get("losses", 0)
                d = vs_best.get("draws", 0)
                sc = vs_best.get("score", 0.5) * 100
                best_iter = vs_best.get("from_iteration", "?")
                t = vs_best.get("avg_time", 0)
                print(
                    f"  vs Best #{best_iter}: {w}W-{l}L-{d}D ({sc:.0f}% score) {t:.1f}s/game"
                )

            # vs Old
            if vs_old:
                w = vs_old.get("wins", 0)
                l = vs_old.get("losses", 0)
                d = vs_old.get("draws", 0)
                sc = vs_old.get("score", 0.5) * 100
                old_iter = vs_old.get("from_iteration", "?")
                t = vs_old.get("avg_time", 0)
                print(
                    f"  vs Old #{old_iter}: {w}W-{l}L-{d}D ({sc:.0f}% score) {t:.1f}s/game"
                )

            # vs Pretrained
            if vs_pretrained:
                w = vs_pretrained.get("wins", 0)
                l = vs_pretrained.get("losses", 0)
                d = vs_pretrained.get("draws", 0)
                sc = vs_pretrained.get("score", 0.5) * 100
                t = vs_pretrained.get("avg_time", 0)
                print(
                    f"  vs Pretrained: {w}W-{l}L-{d}D ({sc:.0f}% score) {t:.1f}s/game"
                )

            # ELO
            print(f"  Elo: {elo:.0f} (best: {best_elo:.0f})")

            # Promotion status
            if is_new_best and not veto:
                print("  NEW BEST MODEL!")
            elif veto:
                print("  VETO: Catastrophic forgetting detected!")
                print(f"  {veto_reason}")

            print("  " + "=" * 38)

        elif phase == "veto_recovery":
            reason = data.get("reason", "")
            actions = data.get("actions", [])
            consecutive = data.get("consecutive_vetoes", 1)

            # Show escalation level with visual emphasis
            if consecutive >= 4:
                header = "VETO RECOVERY [ESCALATION L4 - CRITICAL]"
            elif consecutive >= 3:
                header = "VETO RECOVERY [ESCALATION L3]"
            elif consecutive >= 2:
                header = "VETO RECOVERY [ESCALATION L2]"
            else:
                header = "VETO RECOVERY"

            print(f"\n  ========== {header} ==========")
            print(f"  Reason: {reason}")
            print(f"  Consecutive vetoes: {consecutive}")
            for action in actions:
                print(f"  - {action}")
            print("  " + "=" * (len(header) + 22))

        elif phase == "kl_warning":
            kl_loss = data.get("kl_loss", 0)
            message = data.get("message", "")
            print(f"\n  [KL WARNING] {message} (KL={kl_loss:.4f})")

        elif phase == "kl_critical":
            kl_loss = data.get("kl_loss", 0)
            message = data.get("message", "")
            print(f"\n  [KL CRITICAL] {message} (KL={kl_loss:.4f})")

    # Run training
    remaining = cfg.iterations - start_iteration
    print(f"\nStarting training for {remaining} iterations (total: {cfg.iterations})")
    if start_iteration > 0:
        print(f"Resuming from iteration {start_iteration}")
    print()

    try:
        for i in range(start_iteration, cfg.iterations):
            print(f"=== Iteration {i + 1}/{cfg.iterations} ===")
            # Notify GUI of iteration start
            if gui is not None:
                gui.update({"phase": "iteration_start", "iteration": i + 1})
            trainer.train_iteration(callback)
            print()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        if gui is not None:
            gui.stop()
        return 1

    # Notify completion
    if gui is not None:
        gui.update({"phase": "training_complete"})
        gui.stop()

    print("Training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
