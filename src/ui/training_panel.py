"""
Training panel for AlphaZero self-play visualization.

Provides a UI panel for monitoring and controlling self-play
training sessions.
"""

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable
import threading
import queue

from .styles import (
    COLORS,
    FONTS,
    create_styled_button,
    create_styled_label,
    ProgressBar,
    create_tooltip,
    STATUS_ICONS,
)


class TrainingPanel(tk.Frame):
    """
    Panel for monitoring and controlling AlphaZero training.

    Displays training progress, statistics, and provides controls
    for starting/stopping training.
    """

    def __init__(
        self,
        parent: tk.Widget,
        on_start: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        **kwargs,
    ):
        """
        Initialize the training panel.

        Args:
            parent: Parent widget
            on_start: Callback when training starts
            on_stop: Callback when training stops
        """
        super().__init__(parent, bg=COLORS["bg_secondary"], **kwargs)

        self.on_start = on_start
        self.on_stop = on_stop

        self.is_training = False
        self.update_queue = queue.Queue()

        self._create_widgets()
        self._start_update_loop()

    def _create_widgets(self) -> None:
        """Create panel widgets."""
        # Title
        title = create_styled_label(
            self,
            "Self-Play Training",
            style="heading",
            bg=COLORS["bg_secondary"],
        )
        title.pack(pady=(15, 10), padx=15, anchor="w")

        # Status frame
        status_frame = tk.Frame(self, bg=COLORS["bg_secondary"])
        status_frame.pack(fill="x", padx=15, pady=5)

        self.status_label = create_styled_label(
            status_frame,
            "Ready",
            style="body",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
        )
        self.status_label.pack(side="left")

        self.status_indicator = tk.Canvas(
            status_frame,
            width=12,
            height=12,
            bg=COLORS["bg_secondary"],
            highlightthickness=0,
        )
        self.status_indicator.pack(side="right")
        self._draw_status_indicator("idle")

        # Progress section
        progress_frame = tk.Frame(self, bg=COLORS["bg_secondary"])
        progress_frame.pack(fill="x", padx=15, pady=10)

        # Games progress
        games_label = create_styled_label(
            progress_frame,
            "Games:",
            style="small",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
        )
        games_label.pack(anchor="w")

        self.games_progress = ProgressBar(progress_frame, width=380, height=10)
        self.games_progress.pack(fill="x", pady=(2, 5))

        self.games_text = create_styled_label(
            progress_frame,
            "0 / 0",
            style="small",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_muted"],
        )
        self.games_text.pack(anchor="e")

        # Training progress
        train_label = create_styled_label(
            progress_frame,
            "Training:",
            style="small",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
        )
        train_label.pack(anchor="w", pady=(10, 0))

        self.train_progress = ProgressBar(progress_frame, width=380, height=10)
        self.train_progress.pack(fill="x", pady=(2, 5))

        self.train_text = create_styled_label(
            progress_frame,
            "Epoch 0 / 0",
            style="small",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_muted"],
        )
        self.train_text.pack(anchor="e")

        # Statistics section
        stats_frame = tk.LabelFrame(
            self,
            text=" Statistics ",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
        )
        stats_frame.pack(fill="x", padx=15, pady=10)

        # Stats grid
        self.stats = {}
        stats_data = [
            ("iteration", "Iteration:", "0"),
            ("examples", "Examples:", "0"),
            ("policy_loss", "Policy Loss:", "-"),
            ("value_loss", "Value Loss:", "-"),
            ("games_per_sec", "Games/sec:", "-"),
            ("win_rate", "Win Rate:", "-"),
        ]

        for i, (key, label_text, default) in enumerate(stats_data):
            row = i // 2
            col = (i % 2) * 2

            label = tk.Label(
                stats_frame,
                text=label_text,
                bg=COLORS["bg_secondary"],
                fg=COLORS["text_secondary"],
                font=FONTS["small"],
            )
            label.grid(row=row, column=col, sticky="w", padx=(10, 5), pady=3)

            value = tk.Label(
                stats_frame,
                text=default,
                bg=COLORS["bg_secondary"],
                fg=COLORS["text_primary"],
                font=FONTS["mono_small"],
            )
            value.grid(row=row, column=col + 1, sticky="e", padx=(0, 10), pady=3)

            self.stats[key] = value

        # Configuration section
        config_frame = tk.LabelFrame(
            self,
            text=" Configuration ",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
        )
        config_frame.pack(fill="x", padx=15, pady=10)

        # Iterations
        iter_frame = tk.Frame(config_frame, bg=COLORS["bg_secondary"])
        iter_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(
            iter_frame,
            text="Iterations:",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
        ).pack(side="left")

        self.iterations_var = tk.StringVar(value="10")
        self.iterations_entry = tk.Entry(
            iter_frame,
            textvariable=self.iterations_var,
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"],
            font=FONTS["mono_small"],
            width=8,
        )
        self.iterations_entry.pack(side="right")

        # Output file
        file_frame = tk.Frame(config_frame, bg=COLORS["bg_secondary"])
        file_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(
            file_frame,
            text="Save to:",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
        ).pack(side="left")

        self.output_var = tk.StringVar(value="self.pt")
        self.output_entry = tk.Entry(
            file_frame,
            textvariable=self.output_var,
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"],
            font=FONTS["mono_small"],
            width=12,
        )
        self.output_entry.pack(side="right")

        # Control buttons
        btn_frame = tk.Frame(self, bg=COLORS["bg_secondary"])
        btn_frame.pack(fill="x", padx=15, pady=15)

        self.start_btn = create_styled_button(
            btn_frame,
            "Start Training",
            command=self._on_start_clicked,
            style="accent",
        )
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.stop_btn = create_styled_button(
            btn_frame,
            "Stop",
            command=self._on_stop_clicked,
            style="outline",
        )
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=(5, 0))
        self.stop_btn.configure(state="disabled")

        # Add tooltips
        create_tooltip(self.start_btn, "Start self-play training")
        create_tooltip(self.stop_btn, "Stop training (model will be saved)")

        # Add validation error label
        self.validation_label = tk.Label(
            self,
            text="",
            bg=COLORS["bg_secondary"],
            fg=COLORS["error"],
            font=FONTS["small"],
        )
        self.validation_label.pack(fill="x", padx=15)

    def _draw_status_indicator(self, status: str) -> None:
        """Draw status indicator circle."""
        self.status_indicator.delete("all")

        color_map = {
            "idle": COLORS["text_muted"],
            "self_play": COLORS["warning"],
            "training": COLORS["accent"],
            "complete": COLORS["success"],
            "error": COLORS["error"],
        }

        color = color_map.get(status, COLORS["text_muted"])

        self.status_indicator.create_oval(
            2,
            2,
            10,
            10,
            fill=color,
            outline="",
        )

    def _validate_inputs(self) -> bool:
        """Validate input fields. Returns True if valid."""
        # Validate iterations
        try:
            iterations = int(self.iterations_var.get())
            if iterations < 1 or iterations > 10000:
                self.validation_label.configure(
                    text=f"{STATUS_ICONS['error']} Iterations must be between 1 and 10000"
                )
                return False
        except ValueError:
            self.validation_label.configure(
                text=f"{STATUS_ICONS['error']} Iterations must be a number"
            )
            return False

        # Validate output filename
        output_file = self.output_var.get().strip()
        if not output_file:
            self.validation_label.configure(
                text=f"{STATUS_ICONS['error']} Output filename is required"
            )
            return False

        # Check for invalid characters in filename
        invalid_chars = '<>:"/\\|?*'
        if any(c in output_file for c in invalid_chars):
            self.validation_label.configure(
                text=f"{STATUS_ICONS['error']} Invalid characters in filename"
            )
            return False

        # Ensure .pt extension
        if not output_file.endswith(".pt"):
            self.output_var.set(output_file + ".pt")

        self.validation_label.configure(text="")
        return True

    def _on_start_clicked(self) -> None:
        """Handle start button click."""
        # Validate inputs first
        if not self._validate_inputs():
            return

        self.is_training = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="Starting...")
        self._draw_status_indicator("self_play")

        if self.on_start:
            self.on_start()

    def _on_stop_clicked(self) -> None:
        """Handle stop button click."""
        self.is_training = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Stopped")
        self._draw_status_indicator("idle")

        if self.on_stop:
            self.on_stop()

    def update_progress(self, data: dict[str, Any]) -> None:
        """
        Queue a progress update.

        Args:
            data: Progress data dictionary
        """
        self.update_queue.put(data)

    def _start_update_loop(self) -> None:
        """Start the UI update loop."""
        self._process_updates()

    def _process_updates(self) -> None:
        """Process queued updates."""
        try:
            while True:
                data = self.update_queue.get_nowait()
                self._apply_update(data)
        except queue.Empty:
            pass

        # Schedule next update
        self.after(100, self._process_updates)

    def _apply_update(self, data: dict[str, Any]) -> None:
        """Apply a progress update to the UI."""
        phase = data.get("phase", "")

        if phase == "iteration_start":
            iteration = data.get("iteration", 0)
            total = data.get("total_iterations", 0)
            self.status_label.configure(text=f"Iteration {iteration}/{total}")
            self.stats["iteration"].configure(text=f"{iteration}/{total}")

        elif phase == "self_play":
            self._draw_status_indicator("self_play")
            self.status_label.configure(text="Self-play...")

            games_played = data.get("games_played", 0)
            total_games = data.get("total_games", 0)

            if total_games > 0:
                progress = games_played / total_games
                self.games_progress.set_value(progress)
                self.games_text.configure(text=f"{games_played} / {total_games}")

            if "examples" in data:
                self.stats["examples"].configure(text=str(data["examples"]))

            if "games_per_second" in data:
                self.stats["games_per_sec"].configure(
                    text=f"{data['games_per_second']:.2f}"
                )

        elif phase == "training":
            self._draw_status_indicator("training")
            self.status_label.configure(text="Training...")

            epoch = data.get("epoch", 0)
            epochs = data.get("epochs", 0)

            if epochs > 0:
                progress = epoch / epochs
                self.train_progress.set_value(progress)
                self.train_text.configure(text=f"Epoch {epoch} / {epochs}")

            if "policy_loss" in data:
                self.stats["policy_loss"].configure(text=f"{data['policy_loss']:.4f}")

            if "value_loss" in data:
                self.stats["value_loss"].configure(text=f"{data['value_loss']:.4f}")

        elif phase == "iteration_complete":
            iteration = data.get("iteration", 0)

            # Reset progress bars for next iteration
            self.games_progress.set_value(0)
            self.train_progress.set_value(0)

            if "stats" in data:
                stats = data["stats"]
                if "selfplay_stats" in stats:
                    sp = stats["selfplay_stats"]
                    total = sp.get("games_played", 0)
                    if total > 0:
                        white = sp.get("white_wins", 0)
                        black = sp.get("black_wins", 0)
                        draws = sp.get("draws", 0)
                        self.stats["win_rate"].configure(
                            text=f"W:{white} B:{black} D:{draws}"
                        )

        elif phase == "complete":
            self._draw_status_indicator("complete")
            saved_to = data.get("saved_to", "")
            if saved_to:
                self.status_label.configure(text=f"Saved to {saved_to}")
            else:
                self.status_label.configure(text="Training complete")
            self.is_training = False
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

    def reset(self) -> None:
        """Reset the panel to initial state."""
        self.is_training = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Ready")
        self._draw_status_indicator("idle")

        self.games_progress.set_value(0)
        self.train_progress.set_value(0)
        self.games_text.configure(text="0 / 0")
        self.train_text.configure(text="Epoch 0 / 0")

        for key in self.stats:
            self.stats[key].configure(
                text="-" if key not in ["iteration", "examples"] else "0"
            )

    def get_iterations(self) -> int:
        """Get the number of training iterations."""
        try:
            return int(self.iterations_var.get())
        except ValueError:
            return 10

    def get_output_file(self) -> str:
        """Get the output file path."""
        return self.output_var.get() or "self.pt"


class TrainingConfigDialog(tk.Toplevel):
    """Dialog for configuring training parameters."""

    def __init__(self, parent: tk.Widget, current_config: dict[str, Any] | None = None):
        """
        Initialize the config dialog.

        Args:
            parent: Parent widget
            current_config: Current configuration values
        """
        super().__init__(parent)

        self.title("Training Configuration")
        self.configure(bg=COLORS["bg_primary"])
        self.resizable(True, False)

        self.config = current_config or {}
        self.result = None

        self._create_widgets()

        # Center on parent
        self.transient(parent)
        self.grab_set()

    def _create_widgets(self) -> None:
        """Create dialog widgets."""
        main_frame = tk.Frame(self, bg=COLORS["bg_primary"])
        main_frame.pack(padx=20, pady=20)

        # Title
        title = create_styled_label(
            main_frame, "Training Configuration", style="heading"
        )
        title.pack(pady=(0, 15))

        # Config fields
        fields_frame = tk.Frame(main_frame, bg=COLORS["bg_primary"])
        fields_frame.pack(fill="x")

        self.entries = {}
        fields = [
            ("games_per_iteration", "Games per iteration:", 100),
            ("num_simulations", "MCTS simulations:", 200),
            ("epochs_per_iteration", "Epochs per iteration:", 10),
            ("batch_size", "Batch size:", 256),
            ("learning_rate", "Learning rate:", 0.001),
        ]

        for i, (key, label_text, default) in enumerate(fields):
            label = create_styled_label(
                fields_frame,
                label_text,
                style="body",
            )
            label.grid(row=i, column=0, sticky="w", pady=5)

            entry = tk.Entry(
                fields_frame,
                bg=COLORS["bg_tertiary"],
                fg=COLORS["text_primary"],
                insertbackground=COLORS["text_primary"],
                font=FONTS["mono"],
                width=15,
            )
            entry.insert(0, str(self.config.get(key, default)))
            entry.grid(row=i, column=1, sticky="e", pady=5, padx=(10, 0))

            self.entries[key] = entry

        # Buttons
        btn_frame = tk.Frame(main_frame, bg=COLORS["bg_primary"])
        btn_frame.pack(fill="x", pady=(20, 0))

        cancel_btn = create_styled_button(
            btn_frame,
            "Cancel",
            command=self._on_cancel,
            style="outline",
        )
        cancel_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))

        save_btn = create_styled_button(
            btn_frame,
            "Save",
            command=self._on_save,
            style="accent",
        )
        save_btn.pack(side="left", expand=True, fill="x", padx=(5, 0))

    def _on_save(self) -> None:
        """Handle save button."""
        self.result = {}
        for key, entry in self.entries.items():
            value = entry.get()
            try:
                if "." in value:
                    self.result[key] = float(value)
                else:
                    self.result[key] = int(value)
            except ValueError:
                self.result[key] = value

        self.destroy()

    def _on_cancel(self) -> None:
        """Handle cancel button."""
        self.result = None
        self.destroy()

    def get_result(self) -> dict[str, Any] | None:
        """Get the dialog result."""
        self.wait_window()
        return self.result
