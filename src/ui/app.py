"""
Main chess game application.

Provides the complete UI for playing chess against the AlphaZero AI,
watching AI vs AI games, and managing training.

Professional chess.com-inspired design with AI debugging features.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, filedialog
from typing import Any, Literal, TYPE_CHECKING
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
    create_tooltip,
    create_confirmation_dialog,
    load_piece_image,
    Card,
    ChoiceList,
    HelpBadge,
    SegmentedControl,
    Slider,
    ThinkingIndicator,
)
from .board_widget import ChessBoardWidget
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
from .components.ai_insights import (
    AdvantageGraph,
    AIPlanPanel,
    CandidateMovesPanel,
    WinChancePanel,
    expected_score,
)

if TYPE_CHECKING:
    from alphazero.network import DualHeadNetwork
    from alphazero.mcts import MCTS
    from alphazero.arena import PureMCTSPlayer
    from alphazero.trainer import AlphaZeroTrainer


class ChessGameApp(object):
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
    # Softmax temperature on the network priors (flattens overconfident policies)
    POLICY_TEMPERATURE = 1.6

    # Dynamic time management constants
    # Minimum simulations floor (even in time pressure)
    MIN_SIMULATIONS = 100
    # Number of moves for time-per-sim calculation
    HISTORY_WINDOW = 10
    # Exponential decay for older moves
    DECAY_FACTOR = 0.85
    # Seconds - play instantly from tree, no search
    EMERGENCY_THRESHOLD = 15
    # Seconds - play instantly from tree, no search
    INSTANT_MOVE_THRESHOLD = 3

    def __init__(
        self,
        network_path: str | None = None,
        num_simulations: int = 200,
        mcts_batch_size: int = DEFAULT_MCTS_BATCH_SIZE,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        verbose: bool = False,
        time_control: int | None = None,
    ) -> None:
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
        self.root.title("NeuralMate - Échecs contre l'IA")
        self.root.resizable(True, True)
        self.root.geometry("1720x1000")
        self.root.minsize(1480, 900)

        apply_theme(self.root)

        # Network and AI
        self.network: DualHeadNetwork | None = None
        self.mcts: MCTS | None = None
        self.pure_mcts_player: PureMCTSPlayer | None = None
        self.network_path: str | None = network_path
        self.num_simulations: int = num_simulations
        self.mcts_batch_size: int = mcts_batch_size
        self.history_length: int = history_length
        self.verbose: bool = verbose
        self.use_pure_mcts: None | bool | Literal[""] = (
            network_path and network_path.lower() == "mcts"
        )

        # Load c_puct from config.json
        self.c_puct: float = 1.5  # default
        config_path: str = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config",
            "config.json",
        )
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config: dict[str, Any] = json.load(f)
                    self.c_puct = config.get("training", {}).get("c_puct", 1.5)
            except Exception:
                # Use default on error
                pass

        # Game state
        self.game_mode: str = "human_vs_ai"
        self.human_color: bool = chess.WHITE
        self.ai_thinking: bool = False
        self.ai_thread: threading.Thread | None = None
        self.ai_cancelled: bool = False
        # Incremented on every new game: stale AI results are ignored
        self.game_id: int = 0

        # Move history for back/forward navigation
        self.undone_moves: list[tuple[chess.Move, chess.Move | None]] = []

        # Full game history for MCTS (all positions from start, not limited to 3)
        # This is needed because MCTS takes _initial_history[:3] (first 3 positions)
        # [pos_initial, pos_after_move1, ...]
        self.game_positions: list[chess.Board] = []

        # Evaluation history for graph
        self.eval_history: list[float] = []
        # WDL for each eval
        self.wdl_history: list[np.ndarray | None] = []
        # Beginner analysis history: (ply, white_score, eval_bar_value, wdl_white)
        self.eval_points: list[tuple[int, float, float, tuple[float, float, float]]] = (
            []
        )

        # Move timing
        self.turn_start_time: float = time.time()
        self.last_move_time: float = 0.0

        # Time control
        # Convert minutes to seconds
        self.time_control: int | None = time_control * 60 if time_control else None
        self.white_time: float = 0.0
        self.black_time: float = 0.0
        # Store original for scaling
        self.base_simulations: int = num_simulations
        self.timer_update_id: str | None = None
        # Timer only runs after first move
        self.game_started: bool = False

        # Dynamic time management stats
        # Time taken for each AI move
        self.ai_move_times: list[float] = []
        # Simulations used for each AI move
        self.ai_sims_used: list[int] = []
        # Last WDL evaluation
        self.last_wdl: tuple[float, float, float] | None = None

        # Update queue for thread-safe UI updates
        self.update_queue: queue.Queue[tuple[str, dict[str, Any] | str]] = queue.Queue()

        # Training state
        self.trainer: AlphaZeroTrainer | None = None
        self.training_thread: threading.Thread | None = None
        self.training_cancelled: bool = False

        # Editor mode state
        # "play" or "editor"
        self.app_mode: str = "play"
        self.editor_selected_piece: chess.Piece | None = None
        self.editor_palette_buttons: dict[str, tk.Button] = {}

        self._create_ui()
        self._update_player_names_for_human_ai()
        self._update_game_info()
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
        self.root.bind("<Left>", lambda e: self._unless_typing(self._back_one_move))
        self.root.bind("<Right>", lambda e: self._unless_typing(self._forward_one_move))
        self.root.bind("<f>", lambda e: self._unless_typing(self._flip_board))
        self.root.bind("<F>", lambda e: self._unless_typing(self._flip_board))
        self.root.bind("<Escape>", lambda e: self._cancel_selection())

    def _unless_typing(self, action: Any) -> None:
        """Run a single-key shortcut unless the user is typing in a text field."""
        if isinstance(self.root.focus_get(), tk.Entry):
            return
        action()

    def _create_ui(self) -> None:
        """Create the main UI layout: header, then 3 columns."""
        # Header bar (logo, Play/Editor switch, status, network)
        self._create_top_bar(self.root)

        # Main container
        main_frame = tk.Frame(self.root, bg=COLORS["bg_primary"])
        main_frame.pack(padx=20, pady=(14, 16), fill="both", expand=True)

        # Content area - 3 column layout
        content_frame = tk.Frame(main_frame, bg=COLORS["bg_primary"])
        content_frame.pack(fill="both", expand=True)

        # Left panel - game setup, AI level, game info, move list
        self._create_left_panel(content_frame)

        # Center panel - players, board, eval bar, navigation
        self._create_center_panel(content_frame)

        # Right panel - beginner analysis + technical details
        self._create_right_panel(content_frame)

    def _create_top_bar(self, parent: tk.Widget) -> None:
        """Create the header bar."""
        header = tk.Frame(parent, bg=COLORS["bg_secondary"])
        header.pack(fill="x")
        tk.Frame(parent, bg=COLORS["border"], height=1).pack(fill="x")

        inner = tk.Frame(header, bg=COLORS["bg_secondary"])
        inner.pack(fill="x", padx=20, pady=10)

        # Logo
        logo = tk.Frame(inner, bg=COLORS["bg_secondary"])
        logo.pack(side="left")
        tk.Label(
            logo,
            text="♞",
            font=("Segoe UI Symbol", 22),
            fg=COLORS["accent"],
            bg=COLORS["bg_secondary"],
        ).pack(side="left", padx=(0, 8))
        title_box = tk.Frame(logo, bg=COLORS["bg_secondary"])
        title_box.pack(side="left")
        tk.Label(
            title_box,
            text="NeuralMate",
            font=FONTS["heading"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_secondary"],
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            title_box,
            text="Jouez aux échecs contre une IA qui apprend seule",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_secondary"],
            anchor="w",
        ).pack(fill="x")

        # Mode toggle (Play / Editor)
        self.app_mode_var = tk.StringVar(value="play")
        self.mode_switch = SegmentedControl(
            inner,
            options=[("♟  Jouer", "play"), ("✎  Éditer la position", "editor")],
            variable=self.app_mode_var,
            command=self._on_app_mode_change,
            font=FONTS["small_bold"],
        )
        self.mode_switch.pack(side="left", padx=(36, 0))
        create_tooltip(
            self.mode_switch,
            "Jouer : faire une partie.\nÉditer : placer les pièces librement "
            "pour créer une position, puis la jouer.",
        )

        # Right side - network info + load button
        right_frame = tk.Frame(inner, bg=COLORS["bg_secondary"])
        right_frame.pack(side="right")

        self.load_btn: tk.Button = create_styled_button(
            right_frame,
            "Changer de modèle",
            command=self._load_network_dialog,
            style="outline",
            font=FONTS["small_bold"],
            padx=12,
            pady=6,
        )
        self.load_btn.pack(side="right")
        create_tooltip(
            self.load_btn,
            "Charger un autre réseau de neurones (fichier .pt).\n"
            "Un réseau plus entraîné joue plus fort.",
        )

        net_box = tk.Frame(right_frame, bg=COLORS["bg_secondary"])
        net_box.pack(side="right", padx=(0, 12))
        tk.Label(
            net_box,
            text="CERVEAU DE L'IA",
            font=FONTS["caption_bold"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_secondary"],
            anchor="e",
        ).pack(fill="x")
        self.network_status = tk.Label(
            net_box,
            text="Aucun réseau chargé",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_muted"],
            font=FONTS["small_bold"],
            anchor="e",
        )
        self.network_status.pack(fill="x")

        # Center - status pill
        status_pill = tk.Frame(inner, bg=COLORS["bg_tertiary"])
        status_pill.pack(side="left", expand=True)
        pill_inner = tk.Frame(status_pill, bg=COLORS["bg_tertiary"])
        pill_inner.pack(padx=16, pady=7)

        self.status_dot = tk.Canvas(
            pill_inner,
            width=10,
            height=10,
            bg=COLORS["bg_tertiary"],
            highlightthickness=0,
        )
        self.status_dot.pack(side="left", padx=(0, 9))

        self.status_label = tk.Label(
            pill_inner,
            text="À vous de jouer",
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            font=FONTS["body_bold"],
        )
        self.status_label.pack(side="left")
        self._set_status("À vous de jouer", "turn")

    def _set_status(self, text: str, kind: str = "info") -> None:
        """
        Update the header status pill.

        Args:
            text: Status text
            kind: "turn", "thinking", "check", "over", "editor" or "info"
        """
        colors: dict[str, str] = {
            "turn": COLORS["success"],
            "thinking": COLORS["accent"],
            "check": COLORS["error"],
            "over": COLORS["warning"],
            "editor": COLORS["opening_eco"],
            "info": COLORS["text_muted"],
        }
        color: str = colors.get(kind, COLORS["text_muted"])
        self.status_label.configure(text=text)
        self.status_dot.delete("all")
        self.status_dot.create_oval(1, 1, 9, 9, fill=color, outline="")

    def _create_left_panel(self, parent: tk.Widget) -> None:
        """Create the left column (play setup or editor tools)."""
        left_panel = tk.Frame(parent, bg=COLORS["bg_primary"], width=320)
        left_panel.pack(side="left", fill="y", padx=(0, 18))
        left_panel.pack_propagate(False)

        # Play mode content
        self.play_left_content = tk.Frame(left_panel, bg=COLORS["bg_primary"])
        self.play_left_content.pack(fill="both", expand=True)

        # Game mode + color + new game
        self._create_mode_panel(self.play_left_content)

        # Game info: opening + phase side by side, then the move list
        info_card = Card(self.play_left_content, padding=0)
        info_card.pack(fill="both", expand=True)

        info_row = tk.Frame(info_card.body, bg=COLORS["bg_secondary"])
        info_row.pack(fill="x", padx=14, pady=(12, 0))
        info_row.grid_columnconfigure(0, weight=1, uniform="info")
        info_row.grid_columnconfigure(1, weight=1, uniform="info")

        self.opening_display = OpeningDisplay(info_row)
        self.opening_display.grid(row=0, column=0, sticky="nw", padx=(0, 8))
        self.opening_display._name_label.configure(wraplength=125)

        self.phase_indicator = PhaseIndicator(info_row, show_tip=False)
        self.phase_indicator.grid(row=0, column=1, sticky="nw")

        tk.Frame(info_card.body, bg=COLORS["border"], height=1).pack(
            fill="x", padx=14, pady=(10, 0)
        )

        # Move list fills the remaining height
        self.move_list = MoveList(info_card.body, width=300, height=120)
        self.move_list.configure(highlightthickness=0)
        self.move_list.pack(fill="both", expand=True)

        # Editor mode content (initially hidden)
        self.editor_left_content = tk.Frame(left_panel, bg=COLORS["bg_primary"])
        self._create_editor_panel(self.editor_left_content)

    def _create_mode_panel(self, parent: tk.Widget) -> None:
        """Create the game setup card (mode, color, AI level, new game)."""
        mode_card = Card(parent, title="Nouvelle partie")
        mode_card.pack(fill="x", pady=(0, 12))
        body: tk.Frame = mode_card.body

        self.mode_var = tk.StringVar(value="human_vs_ai")
        ChoiceList(
            body,
            options=[
                ("human_vs_ai", "♔", "Contre l'IA", "Affrontez le réseau de neurones"),
                ("ai_vs_ai", "⚙", "IA contre IA", "Regardez l'IA jouer seule"),
                ("human_vs_human", "⚔", "Deux joueurs", "Jouez à deux sur cet écran"),
            ],
            variable=self.mode_var,
            command=self._on_mode_change,
        ).pack(fill="x")

        # Color selection
        self.color_frame = tk.Frame(body, bg=COLORS["bg_secondary"])
        self.color_frame.pack(fill="x", pady=(10, 0))

        tk.Label(
            self.color_frame,
            text="Vous jouez les",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
        ).pack(side="left", padx=(0, 10))

        self.color_var = tk.StringVar(value="white")
        SegmentedControl(
            self.color_frame,
            options=[("♔ Blancs", "white"), ("♚ Noirs", "black")],
            variable=self.color_var,
            command=self._on_color_change,
        ).pack(side="left", fill="x", expand=True)

        self.new_game_btn: tk.Button = create_styled_button(
            body,
            "Nouvelle partie",
            command=self._new_game_with_confirm,
            style="accent",
            pady=9,
        )
        self.new_game_btn.pack(fill="x", pady=(12, 0))
        create_tooltip(self.new_game_btn, "Recommencer une partie (Ctrl+N)")

        # AI Settings panel
        self._create_ai_settings_panel(parent)

    # Named AI strength presets: (simulations, label, description)
    AI_LEVELS: list[tuple[int, str, str]] = [
        (1, "Instinct", "Joue son premier réflexe, sans réfléchir."),
        (100, "Rapide", "Réfléchit très peu : idéal pour débuter."),
        (200, "Normal", "Un bon compromis entre force et rapidité."),
        (800, "Fort", "Réfléchit longuement avant chaque coup."),
        (1600, "Max", "Toute sa puissance, mais plus lent."),
    ]

    def _create_ai_settings_panel(self, parent: tk.Widget) -> None:
        """Create the AI strength card with named presets and a fine slider."""
        ai_card = Card(
            parent,
            title="Niveau de l'IA",
            help_text=(
                "Avant chaque coup, l'IA imagine des positions futures pour choisir "
                "le meilleur coup.\n\nPlus elle en étudie, plus elle joue fort… "
                "mais plus elle met de temps à répondre."
            ),
        )
        ai_card.pack(fill="x", pady=(0, 12))
        ai_content: tk.Frame = ai_card.body

        self.level_var = tk.StringVar(value="")
        SegmentedControl(
            ai_content,
            options=[(label, str(sims)) for sims, label, _ in self.AI_LEVELS],
            variable=self.level_var,
            command=lambda: self._set_simulations(int(self.level_var.get())),
            font=FONTS["caption_bold"],
        ).pack(fill="x")

        # Simulations slider (fine tuning, logarithmic)
        self.sim_var = tk.IntVar(value=self.num_simulations)
        self.sim_slider = Slider(
            ai_content,
            from_=1,
            to=2000,
            variable=self.sim_var,
            command=self._on_simulations_change,
            log=True,
        )
        self.sim_slider.pack(fill="x", pady=(8, 0))
        create_tooltip(
            self.sim_slider, "Réglage fin : glissez pour choisir précisément."
        )

        self.sim_value_label = tk.Label(
            ai_content,
            text="",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            font=FONTS["caption_bold"],
            anchor="w",
        )
        self.sim_value_label.pack(fill="x", pady=(2, 0))

        self.level_desc_label = tk.Label(
            ai_content,
            text="",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["caption"],
            anchor="w",
            justify="left",
            wraplength=260,
        )
        self.level_desc_label.pack(fill="x")
        self._refresh_level_display()

    def _refresh_level_display(self) -> None:
        """Sync the level presets and labels with the current simulations."""
        sims: int = self.num_simulations
        plural: str = "s" if sims > 1 else ""
        self.sim_value_label.configure(
            text=f"{sims} position{plural} étudiée{plural} par coup"
        )
        preset = next((lvl for lvl in self.AI_LEVELS if lvl[0] == sims), None)
        if preset is not None:
            self.level_var.set(str(sims))
            self.level_desc_label.configure(text=preset[2])
        else:
            self.level_var.set("")
            self.level_desc_label.configure(text="Niveau personnalisé.")

    def _on_simulations_change(self, value: str) -> None:
        """Handle simulations slider change."""
        new_sims: int = int(float(value))
        self.num_simulations = new_sims
        self.base_simulations = new_sims
        self._refresh_level_display()

        # Update MCTS if it exists
        if self.mcts is not None:
            self.mcts.num_simulations = new_sims
        if self.pure_mcts_player is not None:
            self.pure_mcts_player._num_simulations = new_sims

    def _set_simulations(self, value: int) -> None:
        """Set simulations to a preset value."""
        self.sim_var.set(value)
        self._on_simulations_change(str(value))

    def _create_editor_panel(self, parent: tk.Widget) -> None:
        """Create the position editor panel with piece palette and controls."""
        from .styles import PIECE_UNICODE

        # Piece palette
        palette_card = Card(
            parent,
            title="Pièces",
            help_text=(
                "Choisissez une pièce puis cliquez sur une case pour la poser.\n"
                "Clic droit sur une case : retirer la pièce.\n"
                "Vous pouvez aussi déplacer les pièces en les glissant."
            ),
        )
        palette_card.pack(fill="x", pady=(0, 12))
        palette_content: tk.Frame = palette_card.body

        rows = [
            ("Blancs", chess.WHITE, ["K", "Q", "R", "B", "N", "P"]),
            ("Noirs", chess.BLACK, ["k", "q", "r", "b", "n", "p"]),
        ]
        piece_types: dict[str, int] = {
            "k": chess.KING,
            "q": chess.QUEEN,
            "r": chess.ROOK,
            "b": chess.BISHOP,
            "n": chess.KNIGHT,
            "p": chess.PAWN,
        }
        for label, color, symbols in rows:
            tk.Label(
                palette_content,
                text=label.upper(),
                bg=COLORS["bg_secondary"],
                fg=COLORS["text_muted"],
                font=FONTS["caption_bold"],
                anchor="w",
            ).pack(fill="x", pady=(0, 3))

            grid = tk.Frame(palette_content, bg=COLORS["bg_secondary"])
            grid.pack(fill="x", pady=(0, 8))
            for i, symbol in enumerate(symbols):
                piece_type: int = piece_types[symbol.lower()]
                btn = tk.Button(
                    grid,
                    command=lambda c=color, pt=piece_type: self._select_editor_piece(
                        c, pt
                    ),
                    bg=COLORS["palette_bg"],
                    activebackground=COLORS["accent"],
                    relief="flat",
                    bd=0,
                    highlightthickness=0,
                    cursor="hand2",
                )
                img = load_piece_image(symbol, 34)
                if img is not None:
                    btn.configure(image=img, width=40, height=40)
                else:
                    btn.configure(
                        text=PIECE_UNICODE.get(symbol, symbol),
                        font=("Segoe UI Symbol", 18),
                        fg=COLORS["text_primary"],
                        width=2,
                    )
                btn.grid(row=0, column=i, padx=(0, 4))
                key: str = f"w{symbol}" if color == chess.WHITE else f"b{symbol}"
                self.editor_palette_buttons[key] = btn

        self.eraser_btn = create_styled_button(
            palette_content,
            "⌫  Gomme (ou clic droit)",
            command=self._select_editor_eraser,
            style="outline",
            font=FONTS["small_bold"],
            pady=6,
        )
        self.eraser_btn.pack(fill="x")

        # Position controls
        controls_card = Card(parent, title="Réglages de la position")
        controls_card.pack(fill="x", pady=(0, 12))
        controls_content: tk.Frame = controls_card.body

        # Turn to move
        turn_frame = tk.Frame(controls_content, bg=COLORS["bg_secondary"])
        turn_frame.pack(fill="x", pady=(0, 10))

        tk.Label(
            turn_frame,
            text="Au trait",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
            width=8,
            anchor="w",
        ).pack(side="left")

        self.editor_turn_var = tk.StringVar(value="white")
        SegmentedControl(
            turn_frame,
            options=[("Blancs", "white"), ("Noirs", "black")],
            variable=self.editor_turn_var,
            command=self._on_editor_position_change,
        ).pack(side="left", fill="x", expand=True)

        # Castling rights
        tk.Label(
            controls_content,
            text="Roques encore possibles",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
            anchor="w",
        ).pack(fill="x")

        castle_checks = tk.Frame(controls_content, bg=COLORS["bg_secondary"])
        castle_checks.pack(fill="x", pady=(2, 10))

        self.castle_wk_var = tk.BooleanVar(value=True)
        self.castle_wq_var = tk.BooleanVar(value=True)
        self.castle_bk_var = tk.BooleanVar(value=True)
        self.castle_bq_var = tk.BooleanVar(value=True)

        for i, (text, var) in enumerate(
            [
                ("Blancs petit", self.castle_wk_var),
                ("Blancs grand", self.castle_wq_var),
                ("Noirs petit", self.castle_bk_var),
                ("Noirs grand", self.castle_bq_var),
            ]
        ):
            tk.Checkbutton(
                castle_checks,
                text=text,
                variable=var,
                command=self._on_editor_position_change,
                bg=COLORS["bg_secondary"],
                fg=COLORS["text_primary"],
                selectcolor=COLORS["bg_tertiary"],
                activebackground=COLORS["bg_secondary"],
                activeforeground=COLORS["text_primary"],
                font=FONTS["caption"],
                anchor="w",
                highlightthickness=0,
                bd=0,
            ).grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 10))

        # FEN display
        fen_header = tk.Frame(controls_content, bg=COLORS["bg_secondary"])
        fen_header.pack(fill="x")
        tk.Label(
            fen_header,
            text="Code FEN",
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"],
        ).pack(side="left")
        HelpBadge(
            fen_header,
            "Le FEN est un texte qui décrit une position d'échecs. Vous pouvez "
            "en coller un (copié depuis Lichess ou Chess.com) puis cliquer sur "
            "« Charger ».",
        ).pack(side="left", padx=(6, 0))

        fen_row = tk.Frame(controls_content, bg=COLORS["bg_secondary"])
        fen_row.pack(fill="x", pady=(4, 0))

        self.editor_fen_var = tk.StringVar(value=chess.STARTING_FEN)
        self.editor_fen_entry = tk.Entry(
            fen_row,
            textvariable=self.editor_fen_var,
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            font=("Consolas", 9),
            insertbackground=COLORS["text_primary"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=COLORS["border_light"],
            highlightcolor=COLORS["accent"],
        )
        self.editor_fen_entry.pack(side="left", fill="x", expand=True, ipady=5)
        self.editor_fen_entry.bind("<Return>", lambda e: self._load_editor_fen())

        create_styled_button(
            fen_row,
            "Charger",
            command=self._load_editor_fen,
            style="outline",
            font=FONTS["small_bold"],
            padx=10,
            pady=4,
        ).pack(side="left", padx=(6, 0))

        # Action buttons
        actions_card = Card(parent, title="Actions")
        actions_card.pack(fill="x", pady=(0, 12))
        actions_content: tk.Frame = actions_card.body

        row = tk.Frame(actions_content, bg=COLORS["bg_secondary"])
        row.pack(fill="x")
        row.grid_columnconfigure(0, weight=1, uniform="a")
        row.grid_columnconfigure(1, weight=1, uniform="a")

        create_styled_button(
            row,
            "Vider l'échiquier",
            command=self._editor_clear_board,
            style="outline",
            font=FONTS["small_bold"],
            pady=6,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))

        create_styled_button(
            row,
            "Position initiale",
            command=self._editor_reset_position,
            style="outline",
            font=FONTS["small_bold"],
            pady=6,
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # Start game button (prominent)
        self.start_game_btn = create_styled_button(
            actions_content,
            "▶  Jouer cette position",
            command=self._start_game_from_editor,
            style="accent",
            pady=9,
        )
        self.start_game_btn.pack(fill="x", pady=(10, 0))

    def _create_center_panel(self, parent: tk.Widget) -> None:
        """Create the center column: player bars, board, eval bar, navigation."""
        center_panel = tk.Frame(parent, bg=COLORS["bg_primary"])
        center_panel.pack(side="left", fill="y")
        self.center_panel = center_panel

        # Player bars (packed around the board by _layout_player_bars)
        self.black_player_info = PlayerInfo(center_panel, color=chess.BLACK, name="IA")
        self.white_player_info = PlayerInfo(
            center_panel, color=chess.WHITE, name="Vous"
        )

        # Board row (eval bar + board side by side)
        self.board_row = tk.Frame(center_panel, bg=COLORS["bg_primary"])
        self.board_row.pack()

        # Evaluation bar (left of board)
        self.eval_bar = EvalBar(self.board_row, height=640, width=26)
        self.eval_bar.pack(side="left", padx=(0, 8))
        create_tooltip(
            self.eval_bar,
            "Barre d'avantage : plus la partie blanche est grande, plus les "
            "Blancs ont de chances de gagner (et inversement).",
        )

        # Chess board
        self.board_widget = ChessBoardWidget(
            self.board_row,
            size=640,
            on_move=self._on_human_move,
        )
        self.board_widget.pack(side="left")

        # Navigation bar below the board
        self.nav_bar = tk.Frame(center_panel, bg=COLORS["bg_primary"])
        self.nav_bar.pack(fill="x", pady=(10, 0))

        self.back_btn: tk.Button = create_styled_button(
            self.nav_bar,
            "◀  Annuler le coup",
            command=self._back_one_move,
            style="outline",
            font=FONTS["small_bold"],
        )
        self.back_btn.pack(side="left", padx=(0, 6))

        self.forward_btn: tk.Button = create_styled_button(
            self.nav_bar,
            "Rétablir  ▶",
            command=self._forward_one_move,
            style="outline",
            font=FONTS["small_bold"],
        )
        self.forward_btn.pack(side="left", padx=(0, 6))

        self.flip_btn: tk.Button = create_styled_button(
            self.nav_bar,
            "⇅  Retourner",
            command=self._flip_board,
            style="outline",
            font=FONTS["small_bold"],
        )
        self.flip_btn.pack(side="left")

        tk.Label(
            self.nav_bar,
            text="Raccourcis : ← →  ·  F  ·  Ctrl+N",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_primary"],
        ).pack(side="right")

        # Tooltips
        create_tooltip(self.back_btn, "Revenir en arrière (Ctrl+Z ou ←)")
        create_tooltip(self.forward_btn, "Rejouer le coup annulé (→)")
        create_tooltip(self.flip_btn, "Voir l'échiquier de l'autre côté (F)")

        self._layout_player_bars()

    def _layout_player_bars(self) -> None:
        """Place the player bars above / below the board to match its orientation."""
        flipped: bool = self.board_widget.flipped
        top: PlayerInfo = self.white_player_info if flipped else self.black_player_info
        bottom: PlayerInfo = (
            self.black_player_info if flipped else self.white_player_info
        )
        top.pack_forget()
        bottom.pack_forget()
        top.pack(fill="x", before=self.board_row, pady=(0, 8))
        bottom.pack(fill="x", after=self.board_row, pady=(8, 0))
        self.eval_bar.set_flipped(flipped)

    def _create_right_panel(self, parent: tk.Widget) -> None:
        """Create the right column: beginner analysis and technical details."""
        right_panel = tk.Frame(parent, bg=COLORS["bg_primary"])
        right_panel.pack(side="left", fill="both", expand=True, padx=(18, 0))

        # Tabs + thinking indicator
        tab_row = tk.Frame(right_panel, bg=COLORS["bg_primary"])
        tab_row.pack(fill="x", pady=(0, 10))

        self.right_tab_var = tk.StringVar(value="simple")
        SegmentedControl(
            tab_row,
            options=[("Analyse", "simple"), ("Détails techniques", "expert")],
            variable=self.right_tab_var,
            command=self._on_right_tab_change,
        ).pack(side="left")

        # AI Thinking indicator
        self.thinking_frame = tk.Frame(tab_row, bg=COLORS["bg_primary"])
        self.thinking_frame.pack(side="right")

        self.thinking_indicator = ThinkingIndicator(
            self.thinking_frame,
            text="L'IA réfléchit",
            bg=COLORS["bg_primary"],
            font=FONTS["small_bold"],
            # Fixed width: the animated dots must not trigger relayouts
            width=17,
            anchor="w",
        )

        # ----- Beginner analysis tab -----
        self.simple_tab = tk.Frame(right_panel, bg=COLORS["bg_primary"])
        self.simple_tab.pack(fill="both", expand=True)

        # "Who is winning": current chances + how they evolved during the game
        self.win_panel = WinChancePanel(self.simple_tab)
        self.win_panel.pack(fill="x", pady=(0, 10))
        self.advantage_graph = AdvantageGraph(self.win_panel.body, height=78)
        self.advantage_graph.pack(fill="x", pady=(8, 0))

        self.candidates_panel = CandidateMovesPanel(self.simple_tab)
        self.candidates_panel.pack(fill="x", pady=(0, 10))

        self.plan_panel = AIPlanPanel(self.simple_tab)
        self.plan_panel.pack(fill="both", expand=True)

        # ----- Technical details tab (raw MCTS data) -----
        self.expert_tab = tk.Frame(right_panel, bg=COLORS["bg_primary"])

        legend_card = Card(self.expert_tab, title="Comment lire ces chiffres ?")
        legend_card.pack(fill="x", pady=(0, 10))
        legend: list[tuple[str, str]] = [
            ("Visits / N", "nombre de fois où l'IA a exploré ce coup"),
            ("Q-value / Q", "score moyen trouvé : -1 = perdu, 0 = égal, +1 = gagné"),
            ("WR%", "Q traduit en pourcentage de victoire (50 % = égal)"),
            ("W / D / L", "chances de Victoire / Nulle / Défaite"),
            ("Prior / P", "intuition du réseau avant toute réflexion"),
        ]
        for term, meaning in legend:
            line = tk.Frame(legend_card.body, bg=COLORS["bg_secondary"])
            line.pack(fill="x")
            tk.Label(
                line,
                text=term,
                font=FONTS["caption_bold"],
                fg=COLORS["text_primary"],
                bg=COLORS["bg_secondary"],
                width=12,
                anchor="w",
            ).pack(side="left")
            tk.Label(
                line,
                text=meaning,
                font=FONTS["caption"],
                fg=COLORS["text_secondary"],
                bg=COLORS["bg_secondary"],
                anchor="w",
            ).pack(side="left")

        # MCTS Panel
        self.mcts_panel = MCTSPanel(self.expert_tab, width=280, height=260)
        self.mcts_panel.pack(fill="x", pady=(0, 10))

        # Search Tree Panel
        self.search_tree_panel = SearchTreePanel(self.expert_tab, width=280, height=200)
        self.search_tree_panel.pack(fill="both", expand=True, pady=(0, 10))

        # Raw eval graph (value from -1 to +1, White's point of view)
        self.eval_graph = EvalGraph(self.expert_tab, width=280, height=100)
        self.eval_graph.pack(fill="x", pady=(0, 10))

        # Root value (raw)
        value_frame = tk.Frame(self.expert_tab, bg=COLORS["bg_secondary"])
        value_frame.pack(fill="x")

        tk.Label(
            value_frame,
            text="Valeur de la racine (point de vue de l'IA) :",
            font=FONTS["small"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_secondary"],
        ).pack(side="left", padx=(12, 6), pady=6)

        self.value_label = tk.Label(
            value_frame,
            text="0.00",
            font=("Consolas", 12, "bold"),
            fg=COLORS["text_primary"],
            bg=COLORS["bg_secondary"],
        )
        self.value_label.pack(side="left", pady=6)

        # Training panel - hidden, kept for compatibility
        self.training_panel = None

    def _on_right_tab_change(self) -> None:
        """Switch between the beginner analysis and the technical details."""
        if self.right_tab_var.get() == "expert":
            self.simple_tab.pack_forget()
            self.expert_tab.pack(fill="both", expand=True)
        else:
            self.expert_tab.pack_forget()
            self.simple_tab.pack(fill="both", expand=True)

    def _show_thinking(self) -> None:
        """Show that the AI is thinking."""
        self.thinking_indicator.pack()
        self.thinking_indicator.start()
        self._set_status("L'IA réfléchit…", "thinking")

    def _hide_thinking(self) -> None:
        """Hide the AI thinking indicator."""
        self.thinking_indicator.stop()
        self.thinking_indicator.pack_forget()

    def _reset_analysis(self) -> None:
        """Clear every analysis display (new game / new position)."""
        self.eval_history.clear()
        self.wdl_history.clear()
        self.eval_points.clear()
        self.mcts_panel.clear()
        self.search_tree_panel.clear()
        self.eval_bar.clear()
        self.eval_graph.clear()
        self.win_panel.clear()
        self.candidates_panel.clear()
        self.plan_panel.clear()
        self.advantage_graph.clear()
        self.value_label.configure(text="0.00", fg=COLORS["text_primary"])
        self.opening_display.clear()
        self.move_list.clear()

    def _side_names(self) -> dict[bool, str]:
        """How each color is called in analysis sentences."""
        if self.game_mode == "human_vs_ai":
            return {
                self.human_color: "vous",
                not self.human_color: "l'IA",
            }
        if self.game_mode == "ai_vs_ai":
            return {chess.WHITE: "l'IA des Blancs", chess.BLACK: "l'IA des Noirs"}
        return {chess.WHITE: "les Blancs", chess.BLACK: "les Noirs"}

    def _load_network(self) -> None:
        """Load the neural network or pure MCTS player."""

        def get_history_length(network: DualHeadNetwork) -> int:
            """Derive history_length from the network's input plane count."""
            planes: int = network.num_input_planes
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
                    text="MCTS pur (sans réseau de neurones)",
                    fg=COLORS["accent"],
                )
                return
            except Exception as e:
                self.network_status.configure(
                    text=f"Erreur : {e}",
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
                    policy_temperature=self.POLICY_TEMPERATURE,
                )
                # Initialize game history with starting position
                self.game_positions = [self.board_widget.get_board().copy()]
                self.network_status.configure(
                    text=f"✓ {os.path.basename(self.network_path)}",
                    fg=COLORS["success"],
                )
            except Exception as e:
                self.network_status.configure(
                    text=f"Erreur : {e}",
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
                    policy_temperature=self.POLICY_TEMPERATURE,
                )
                # Initialize game history with starting position
                self.game_positions = [self.board_widget.get_board().copy()]
                self.network_status.configure(
                    text="⚠ Réseau non entraîné (joue au hasard)",
                    fg=COLORS["warning"],
                )
            except Exception as e:
                self.network_status.configure(
                    text=f"Erreur : {e}",
                    fg=COLORS["error"],
                )

    def _load_network_dialog(self) -> None:
        """Show dialog to load a network file."""
        filepath: str = filedialog.askopenfilename(
            title="Charger un réseau de neurones",
            filetypes=[("Réseaux NeuralMate", "*.pt"), ("Tous les fichiers", "*.*")],
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
            self._update_player_names("IA (Blancs)", "IA (Noirs)")
        elif self.game_mode == "human_vs_human":
            self.color_frame.pack_forget()
            self.board_widget.set_interactive(True)
            self._update_player_names("Joueur 1", "Joueur 2")
        else:
            self.color_frame.pack(fill="x", pady=(10, 0), before=self.new_game_btn)
            self.board_widget.set_interactive(True)
            self._update_player_names_for_human_ai()

        self._new_game()

    def _on_color_change(self) -> None:
        """Handle player color change."""
        self.human_color = (
            chess.WHITE if self.color_var.get() == "white" else chess.BLACK
        )
        self.board_widget.set_player_color(self.human_color)

        if (self.human_color == chess.BLACK) != self.board_widget.flipped:
            self._flip_board()

        self._update_player_names_for_human_ai()
        self._new_game()

    def _update_player_names(self, white_name: str, black_name: str) -> None:
        """Update player info names."""
        self.white_player_info.set_name(white_name)
        self.black_player_info.set_name(black_name)
        names: dict[bool, str] = self._side_names()
        self.win_panel.set_names(names[chess.WHITE], names[chess.BLACK])

    def _update_player_names_for_human_ai(self) -> None:
        """Update player names for human vs AI mode."""
        if self.human_color == chess.WHITE:
            self._update_player_names("Vous", "IA NeuralMate")
        else:
            self._update_player_names("IA NeuralMate", "Vous")

    def _new_game_with_confirm(self) -> None:
        """Start a new game with confirmation."""
        board: chess.Board = self.board_widget.get_board()
        if len(board.move_stack) > 0 and not board.is_game_over():
            create_confirmation_dialog(
                self.root,
                "Nouvelle partie ?",
                "La partie en cours sera perdue.",
                on_confirm=self._new_game,
                confirm_text="Recommencer",
                cancel_text="Continuer la partie",
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
        self.game_id += 1
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=0.5)

        self.ai_cancelled = False
        self.ai_thinking = False

        self._hide_thinking()

        # Clear MCTS tree for new game
        if self.mcts is not None:
            self.mcts.clear_cache()

        self.undone_moves.clear()

        self.board_widget.reset()

        # Reset game history for new game
        self.game_positions = [self.board_widget.get_board().copy()]

        # Reset all analysis panels
        self._reset_analysis()

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
        board: chess.Board = self.board_widget.get_board()
        move_stack: list[chess.Move] = list(board.move_stack)

        # Replay from the game's starting position (may be a custom FEN)
        temp_board: chess.Board = board.root()
        # Starting position
        self.game_positions = [temp_board.copy()]

        for move in move_stack:
            temp_board.push(move)
            self.game_positions.append(temp_board.copy())

    def _back_one_move(self) -> None:
        """Go back one move (or move pair in human vs AI)."""
        if self.ai_thinking or self.board_widget.is_animating:
            return

        board: chess.Board = self.board_widget.get_board()
        if not board.move_stack:
            return

        # Clear MCTS tree when going back (tree is no longer valid)
        if self.mcts is not None:
            self.mcts.clear_cache()

        self.ai_cancelled = True
        anim_duration: int = 100

        if self.game_mode == "human_vs_ai":
            if board.turn == self.human_color:
                if len(board.move_stack) >= 2:
                    ai_move: chess.Move = board.peek()
                    self.undone_moves.append((board.move_stack[-2], ai_move))

                    def after_ai_undo() -> None:
                        """Pop the AI move, then animate the human move undo."""
                        board.pop()
                        human_move: chess.Move = board.peek()

                        def after_human_undo() -> None:
                            """Pop the human move and refresh the board."""
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
                    human_move: chess.Move = board.peek()
                    self.undone_moves.append((human_move, None))

                    def after_undo() -> None:
                        """Pop the human move and refresh the board."""
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
            last_move: chess.Move = board.peek()
            self.undone_moves.append((last_move, None))

            def after_undo() -> None:
                """Pop the last move and refresh the board."""
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

        board: chess.Board = self.board_widget.get_board()
        human_move: chess.Move
        ai_move: chess.Move | None
        human_move, ai_move = self.undone_moves.pop()

        if human_move not in board.legal_moves:
            self.undone_moves.clear()
            return

        anim_duration: int = 100

        def after_human_move() -> None:
            """Replay the human move, then replay the AI move if present."""
            board.push(human_move)
            self.board_widget.set_board(board)

            if ai_move and ai_move in board.legal_moves:
                confirmed_ai_move: chess.Move = ai_move

                def after_ai_move() -> None:
                    """Replay the AI move and trigger next AI turn if needed."""
                    board.push(confirmed_ai_move)
                    self.board_widget.set_board(board)
                    self.board_widget.set_last_move(confirmed_ai_move)
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
                    confirmed_ai_move,
                    duration_ms=anim_duration,
                    on_complete=after_ai_move,
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
        self._layout_player_bars()

    # ==================== Time Control Methods ====================

    def _init_time_control(self) -> None:
        """Initialize timers for a new game with time control."""
        if self.time_control is None:
            return
        self.white_time = float(self.time_control)
        self.black_time = float(self.time_control)
        # Timer doesn't run until first move
        self.game_started = False

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

        board: chess.Board = self.board_widget.get_board()
        if board.is_game_over():
            return

        # Calculate elapsed time since turn started
        elapsed: float = time.time() - self.turn_start_time
        current_turn: bool = board.turn

        # Compute display time (remaining - elapsed)
        if current_turn == chess.WHITE:
            display_time: float = max(0, self.white_time - elapsed)
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
        self, color: chess.Color, time_remaining: float | None = None
    ) -> None:
        """Update the timer display in PlayerInfo widget."""
        if time_remaining is None:
            time_remaining = (
                self.white_time if color == chess.WHITE else self.black_time
            )

        # Format as MM:SS
        minutes: int = int(time_remaining // 60)
        seconds: int = int(time_remaining % 60)
        time_str: str = f"{minutes:02d}:{seconds:02d}"

        # Color based on urgency
        timer_color: str
        if time_remaining < 30:
            timer_color = COLORS.get("timer_urgent", "#ef4444")
        elif time_remaining < 60:
            timer_color = COLORS.get("timer_warning", "#f59e0b")
        else:
            timer_color = COLORS["text_primary"]

        # Update the correct PlayerInfo widget
        player_info: PlayerInfo = (
            self.white_player_info if color == chess.WHITE else self.black_player_info
        )
        player_info.set_timer(time_str, timer_color)

    def _commit_time_for_turn(self, color: chess.Color) -> None:
        """Deduct time used for the completed turn."""
        if self.time_control is None:
            return
        elapsed: float = time.time() - self.turn_start_time
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
        self._hide_thinking()

        loser: str = "Blancs" if color == chess.WHITE else "Noirs"
        winner: str = "Noirs" if color == chess.WHITE else "Blancs"

        self._set_status(f"Temps écoulé — victoire des {winner}", "over")
        self._show_result_dialog(
            self._result_title(not color),
            f"Les {loser} n'ont plus de temps : les {winner} gagnent au temps.",
        )

    @staticmethod
    def _estimate_moves_remaining(board: chess.Board) -> int:
        """Estimate remaining moves based on game phase."""
        pieces: int = len(board.piece_map())

        # Opening (> 28 pieces): ~28 moves remaining
        if pieces > 28:
            return 28

        # Middlegame (16-28 pieces): linear estimation
        elif pieces > 16:
            return int(15 + (pieces - 16) * 0.8)

        # Endgame (< 16 pieces): can be long
        else:
            # Pawn presence = longer endgame
            pawns: int = len(board.pieces(chess.PAWN, chess.WHITE)) + len(
                board.pieces(chess.PAWN, chess.BLACK)
            )
            if pawns > 4:
                # Pawn endgames are long
                return max(20, pieces * 2)
            # Light piece endgames
            return max(10, pieces)

    def _get_smoothed_time_per_sim(self) -> float:
        """Get exponentially smoothed time per simulation from history."""
        if not self.ai_move_times or not self.ai_sims_used:
            return 0.001  # Fallback

        n: int = min(len(self.ai_move_times), self.HISTORY_WINDOW)
        total_weighted_time: float = 0.0
        total_weighted_sims: float = 0.0
        weight: float = 1.0

        for i in range(n):
            idx: int = -(i + 1)
            total_weighted_time += self.ai_move_times[idx] * weight
            total_weighted_sims += self.ai_sims_used[idx] * weight
            weight *= self.DECAY_FACTOR

        return total_weighted_time / max(1, total_weighted_sims)

    def _get_wdl_factor(self) -> float:
        """Get time factor based on WDL entropy (position uncertainty)."""
        if self.last_wdl is None:
            return 1.0

        win: float
        draw: float
        loss: float
        win, draw, loss = self.last_wdl

        # Shannon entropy (max ~1.58 for uniform distribution)
        entropy: float = 0.0
        for p in [win, draw, loss]:
            if p > 0.001:
                entropy -= p * math.log2(p)

        # Gentle adjustments only (0.9 - 1.1 range)
        # Low entropy = clear position = slightly less time
        # High entropy = uncertain = slightly more time
        if entropy < 0.3:
            # Very clear (winning/losing)
            return 0.9
        elif entropy > 1.4:
            # Very uncertain
            return 1.1

        return 1.0

    def _adjust_for_complexity(self, base_sims: int, board: chess.Board) -> int:
        """Adjust simulations based on position complexity from MCTS data."""
        if self.mcts is None:
            return base_sims

        try:
            branching: int = self.mcts.get_max_branching_factor(board)
            depth: int = self.mcts.get_tree_depth(board)

            # Simple position (few branches, deep search) - gentle reduction
            if branching < 5 and depth > 15:
                return int(base_sims * 0.9)

            # Complex position (many branches) - slight increase
            if branching > 25:
                return int(base_sims * 1.1)
        except Exception:
            # MCTS methods may fail if tree is empty
            pass

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
        ai_color: bool = chess.BLACK if self.human_color == chess.WHITE else chess.WHITE
        ai_time: float = self.black_time if ai_color == chess.BLACK else self.white_time
        board: chess.Board = self.board_widget.get_board()

        if ai_time <= 0:
            # Absolute minimum
            return 30

        # Get smoothed time per simulation
        time_per_sim: float = self._get_smoothed_time_per_sim()

        # EMERGENCY MODE: < 15 seconds
        if ai_time < self.EMERGENCY_THRESHOLD:
            emergency_sims: int = max(30, int(ai_time / time_per_sim * 0.8))
            return min(emergency_sims, self.MIN_SIMULATIONS)

        # Estimate remaining moves (phase-aware)
        moves_left: int = self._estimate_moves_remaining(board)

        # Time budget per move
        time_budget: float = ai_time / moves_left

        # Max simulations we can afford
        max_sims: int
        if time_per_sim > 0:
            max_sims = int(time_budget / time_per_sim)
        else:
            max_sims = self.base_simulations

        # Apply WDL entropy factor
        wdl_factor: float = self._get_wdl_factor()
        target_sims: int = int(max_sims * wdl_factor)

        # Adjust for position complexity (uses MCTS data)
        target_sims = self._adjust_for_complexity(target_sims, board)

        # Clamp to reasonable bounds (max 2x base when time allows)
        final_sims: int = max(
            self.MIN_SIMULATIONS, min(self.base_simulations, target_sims)
        )

        return final_sims

    def _update_ai_simulations_for_time(self) -> None:
        """Adapt AI simulations based on dynamic time management."""
        if self.time_control is None:
            return

        new_sims: int = self._get_adapted_simulations()
        print(f"New simulations: {new_sims}")

        if self.mcts is not None:
            self.mcts.num_simulations = new_sims

    # ==================== End Time Control Methods ====================

    def _on_human_move(self, move: chess.Move) -> None:
        """Handle human move."""
        # Start game timer on first move (don't commit time for first move)
        if self.time_control is not None and not self.game_started:
            self.game_started = True
            # Start timing from now
            self.turn_start_time = time.time()
            self._start_timer_loop()
        else:
            # Commit time for human's turn (time control)
            self._commit_time_for_turn(self.human_color)

        # Record human thinking time
        self.last_move_time = time.time() - self.turn_start_time

        self.undone_moves.clear()
        self._update_game_info()

        # Update game history
        board: chess.Board = self.board_widget.get_board()
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
            ai_color: bool = (
                chess.BLACK if self.human_color == chess.WHITE else chess.WHITE
            )
            ai_time: float = (
                self.black_time if ai_color == chess.BLACK else self.white_time
            )

            if ai_time < self.INSTANT_MOVE_THRESHOLD:
                instant_move: chess.Move | None = self._get_instant_move()
                if instant_move is not None:
                    # Play immediately without search
                    self._apply_instant_move(instant_move)
                    return

        self.ai_thinking = True
        self.ai_cancelled = False

        self._show_thinking()

        self.ai_thread = threading.Thread(
            target=self._compute_ai_move, args=(self.game_id,), daemon=True
        )
        self.ai_thread.start()

    def _get_instant_move(self) -> chess.Move | None:
        """Get best move from current MCTS tree without searching."""
        board: chess.Board = self.board_widget.get_board()

        # Try to get move with most visits from existing tree
        if self.mcts is not None:
            try:
                visit_counts: dict[chess.Move, int] = self.mcts.get_visit_counts(board)
                if visit_counts:
                    best_move: chess.Move = max(
                        visit_counts, key=visit_counts.__getitem__
                    )
                    print(
                        f"Instant move from tree: {best_move} ({visit_counts[best_move]} visits)"
                    )
                    return best_move
            except Exception:
                pass

        # No tree: best move of the raw network policy (single evaluation)
        if self.mcts is not None:
            saved_sims: int = self.mcts.num_simulations
            self.mcts.num_simulations = 0
            try:
                history_boards: list[chess.Board] | None = (
                    self.game_positions[:-1] if len(self.game_positions) > 1 else None
                )
                policy_move: chess.Move | None = self.mcts.get_best_move(
                    board, history_boards=history_boards
                )
                if policy_move is not None:
                    print(f"Instant move from policy: {policy_move}")
                    return policy_move
            except Exception:
                pass
            finally:
                self.mcts.num_simulations = saved_sims

        # Fallback: random legal move
        legal_moves: list[chess.Move] = list(board.legal_moves)
        if legal_moves:
            import random

            move: chess.Move = random.choice(legal_moves)
            print(f"Instant move (random fallback): {move}")
            return move

        return None

    def _apply_instant_move(self, move: chess.Move) -> None:
        """Apply an instant move without full AI processing."""
        board: chess.Board = self.board_widget.get_board()

        # Commit AI time
        ai_color: bool = chess.BLACK if self.human_color == chess.WHITE else chess.WHITE
        self._commit_time_for_turn(ai_color)

        # Apply the move
        board.push(move)
        self.board_widget.set_board(board)
        self.board_widget.set_last_move(move)

        # Update game history
        self.game_positions.append(board.copy())

        # Update UI
        self._update_game_info()
        self.turn_start_time = time.time()

        # Check game over
        if board.is_game_over():
            self._show_game_over()

    def _compute_ai_move(self, game_id: int = -1) -> None:
        """Compute AI move in background thread."""
        try:
            board: chess.Board = self.board_widget.get_board()

            if board.is_game_over():
                return

            root_value: float = 0.0
            mcts_stats: list[dict[str, Any]] | None = None
            tree_data: list[dict[str, Any]] | None = None
            tree_depth: int = 0
            tree_branching: int = 0
            mate_in: int | None = None

            move: chess.Move | None = None

            if self.use_pure_mcts and self.pure_mcts_player is not None:
                move = self.pure_mcts_player.select_move(board)
            elif self.mcts is not None:
                # Pass full game history to MCTS (excluding current position)
                # IMPORTANT: MCTS does _initial_history[:history_length], taking first positions
                # So we pass all positions from game start in order [pos_initial, pos_after_move1, ...]
                history_boards: list[chess.Board] | None = (
                    self.game_positions[:-1] if len(self.game_positions) > 1 else None
                )
                move = self.mcts.get_best_move(
                    board, add_noise=False, history_boards=history_boards
                )

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
                            "game_id": game_id,
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
                update_type: str
                data: dict[str, Any] | str
                update_type, data = self.update_queue.get_nowait()

                if update_type == "ai_move" and isinstance(data, dict):
                    self._apply_ai_move(data)
                elif update_type == "training" and isinstance(data, dict):
                    if self.training_panel:
                        self.training_panel.update_progress(data)
                    if data.get("phase") == "move":
                        fen: str | None = data.get("fen")
                        if fen:
                            self._show_training_position(fen)
                elif update_type == "error" and isinstance(data, str):
                    self._hide_thinking()
                    self.ai_thinking = False
                    messagebox.showerror("Erreur", data)

        except queue.Empty:
            pass

        self.root.after(50, self._process_updates)

    def _show_training_position(self, fen: str) -> None:
        """Show a position from training on the board."""
        try:
            board: chess.Board = chess.Board(fen)
            self.board_widget.set_board(board)
        except Exception:
            pass

    def _apply_ai_move(self, data: dict[str, Any]) -> None:
        """Apply AI move to the board with animation."""
        # Ignore results computed for a previous game / position
        if data.get("game_id", self.game_id) != self.game_id:
            return
        game_id: int = self.game_id
        self.ai_thinking = False

        ai_color: bool = chess.BLACK if self.human_color == chess.WHITE else chess.WHITE

        # Start game timer on first move (AI plays first when human is Black)
        if self.time_control is not None and not self.game_started:
            self.game_started = True
            # Start timing from now
            self.turn_start_time = time.time()
            self._start_timer_loop()
        else:
            # Commit time for AI's turn (time control)
            self._commit_time_for_turn(ai_color)

        # Record AI thinking time
        self.last_move_time = time.time() - self.turn_start_time

        self._hide_thinking()

        move: chess.Move = data["move"]
        mcts_stats: list[dict[str, Any]] | None = data.get("mcts_stats")
        root_value: float = data.get("root_value", 0.0)
        tree_data: list[dict[str, Any]] | None = data.get("tree_data")
        tree_depth: int = data.get("tree_depth", 0)
        tree_branching: int = data.get("tree_branching", 0)
        mate_in: int | None = data.get("mate_in")

        board: chess.Board = self.board_widget.get_board()

        # Side that searched (the AI to move) and how to call each side
        mover: bool = board.turn
        names: dict[bool, str] = self._side_names()
        mover_name: str = names[mover]

        # Update MCTS panel with search statistics
        if mcts_stats:
            total_visits: int = sum(s["visits"] for s in mcts_stats)
            self.mcts_panel.update_stats(board, mcts_stats, total_visits)
            self.candidates_panel.update_moves(board, mcts_stats, mover_name)

            # Record stats for dynamic time management
            if self.time_control is not None:
                self.ai_move_times.append(self.last_move_time)
                self.ai_sims_used.append(total_visits)

                # Extract WDL from best move (first in list, sorted by visits)
                if mcts_stats and "wdl" in mcts_stats[0]:
                    wdl: np.ndarray = mcts_stats[0]["wdl"]
                    # wdl is numpy array [win, draw, loss]
                    self.last_wdl = (float(wdl[0]), float(wdl[1]), float(wdl[2]))

        # Update Search Tree panel
        if tree_data:
            self.search_tree_panel.update_tree(
                tree_data, tree_depth, tree_branching, mate_in
            )
            self.plan_panel.update_plan(
                board,
                tree_data,
                depth=tree_depth,
                mate_in=mate_in,
                mover_name=mover_name[0].upper() + mover_name[1:],
                opponent_name=names[not mover],
            )

        # Update Value label
        value_color: str
        if root_value > 0.05:
            value_color = COLORS["q_value_positive"]
        elif root_value < -0.05:
            value_color = COLORS["q_value_negative"]
        else:
            value_color = COLORS["text_primary"]
        self.value_label.configure(text=f"{root_value:+.2f}", fg=value_color)

        if move in board.legal_moves:

            # Extract WDL from best move for eval graph
            best_wdl: np.ndarray | None = None
            if mcts_stats and "wdl" in mcts_stats[0]:
                best_wdl = mcts_stats[0]["wdl"]
                if not isinstance(best_wdl, np.ndarray):
                    best_wdl = np.array(best_wdl, dtype=np.float32)

            # Forced mate, from the searching side's point of view
            mate_for: bool | None = None
            if mcts_stats:
                if mcts_stats[0].get("our_mate_in") is not None or mate_in:
                    mate_for = mover
                elif mcts_stats[0].get("opponent_mate_in") is not None:
                    mate_for = not mover
            total_sims: int = sum(s["visits"] for s in mcts_stats) if mcts_stats else 0

            def on_animation_complete() -> None:
                """Apply the AI move to the board and schedule the next turn."""
                if game_id != self.game_id:
                    return
                board.push(move)
                self.board_widget.set_board(board)
                self.board_widget.set_last_move(move)
                self._update_game_info()

                # Update game history
                self.game_positions.append(board.copy())

                # Tree reuse: preserve subtree for next search
                if self.mcts is not None:
                    self.mcts.advance_root(board)

                # Update evaluation after AI move (converted to White's view)
                if best_wdl is not None:
                    self._record_ai_evaluation(
                        ply=len(board.move_stack),
                        mover=mover,
                        wdl=best_wdl,
                        mate_for=mate_for,
                        total_sims=total_sims,
                    )

                updated_board: chess.Board = self.board_widget.get_board()
                if updated_board.is_game_over():
                    self._show_game_over()
                elif self.game_mode == "ai_vs_ai":
                    # Start timer for next AI
                    self.turn_start_time = time.time()
                    self.root.after(
                        300,
                        lambda: (
                            self._trigger_ai_move()
                            if game_id == self.game_id and self.game_mode == "ai_vs_ai"
                            else None
                        ),
                    )
                else:
                    # Start timer for human
                    self.turn_start_time = time.time()

            self.board_widget.animate_move(
                move, duration_ms=300, on_complete=on_animation_complete
            )
        else:
            self._update_game_info()

    def _update_eval_after_move(
        self, ai_value: float | None = None, wdl: np.ndarray | None = None
    ) -> None:
        """Update evaluation displays after a move."""
        # Only update eval graph after AI moves (when ai_value is provided)
        if ai_value is None:
            return

        # Use AI's evaluation directly (matches Value Prediction display)
        # Positive = AI thinks it's winning, Negative = AI thinks it's losing
        display_eval: float = ai_value

        # Update eval bar (scale from [-1,1] to [-10,10])
        self.eval_bar.set_evaluation(display_eval * 10)

        # Add to eval history and update graph
        self.eval_history.append(display_eval)
        self.wdl_history.append(wdl)
        self.eval_graph.add_evaluation(display_eval, wdl)

    def _record_ai_evaluation(
        self,
        ply: int,
        mover: bool,
        wdl: np.ndarray,
        mate_for: bool | None,
        total_sims: int,
    ) -> None:
        """
        Store and display the AI's evaluation after one of its moves.

        All beginner displays use White's point of view, so the bar and the
        graph mean the same thing whichever side the AI plays.

        Args:
            ply: Number of half-moves played (after the AI move)
            mover: Color of the AI that searched
            wdl: [win, draw, loss] for the side that searched
            mate_for: Side with a forced mate, if any
            total_sims: Number of simulations behind this estimate
        """
        w, d, l = float(wdl[0]), float(wdl[1]), float(wdl[2])
        wdl_white: tuple[float, float, float] = (
            (w, d, l) if mover == chess.WHITE else (l, d, w)
        )
        white_score: float = expected_score(wdl_white)
        if mate_for is not None:
            bar_value: float = 10.0 if mate_for == chess.WHITE else -10.0
        else:
            bar_value = (2 * white_score - 1) * 10

        self.eval_points = [p for p in self.eval_points if p[0] < ply]
        self.eval_points.append((ply, white_score, bar_value, wdl_white))

        # Raw history for the technical eval graph (value in [-1, 1])
        self.eval_history.append(2 * white_score - 1)
        self.wdl_history.append(np.array(wdl_white, dtype=np.float32))
        self.eval_graph.add_evaluation(
            2 * white_score - 1, np.array(wdl_white, dtype=np.float32)
        )

        self.eval_bar.set_evaluation(bar_value)
        self.advantage_graph.add_point(ply, white_score)
        self.win_panel.update_wdl(
            *wdl_white,
            subtitle=(
                f"Estimation de l'IA après avoir étudié {total_sims:,} positions.".replace(
                    ",", " "
                )
            ),
            mate_for=mate_for,
        )

    def _update_eval_for_position(self) -> None:
        """Update evaluation for current position (after back/forward)."""
        board: chess.Board = self.board_widget.get_board()
        move_index: int = len(board.move_stack)

        # Forget evaluations of positions that were taken back
        kept = [p for p in self.eval_points if p[0] <= move_index]
        if len(kept) != len(self.eval_points):
            self.eval_points = kept
            self.eval_history = [2 * p[1] - 1 for p in kept]
            self.wdl_history = [np.array(p[3], dtype=np.float32) for p in kept]
            self.eval_graph.set_history(self.eval_history, self.wdl_history)
            self.advantage_graph.trim(move_index)

        if self.eval_points:
            _, _, bar_value, wdl_white = self.eval_points[-1]
            self.eval_bar.set_evaluation(bar_value)
            self.win_panel.update_wdl(
                *wdl_white,
                subtitle="Dernière estimation de l'IA pour cette partie.",
            )
        else:
            self.eval_bar.clear()
            self.win_panel.clear()

        # Note: candidate moves / plan panels keep showing the last AI search

        self.move_list.set_current_move(move_index - 1)

    def _update_game_info(self) -> None:
        """Update all game information displays."""
        board: chess.Board = self.board_widget.get_board()

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
        side: str = "Blancs" if board.turn == chess.WHITE else "Noirs"
        human_turn: bool = (
            self.game_mode == "human_vs_ai" and board.turn == self.human_color
        )
        if board.is_checkmate():
            winner: str = "Noirs" if board.turn == chess.WHITE else "Blancs"
            self._set_status(f"Échec et mat — victoire des {winner}", "over")
        elif board.is_stalemate():
            self._set_status("Pat — partie nulle", "over")
        elif board.is_insufficient_material():
            self._set_status("Nulle — plus assez de pièces pour mater", "over")
        elif board.is_game_over():
            self._set_status("Partie nulle", "over")
        elif board.is_check():
            if human_turn:
                self._set_status("Échec ! Protégez votre roi", "check")
            else:
                self._set_status(f"Échec au roi des {side} !", "check")
        elif self.ai_thinking:
            self._set_status("L'IA réfléchit…", "thinking")
        elif human_turn:
            self._set_status("À vous de jouer", "turn")
        elif self.game_mode == "human_vs_human":
            self._set_status(f"Aux {side} de jouer", "turn")
        else:
            self._set_status(f"Aux {side} de jouer", "info")

    def _result_title(self, winner: bool | None) -> str:
        """Headline of the result dialog for a given winner (None = draw)."""
        if winner is None:
            return "Partie nulle"
        if self.game_mode == "human_vs_ai":
            return (
                "Bravo, vous avez gagné !"
                if winner == self.human_color
                else "L'IA a gagné"
            )
        return f"Victoire des {'Blancs' if winner == chess.WHITE else 'Noirs'}"

    def _show_game_over(self) -> None:
        """Show game over dialog."""
        board: chess.Board = self.board_widget.get_board()
        self._update_game_info()

        message: str
        winner: bool | None = None
        if board.is_checkmate():
            winner = not board.turn
            loser: str = "Blancs" if board.turn == chess.WHITE else "Noirs"
            message = f"Échec et mat : le roi des {loser} ne peut plus s'échapper."
        elif board.is_stalemate():
            message = (
                "Pat : le camp au trait n'a aucun coup légal mais n'est pas en "
                "échec. C'est une nulle."
            )
        elif board.is_insufficient_material():
            message = "Il ne reste plus assez de pièces pour faire échec et mat."
        elif board.can_claim_fifty_moves():
            message = (
                "Règle des 50 coups : aucune prise ni coup de pion depuis 50 coups."
            )
        elif board.is_repetition():
            message = "La même position s'est répétée trois fois."
        else:
            message = "La partie est terminée."

        # The analysis panel and the bar show the final result too
        self.win_panel.set_result(winner, self._result_title(winner), message)
        if winner is None:
            self.eval_bar.set_evaluation(0.0)
        else:
            self.eval_bar.set_evaluation(10.0 if winner == chess.WHITE else -10.0)

        self._show_result_dialog(self._result_title(winner), message)

    def _show_result_dialog(self, title: str, message: str) -> None:
        """Styled end-of-game dialog with a quick "play again" button."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Fin de partie")
        dialog.configure(bg=COLORS["bg_secondary"])
        dialog.transient(self.root)
        dialog.resizable(False, False)

        body = tk.Frame(dialog, bg=COLORS["bg_secondary"])
        body.pack(padx=36, pady=28)

        tk.Label(
            body,
            text="♚",
            font=("Segoe UI Symbol", 36),
            fg=COLORS["accent"],
            bg=COLORS["bg_secondary"],
        ).pack()
        tk.Label(
            body,
            text=title,
            font=FONTS["title"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_secondary"],
        ).pack(pady=(4, 6))
        tk.Label(
            body,
            text=message,
            font=FONTS["body"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_secondary"],
            wraplength=360,
            justify="center",
        ).pack(pady=(0, 20))

        buttons = tk.Frame(body, bg=COLORS["bg_secondary"])
        buttons.pack()

        def play_again() -> None:
            dialog.destroy()
            self._new_game()

        create_styled_button(
            buttons, "Revoir la partie", dialog.destroy, style="outline"
        ).pack(side="left", padx=(0, 8))
        create_styled_button(buttons, "Rejouer", play_again, style="accent").pack(
            side="left"
        )

        dialog.bind("<Escape>", lambda e: dialog.destroy())
        dialog.bind("<Return>", lambda e: play_again())

        dialog.update_idletasks()
        w: int = dialog.winfo_reqwidth()
        h: int = dialog.winfo_reqheight()
        x: int = self.root.winfo_rootx() + (self.root.winfo_width() - w) // 2
        y: int = self.root.winfo_rooty() + (self.root.winfo_height() - h) // 3
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
        dialog.grab_set()
        dialog.focus_set()

    def _start_training(self) -> None:
        """Start self-play training."""
        if not self.training_panel:
            return
        if self.training_thread and self.training_thread.is_alive():
            return

        self.training_cancelled = False
        self.training_iterations: int = self.training_panel.get_iterations()
        self.training_output_file: str = self.training_panel.get_output_file()

        try:
            from alphazero import AlphaZeroTrainer
            from config import TrainingConfig  # type: ignore[import-not-found]

            config: TrainingConfig = TrainingConfig(
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
        assert self.trainer is not None
        assert self.network is not None
        try:

            def callback(data: dict[str, Any]) -> None:
                """Forward training progress to the UI update queue."""
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
                policy_temperature=self.POLICY_TEMPERATURE,
            )

        except Exception as e:
            self.update_queue.put(("error", f"Training error: {e}"))

    def _stop_training(self) -> None:
        """Stop training."""
        self.training_cancelled = True

    # ==================== Editor Mode Methods ====================

    def _on_app_mode_change(self) -> None:
        """Handle switch between Play and Editor mode."""
        mode: str = self.app_mode_var.get()
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
            self.game_id += 1
            self.ai_thinking = False
            self._hide_thinking()

            # Editing must not be restricted to one side's pieces
            self.board_widget.set_interactive(True)

            # Update status
            self._set_status("Éditeur — placez les pièces librement", "editor")

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

            # Restore interactivity for the current game mode
            self.board_widget.set_interactive(self.game_mode != "ai_vs_ai")

            # Update status
            self._update_game_info()

    def _select_editor_piece(self, color: chess.Color, piece_type: int) -> None:
        """Select a piece from the palette to place on the board."""
        piece: chess.Piece = chess.Piece(piece_type, color)
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
        piece_symbols: dict[int, str] = {
            chess.KING: "K",
            chess.QUEEN: "Q",
            chess.ROOK: "R",
            chess.BISHOP: "B",
            chess.KNIGHT: "N",
            chess.PAWN: "P",
        }
        symbol: str = piece_symbols.get(piece_type, "")

        key: str
        if color == chess.WHITE:
            key = f"w{symbol}"
        else:
            key = f"b{symbol.lower()}"

        if key in self.editor_palette_buttons:
            self.editor_palette_buttons[key].configure(bg=COLORS["accent"])

    def _clear_palette_highlight(self) -> None:
        """Clear all palette button highlights."""
        for btn in self.editor_palette_buttons.values():
            btn.configure(bg=COLORS["palette_bg"])
        self.eraser_btn.configure(bg=COLORS["bg_tertiary"])

    def _on_editor_position_change(self) -> None:
        """Called when the board position changes in editor mode."""
        self._update_editor_fen()

    def _sync_editor_controls_from_board(self) -> None:
        """Sync editor controls (turn, castling) from current board state."""
        board: chess.Board = self.board_widget.get_board()

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
        board: chess.Board = self.board_widget.get_board().copy()

        # Apply turn from editor control
        board.turn = (
            chess.WHITE if self.editor_turn_var.get() == "white" else chess.BLACK
        )

        # Apply castling rights from editor controls
        castling: str = ""
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
        fen: str = self.editor_fen_var.get().strip()
        if not fen:
            messagebox.showwarning("FEN vide", "Collez d'abord un code FEN.")
            return

        try:
            board: chess.Board = chess.Board(fen)
            self.board_widget.set_board(board)
            self.board_widget._draw_board()

            # Sync controls from loaded FEN
            self._sync_editor_controls_from_board()

        except ValueError as e:
            messagebox.showerror("FEN invalide", f"Impossible de lire ce FEN :\n{e}")

    def _editor_clear_board(self) -> None:
        """Clear all pieces from the board."""
        board: chess.Board = self.board_widget.get_board()
        board.clear()
        self.board_widget.set_board(board)
        self.board_widget._draw_board()
        self._update_editor_fen()

    def _editor_reset_position(self) -> None:
        """Reset to the starting position."""
        board: chess.Board = chess.Board()
        self.board_widget.set_board(board)
        self.board_widget._draw_board()
        self._sync_editor_controls_from_board()

    def _start_game_from_editor(self) -> None:
        """Start a game from the current editor position."""
        board: chess.Board = self.board_widget.get_board().copy()

        # Apply turn from editor
        board.turn = (
            chess.WHITE if self.editor_turn_var.get() == "white" else chess.BLACK
        )

        # Apply castling rights
        castling: str = ""
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
        self.game_id += 1
        self.ai_cancelled = False
        if self.mcts is not None:
            self.mcts.clear_cache()

        # Reset game state
        self.undone_moves.clear()
        self._reset_analysis()
        self.game_positions = [board.copy()]

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

    @staticmethod
    def _validate_editor_position(board: chess.Board) -> bool:
        """Validate the editor position before starting a game."""
        errors: list[str] = []

        # Check for both kings
        white_kings: int = len(board.pieces(chess.KING, chess.WHITE))
        black_kings: int = len(board.pieces(chess.KING, chess.BLACK))

        if white_kings == 0:
            errors.append("Il manque le roi blanc")
        elif white_kings > 1:
            errors.append("Il y a plusieurs rois blancs")

        if black_kings == 0:
            errors.append("Il manque le roi noir")
        elif black_kings > 1:
            errors.append("Il y a plusieurs rois noirs")

        # Check for pawns on first/last rank
        for square in chess.SQUARES:
            piece: chess.Piece | None = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                rank: int = chess.square_rank(square)
                if rank == 0 or rank == 7:
                    errors.append(
                        f"Pion impossible en {chess.square_name(square)} "
                        "(un pion ne peut pas être sur la 1re ou la 8e rangée)"
                    )

        # Check if opponent king is in check (illegal position)
        if white_kings == 1 and black_kings == 1:
            # Temporarily flip turn to check if opponent is in check
            test_board: chess.Board = board.copy()
            test_board.turn = not board.turn
            if test_board.is_check():
                opponent: str = "blanc" if board.turn == chess.BLACK else "noir"
                errors.append(
                    f"Le roi {opponent} est en échec alors que ce n'est pas "
                    "à lui de jouer (position impossible)"
                )

        if errors:
            messagebox.showerror(
                "Position impossible",
                "La partie ne peut pas commencer :\n\n• " + "\n• ".join(errors),
            )
            return False

        return True

    # ==================== End Editor Mode Methods ====================

    def run(self) -> None:
        """Run the application main loop."""
        self.root.update_idletasks()
        width: int = self.root.winfo_width()
        height: int = self.root.winfo_height()
        x: int = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y: int = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"+{x}+{y}")

        # On screens that are barely taller than the window, use all the space
        if self.root.winfo_screenheight() < height + 160:
            try:
                self.root.state("zoomed")
            except tk.TclError:
                pass

        self.root.mainloop()


def main() -> None:
    """Main entry point for the chess game."""
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="NeuralMate Chess - AlphaZero"
    )
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

    args: argparse.Namespace = parser.parse_args()

    app: ChessGameApp = ChessGameApp(
        network_path=args.network,
        num_simulations=args.simulations,
        verbose=args.verbose,
        time_control=args.time,
    )
    app.run()


if __name__ == "__main__":
    main()
