"""
AlphaZero trainer for self-play reinforcement learning.

Orchestrates:
1. Self-play game generation with MCTS
2. Training on collected examples
3. Arena evaluation against previous best
4. Checkpoint management
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import chess

from .network import DualHeadNetwork
from .mcts import MCTS
from .arena import Arena, NetworkPlayer, RandomPlayer, ArenaStats
from .replay_buffer import ReplayBuffer
from .checkpoint_manager import CheckpointManager
from .move_encoding import encode_move_from_perspective, MOVE_ENCODING_SIZE
from .spatial_encoding import PositionHistory, DEFAULT_HISTORY_LENGTH
from .device import get_device, supports_mixed_precision


def values_to_wdl_targets(values: torch.Tensor) -> torch.Tensor:
    """
    Convert scalar values to WDL class targets.

    Args:
        values: Tensor of shape (batch,) with values in {-1, 0, 1}
                1.0 = win, 0.0 = draw, -1.0 = loss

    Returns:
        Tensor of shape (batch,) with class indices:
        0 = win, 1 = draw, 2 = loss
    """
    # Convert: 1.0 -> 0, 0.0 -> 1, -1.0 -> 2
    wdl_targets = torch.round(1.0 - values).long()
    return torch.clamp(wdl_targets, min=0, max=2)


@dataclass
class TrainingConfig:
    """Configuration for AlphaZero training."""

    # Self-play settings
    games_per_iteration: int = 100
    num_simulations: int = 100
    mcts_batch_size: int = 8  # Batch size for MCTS GPU inference
    max_moves: int = 200

    # Training settings
    epochs_per_iteration: int = 3
    batch_size: int = 256
    learning_rate: float = 0.01
    lr_decay: float = 0.95
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4

    # Replay buffer
    buffer_size: int = 500000
    min_buffer_size: int = 5000
    recent_weight: float = 0.8

    # Arena evaluation
    arena_interval: int = 5  # Evaluate every N iterations
    arena_games: int = 20
    arena_simulations: int = 100
    win_threshold: float = 0.55  # Win rate to replace best model
    veto_threshold: float = 0.35  # Min score vs old/pretrained to avoid veto
    pretrained_path: Optional[str] = (
        None  # Path to pretrained model for arena comparison
    )

    # Checkpointing
    checkpoint_path: str = "checkpoints"
    checkpoint_interval: int = 1

    # MCTS settings
    temperature_moves: int = 30  # Use temperature=1 for first N moves
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    c_puct: float = 1.5  # Exploration constant (higher = more exploration)

    # History
    history_length: int = DEFAULT_HISTORY_LENGTH


class AlphaZeroTrainer:
    """
    AlphaZero training loop.

    Implements the self-play -> train -> evaluate cycle.
    """

    def __init__(
        self,
        network: DualHeadNetwork,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            network: Neural network to train.
            config: Training configuration.
        """
        self.network = network
        self.config = config or TrainingConfig()

        # Create components
        self._buffer = ReplayBuffer(
            max_size=self.config.buffer_size,
            recent_weight=self.config.recent_weight,
        )

        self.arena = Arena(
            num_games=self.config.arena_games,
            num_simulations=self.config.arena_simulations,
            max_moves=self.config.max_moves,
            history_length=self.config.history_length,
        )

        # Training state
        self._optimizer = optim.AdamW(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Mixed precision
        self._use_amp = supports_mixed_precision()
        self._scaler = GradScaler("cuda") if self._use_amp else None

        self._rng = np.random.default_rng(42)

        # Tracking
        self._iteration = 0
        self._best_network_path: Optional[str] = None
        self._best_network: Optional[DualHeadNetwork] = None
        self._best_iteration: int = 0
        self._old_checkpoints: list = []  # List of (iteration, path) tuples
        self._last_training_stats: Optional[dict] = None  # For checkpoint saving
        # Adaptive KL tracking
        self._last_avg_kl: float = 0.0  # Track KL for adaptive weight
        self._current_kl_weight: float = 0.1  # Current adaptive KL weight
        # Veto recovery tracking
        self._veto_recovery_remaining: int = 0  # Iterations remaining with boosted mix
        self._original_pretrain_mix: float = 0.0  # Original mix ratio before boost
        self._kl_warning_count: int = 0  # Consecutive KL warnings
        # Veto escalation tracking (prevents infinite rollback loops)
        self._consecutive_vetoes: int = 0  # Counter of consecutive vetoes
        self._exploration_ratio: float = 0.0  # Ratio of games using current network
        self._exploration_remaining: int = 0  # Iterations of forced exploration

        # Load pretrained model for arena comparison
        self._pretrained_network: Optional[DualHeadNetwork] = None
        if self.config.pretrained_path and os.path.exists(self.config.pretrained_path):
            try:
                self._pretrained_network = DualHeadNetwork.load(
                    self.config.pretrained_path
                )
                self._pretrained_network.eval()
                print(
                    f"Loaded pretrained model for arena: {self.config.pretrained_path}"
                )
            except Exception as e:
                print(f"Warning: Could not load pretrained model: {e}")

        # Checkpoint manager
        self._checkpoint_manager = CheckpointManager(
            self.config.checkpoint_path,
            keep_last_n=5,
            milestone_interval=5,
            verbose=True,
        )

        # Data mixing: load pretrain data if configured
        self._pretrain_states: Optional[np.ndarray] = None
        self._pretrain_policy_indices: Optional[np.ndarray] = None
        self._pretrain_values: Optional[np.ndarray] = None
        self._pretrain_mix_ratio = getattr(self.config, "pretrain_mix_ratio", 0.0)

        if self._pretrain_mix_ratio > 0:
            chunks_dir = getattr(self.config, "pretrain_chunks_dir", "data/chunks")
            self._load_pretrain_data(chunks_dir)

    def train_iteration(
        self,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """
        Run one training iteration.

        Steps:
        1. Self-play games
        2. Train on buffer
        3. Optional arena evaluation

        Args:
            callback: Optional callback for progress updates.

        Returns:
            Iteration statistics.
        """
        self._iteration += 1

        # Reseed RNG based on iteration (reproducible + works with resume)
        self._rng = np.random.default_rng(self._iteration)

        # Check if veto recovery period is over
        self._check_veto_recovery()

        stats = {
            "iteration": self._iteration,
            "selfplay_stats": {},
            "training_stats": {},
            "arena_stats": {},
        }

        # Phase 1: Self-play
        if callback:
            callback({"phase": "self_play_start", "iteration": self._iteration})

        selfplay_stats = self._run_self_play(callback)
        stats["selfplay_stats"] = selfplay_stats

        # Phase 2: Training
        if len(self._buffer) >= self.config.min_buffer_size:
            if callback:
                callback({"phase": "training_start", "iteration": self._iteration})

            training_stats = self._train_on_buffer(callback)
            stats["training_stats"] = training_stats
            self._last_training_stats = training_stats

            # Early warning: check KL divergence thresholds
            avg_kl = training_stats.get("avg_kl_loss", 0)
            kl_warning = getattr(self.config, "kl_warning_threshold", 0.20)
            kl_critical = getattr(self.config, "kl_critical_threshold", 0.30)

            if avg_kl > kl_critical:
                # Critical: force arena evaluation
                stats["force_arena"] = True
                self._kl_warning_count += 1
                if callback:
                    callback(
                        {
                            "phase": "kl_critical",
                            "kl_loss": avg_kl,
                            "threshold": kl_critical,
                            "message": "KL divergence critical - forcing arena evaluation",
                        }
                    )
            elif avg_kl > kl_warning:
                # Warning: boost pretrain mix
                self._kl_warning_count += 1
                if self._kl_warning_count >= 2 and self._pretrain_mix_ratio < 0.6:
                    old_mix = self._pretrain_mix_ratio
                    self._pretrain_mix_ratio = min(0.6, self._pretrain_mix_ratio + 0.1)
                    if callback:
                        callback(
                            {
                                "phase": "kl_warning",
                                "kl_loss": avg_kl,
                                "threshold": kl_warning,
                                "message": f"KL elevated - boosted pretrain mix {old_mix:.0%} -> {self._pretrain_mix_ratio:.0%}",
                            }
                        )
                elif callback:
                    callback(
                        {
                            "phase": "kl_warning",
                            "kl_loss": avg_kl,
                            "threshold": kl_warning,
                            "message": "KL divergence elevated - monitoring",
                        }
                    )
            else:
                # KL is healthy, reset warning count
                self._kl_warning_count = 0
        else:
            if callback:
                callback(
                    {
                        "phase": "training_skip",
                        "reason": f"Buffer size {len(self._buffer)} < {self.config.min_buffer_size}",
                    }
                )

        # Phase 3: Arena evaluation
        force_arena = stats.get("force_arena", False)
        if self._iteration % self.config.arena_interval == 0 or force_arena:
            if callback:
                callback({"phase": "arena_start", "iteration": self._iteration})

            arena_stats = self._run_arena(callback)
            stats["arena_stats"] = arena_stats

        # Checkpoint (only if training actually happened)
        if self._iteration % self.config.checkpoint_interval == 0:
            if "training_stats" in stats:
                self._save_checkpoint()
            elif callback:
                callback(
                    {
                        "phase": "checkpoint_skip",
                        "reason": "Training was skipped, no checkpoint saved",
                    }
                )

        # Update learning rate
        self._decay_learning_rate()

        stats["buffer_size"] = len(self._buffer)
        stats["learning_rate"] = self._optimizer.param_groups[0]["lr"]

        if callback:
            callback(
                {
                    "phase": "iteration_complete",
                    "iteration": self._iteration,
                    "stats": stats,
                }
            )

        return stats

    def _run_self_play(
        self,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Generate self-play games."""
        stats = {
            "games_played": 0,
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
            "avg_game_length": 0,
            "avg_game_time": 0.0,
            "examples_generated": 0,
            # Termination details
            "checkmates": 0,
            "stalemates": 0,
            "max_moves": 0,
            "other": 0,
        }

        total_moves = 0
        total_time = 0.0

        for game_idx in range(self.config.games_per_iteration):
            # Play one game
            game_start = time.time()
            game_data = self._play_self_play_game(game_idx, callback)
            game_time = time.time() - game_start
            total_time += game_time

            # Update stats
            stats["games_played"] += 1
            total_moves += game_data["moves"]

            if game_data["winner"] == chess.WHITE:
                stats["white_wins"] += 1
                outcome = 1.0
            elif game_data["winner"] == chess.BLACK:
                stats["black_wins"] += 1
                outcome = -1.0
            else:
                stats["draws"] += 1
                outcome = 0.0

            # Track termination type
            termination = game_data.get("termination", "other")
            if termination == "checkmate":
                stats["checkmates"] += 1
            elif termination == "stalemate":
                stats["stalemates"] += 1
            elif termination == "max_moves":
                stats["max_moves"] += 1
            else:
                stats["other"] += 1

            # Add to buffer
            self._buffer.add_game(
                game_data["states"],
                game_data["policies"],
                outcome,
            )
            stats["examples_generated"] += len(game_data["states"])

            if callback:
                callback(
                    {
                        "phase": "self_play",
                        "games_played": game_idx + 1,
                        "total_games": self.config.games_per_iteration,
                        "examples": len(self._buffer),
                        "white_wins": stats["white_wins"],
                        "black_wins": stats["black_wins"],
                        "draws": stats["draws"],
                        "avg_moves": total_moves / max(1, game_idx + 1),
                        "avg_time": total_time / max(1, game_idx + 1),
                        # Termination details
                        "checkmates": stats["checkmates"],
                        "stalemates": stats["stalemates"],
                        "max_moves_reached": stats["max_moves"],
                    }
                )

        stats["avg_game_length"] = total_moves / max(1, stats["games_played"])
        stats["avg_game_time"] = total_time / max(1, stats["games_played"])
        return stats

    def _play_self_play_game(
        self,
        game_idx: int,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Play a single self-play game."""
        board = chess.Board()
        history = PositionHistory(self.config.history_length)
        history.push(board)

        # Use best network for self-play if available (prevents buffer pollution)
        # Exploration mode: use current network for some games after repeated vetoes
        use_best = getattr(self.config, "use_best_for_selfplay", True)

        if self._exploration_ratio > 0 and self._rng.random() < self._exploration_ratio:
            # Exploration: use current network to generate diverse data
            selfplay_network = self.network
        elif use_best and self._best_network is not None:
            selfplay_network = self._best_network
        else:
            selfplay_network = self.network

        mcts = MCTS(
            selfplay_network,
            num_simulations=self.config.num_simulations,
            batch_size=self.config.mcts_batch_size,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
            history_length=self.config.history_length,
        )

        states = []
        policies = []
        move_count = 0

        while not board.is_game_over() and move_count < self.config.max_moves:
            # Set temperature based on move count
            if move_count < self.config.temperature_moves:
                mcts.temperature = 1.0
            else:
                mcts.temperature = 0.1

            # Get state encoding
            state = history.encode(from_perspective=True)
            states.append(state)

            # Run MCTS
            policy = mcts.search(board, add_noise=True)
            policies.append(policy)

            # Select move by sampling from policy (uses temperature for exploration)
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            # Get move probabilities for legal moves
            flip = board.turn == chess.BLACK
            move_probs = []
            for m in legal_moves:
                idx = encode_move_from_perspective(m, flip)
                if idx is not None and idx < len(policy):
                    move_probs.append(policy[idx])
                else:
                    move_probs.append(0.0)

            move_probs = np.array(move_probs)
            if move_probs.sum() > 0:
                move_probs = move_probs / move_probs.sum()
                move_idx = self._rng.choice(len(legal_moves), p=move_probs)
                move = legal_moves[move_idx]
            else:
                # Fallback to random if no valid probabilities
                move = self._rng.choice(legal_moves)

            # Get move info before pushing
            move_san = board.san(move)
            from_square = move.from_square
            to_square = move.to_square

            board.push(move)
            history.push(board)
            move_count += 1

            # Tree reuse: preserve subtree for deeper analysis
            mcts.advance_root(board)

            if callback:
                # Convert squares to (row, col) for GUI
                from_row = 7 - (from_square // 8)
                from_col = from_square % 8
                to_row = 7 - (to_square // 8)
                to_col = to_square % 8

                callback(
                    {
                        "phase": "board_update",
                        "fen": board.fen().split(" ")[0],  # Just piece positions
                        "last_move": ((from_row, from_col), (to_row, to_col)),
                        "move_san": move_san,
                        "move_number": move_count,
                        "game": game_idx,
                    }
                )

        # Determine winner
        if board.is_checkmate():
            winner = chess.BLACK if board.turn == chess.WHITE else chess.WHITE
            termination = "checkmate"
        elif board.is_stalemate():
            winner = None
            termination = "stalemate"
        elif move_count >= self.config.max_moves:
            winner = None
            termination = "max_moves"
        else:
            winner = None
            termination = "other"

        if callback:
            callback(
                {
                    "phase": "game_end",
                    "winner": (
                        "white"
                        if winner == chess.WHITE
                        else "black" if winner == chess.BLACK else "draw"
                    ),
                    "termination": termination,
                    "moves": move_count,
                    "game": game_idx,
                }
            )

        return {
            "states": states,
            "policies": policies,
            "winner": winner,
            "moves": move_count,
            "termination": termination,
        }

    def _train_on_buffer(
        self,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Train network on replay buffer."""
        self.network.train()
        device = get_device()

        # Enable TF32 for faster matmul on Ampere+ GPUs (RTX 30xx, 40xx)
        torch.set_float32_matmul_precision("high")

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        # Compute adaptive KL weight based on previous iteration's KL
        self._current_kl_weight = self._compute_adaptive_kl_weight(self._last_avg_kl)

        for epoch in range(self.config.epochs_per_iteration):
            # Number of batches per epoch
            num_samples = min(len(self._buffer), 10000)
            batches_per_epoch = num_samples // self.config.batch_size

            # Skip if not enough samples for even one batch
            if batches_per_epoch == 0:
                if callback:
                    callback(
                        {
                            "phase": "training_skip",
                            "reason": f"Not enough samples for batch ({num_samples} < {self.config.batch_size})",
                        }
                    )
                break

            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_kl_loss = 0.0

            for _ in range(batches_per_epoch):
                # Sample batch (with optional pretrain data mixing)
                if self._pretrain_mix_ratio > 0 and self._pretrain_states is not None:
                    states, policies, values = self._sample_mixed_batch(
                        self.config.batch_size
                    )
                else:
                    states, policies, values = self._buffer.sample(
                        self.config.batch_size
                    )

                # Convert to tensors (non_blocking for async CPU->GPU transfer)
                states_t = (
                    torch.from_numpy(states).float().to(device, non_blocking=True)
                )
                policies_t = (
                    torch.from_numpy(policies).float().to(device, non_blocking=True)
                )
                values_t = (
                    torch.from_numpy(values).float().to(device, non_blocking=True)
                )

                # Forward pass with optional mixed precision
                self._optimizer.zero_grad()

                if self._use_amp:
                    with autocast(device_type="cuda"):
                        pred_policies, _, wdl_logits = self.network(states_t)
                        policy_loss = self._policy_loss(pred_policies, policies_t)
                        value_loss = self._value_loss(wdl_logits, values_t)
                        loss = policy_loss + value_loss

                        # Add KL divergence loss to prevent catastrophic forgetting
                        kl_loss = None
                        if self._pretrained_network is not None:
                            kl_loss = self._kl_divergence_loss(pred_policies, states_t)
                            loss = loss + self._current_kl_weight * kl_loss

                    self._scaler.scale(loss).backward()
                    # Note: unscale_() is only needed if doing gradient clipping
                    # step() handles unscaling automatically
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    pred_policies, pred_values, wdl_logits = self.network(states_t)
                    policy_loss = self._policy_loss(pred_policies, policies_t)
                    value_loss = self._value_loss(wdl_logits, values_t)
                    loss = policy_loss + value_loss

                    # Add KL divergence loss to prevent catastrophic forgetting
                    kl_loss = None
                    if self._pretrained_network is not None:
                        kl_loss = self._kl_divergence_loss(pred_policies, states_t)
                        loss = loss + self._current_kl_weight * kl_loss

                    loss.backward()
                    self._optimizer.step()

                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                if kl_loss is not None:
                    epoch_kl_loss += kl_loss.item()
                num_batches += 1

            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss
            total_kl_loss += epoch_kl_loss
            total_loss += (
                epoch_policy_loss
                + epoch_value_loss
                + self._current_kl_weight * epoch_kl_loss
            )

            if callback:
                callback(
                    {
                        "phase": "training",
                        "epoch": epoch + 1,
                        "epochs": self.config.epochs_per_iteration,
                        "policy_loss": epoch_policy_loss / max(1, batches_per_epoch),
                        "value_loss": epoch_value_loss / max(1, batches_per_epoch),
                        "kl_loss": epoch_kl_loss / max(1, batches_per_epoch),
                        "kl_weight": self._current_kl_weight,
                        "total_loss": (
                            epoch_policy_loss
                            + epoch_value_loss
                            + self._current_kl_weight * epoch_kl_loss
                        )
                        / max(1, batches_per_epoch),
                    }
                )

        self.network.eval()

        # Update KL tracking for next iteration's adaptive weight
        avg_kl_loss = total_kl_loss / max(1, num_batches)
        self._last_avg_kl = avg_kl_loss

        return {
            "avg_policy_loss": total_policy_loss / max(1, num_batches),
            "avg_value_loss": total_value_loss / max(1, num_batches),
            "avg_kl_loss": avg_kl_loss,
            "avg_total_loss": total_loss / max(1, num_batches),
            "num_batches": num_batches,
            "kl_weight": self._current_kl_weight,
        }

    def _policy_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss for policy."""
        return torch.nn.functional.cross_entropy(pred, target)

    def _value_loss(
        self,
        wdl_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Value loss: WDL cross-entropy.

        Args:
            wdl_logits: WDL logits (batch, 3)
            target: Target values in {-1, 0, 1}

        Returns:
            Loss tensor
        """
        wdl_targets = values_to_wdl_targets(target)
        return torch.nn.functional.cross_entropy(wdl_logits, wdl_targets)

    def _kl_divergence_loss(
        self,
        current_policy: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence loss to keep policy close to pretrained model.

        This prevents catastrophic forgetting by penalizing deviation
        from pretrained policy distributions.
        """
        if self._pretrained_network is None:
            return torch.tensor(0.0, device=states.device)

        with torch.no_grad():
            pretrain_policy, _, _ = self._pretrained_network(states)
            pretrain_policy = torch.nn.functional.softmax(pretrain_policy, dim=-1)

        current_log_policy = torch.nn.functional.log_softmax(current_policy, dim=-1)
        return torch.nn.functional.kl_div(
            current_log_policy, pretrain_policy, reduction="batchmean"
        )

    def _compute_adaptive_kl_weight(self, current_kl: float) -> float:
        """
        Compute adaptive KL weight based on current divergence.

        When KL is below target, use base weight. When above, scale up
        progressively to create a "soft wall" against drift.
        """
        kl_target = getattr(self.config, "kl_target", 0.15)
        kl_weight_base = getattr(self.config, "kl_weight_base", 0.1)
        kl_weight_max = getattr(self.config, "kl_weight_max", 2.0)
        kl_adaptive_factor = getattr(self.config, "kl_adaptive_factor", 10.0)

        # If using old-style single kl_loss_weight, fall back to it
        if self.config.kl_loss_weight > 0 and kl_weight_base == 0.1:
            kl_weight_base = self.config.kl_loss_weight

        if current_kl <= kl_target:
            return kl_weight_base

        # Scale weight progressively above target
        excess_ratio = (current_kl - kl_target) / kl_target
        scaled_weight = kl_weight_base * (1 + kl_adaptive_factor * excess_ratio)
        return min(scaled_weight, kl_weight_max)

    def _handle_veto(
        self,
        stats,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> None:
        """
        Handle veto by rolling back to best known state with escalation.

        When veto is triggered (catastrophic forgetting detected):
        1. Restore network weights from best_network
        2. Purge recent buffer entries (likely from degraded model)
        3. Reset optimizer (prevents carrying bad momentum)
        4. Temporarily boost pretrain mix ratio

        Escalation levels for repeated vetoes:
        - 2+ vetoes: Reduce learning rate by 50%
        - 3+ vetoes: Enable exploration mode (30% current network for self-play)
        - 4+ vetoes: Critical purge (50% buffer) + max pretrain mix (80%)
        """
        if not stats.veto:
            return

        # Increment consecutive veto counter
        self._consecutive_vetoes += 1

        recovery_actions = []

        # 1. Restore network weights from best_network
        if self._best_network is not None:
            self.network.load_state_dict(self._best_network.state_dict())
            recovery_actions.append(
                f"Restored network to iteration {self._best_iteration}"
            )

        # 2. Purge recent buffer entries (they came from degraded model)
        purge_ratio = getattr(self.config, "veto_buffer_purge_ratio", 0.25)
        if purge_ratio > 0:
            purged = self._buffer.purge_recent(purge_ratio)
            recovery_actions.append(
                f"Purged {purged} buffer entries ({purge_ratio:.0%})"
            )

        # 3. Reset optimizer momentum (prevents carrying bad gradients)
        current_lr = self._optimizer.param_groups[0]["lr"]
        self._optimizer = optim.AdamW(
            self.network.parameters(),
            lr=current_lr,
            weight_decay=self.config.weight_decay,
        )
        # Also reset AMP scaler if using mixed precision
        if self._use_amp:
            self._scaler = GradScaler("cuda")
        recovery_actions.append("Reset optimizer state")

        # 4. Boost pretrain mix temporarily
        veto_recovery_iters = getattr(self.config, "veto_recovery_iterations", 3)
        if veto_recovery_iters > 0 and self._pretrain_mix_ratio > 0:
            self._veto_recovery_remaining = veto_recovery_iters
            self._original_pretrain_mix = self._pretrain_mix_ratio
            self._pretrain_mix_ratio = min(0.6, self._pretrain_mix_ratio + 0.2)
            recovery_actions.append(
                f"Boosted pretrain mix: {self._original_pretrain_mix:.0%} -> {self._pretrain_mix_ratio:.0%} for {veto_recovery_iters} iterations"
            )

        # === ESCALATION LEVELS ===

        # Escalation level 2: Reduce learning rate after 2+ consecutive vetoes
        if self._consecutive_vetoes >= 2:
            lr_factor = getattr(self.config, "veto_escalation_lr_factor", 0.5)
            new_lr = max(self.config.min_learning_rate, current_lr * lr_factor)
            for pg in self._optimizer.param_groups:
                pg["lr"] = new_lr
            recovery_actions.append(
                f"[Escalation L2] Reduced LR: {current_lr:.2e} -> {new_lr:.2e}"
            )

        # Escalation level 3: Enable exploration mode after 3+ consecutive vetoes
        if self._consecutive_vetoes >= 3:
            exploration_ratio = getattr(self.config, "veto_exploration_ratio", 0.3)
            self._exploration_ratio = exploration_ratio
            self._exploration_remaining = veto_recovery_iters
            recovery_actions.append(
                f"[Escalation L3] Enabled {exploration_ratio:.0%} exploration with current network"
            )

        # Escalation level 4: Critical measures after 4+ consecutive vetoes
        if self._consecutive_vetoes >= 4:
            critical_purge = getattr(self.config, "veto_critical_purge_ratio", 0.5)
            # Additional purge (on top of normal purge)
            extra_purged = self._buffer.purge_recent(critical_purge - purge_ratio)
            # Force max pretrain mix
            self._pretrain_mix_ratio = 0.8
            recovery_actions.append(
                f"[Escalation L4] CRITICAL: purged {extra_purged} more, pretrain mix=80%"
            )

        if callback:
            callback(
                {
                    "phase": "veto_recovery",
                    "reason": stats.veto_reason,
                    "consecutive_vetoes": self._consecutive_vetoes,
                    "actions": recovery_actions,
                }
            )

    def _check_veto_recovery(self) -> None:
        """Check if veto recovery period is over and restore pretrain mix."""
        if self._veto_recovery_remaining > 0:
            self._veto_recovery_remaining -= 1
            if self._veto_recovery_remaining == 0:
                # Restore original pretrain mix ratio
                self._pretrain_mix_ratio = self._original_pretrain_mix

        # Check if exploration mode period is over
        if self._exploration_remaining > 0:
            self._exploration_remaining -= 1
            if self._exploration_remaining == 0:
                self._exploration_ratio = 0.0

    def _load_pretrain_data(self, chunks_dir: str) -> None:
        """
        Initialize streaming pretrain data for data mixing.

        Uses streaming approach: keeps only a few chunks in memory at a time
        and periodically refreshes them. This prevents high RAM usage.
        """
        from pretraining.chunk_manager import ChunkManager

        if not os.path.exists(chunks_dir):
            print(f"Warning: Pretrain chunks dir not found: {chunks_dir}")
            return

        metadata = ChunkManager.load_metadata(chunks_dir)
        if not metadata:
            print(f"Warning: No pretrain metadata found in {chunks_dir}")
            return

        # Store chunk paths for streaming
        self._pretrain_chunk_paths = list(ChunkManager.iter_chunk_paths(chunks_dir))
        if not self._pretrain_chunk_paths:
            print("Warning: No pretrain chunks found")
            return

        self._pretrain_chunks_dir = chunks_dir
        self._pretrain_batch_count = 0
        self._pretrain_refresh_interval = 500  # Refresh chunks every N batches

        total_examples = metadata.get("total_examples", 0)
        num_chunks = len(self._pretrain_chunk_paths)

        # Load initial chunks (configurable, default 15 for ~300K positions)
        max_chunks_in_memory = min(self.config.pretrain_chunks_loaded, num_chunks)
        self._pretrain_loaded_chunk_indices = list(range(max_chunks_in_memory))
        self._load_pretrain_chunks()

        print(f"Pretrain streaming: {total_examples:,} examples in {num_chunks} chunks")
        print(
            f"  Loaded {max_chunks_in_memory} chunks in memory ({len(self._pretrain_states):,} positions)"
        )
        print(
            f"  Mix ratio: {self._pretrain_mix_ratio:.0%}, refresh every {self._pretrain_refresh_interval} batches"
        )

    def _load_pretrain_chunks(self) -> None:
        """Load the currently selected chunks into memory."""
        from pretraining.chunk_manager import ChunkManager

        all_states = []
        all_policy_indices = []
        all_values = []

        for chunk_idx in self._pretrain_loaded_chunk_indices:
            if chunk_idx < len(self._pretrain_chunk_paths):
                chunk_path = self._pretrain_chunk_paths[chunk_idx]
                try:
                    chunk = ChunkManager.load_chunk(chunk_path)
                    all_states.append(chunk["states"])
                    all_policy_indices.append(chunk["policy_indices"])
                    all_values.append(chunk["values"])
                except Exception:
                    continue

        if all_states:
            self._pretrain_states = np.concatenate(all_states, axis=0)
            self._pretrain_policy_indices = np.concatenate(all_policy_indices, axis=0)
            self._pretrain_values = np.concatenate(all_values, axis=0)

    def _maybe_refresh_pretrain_chunks(self) -> None:
        """Periodically refresh one chunk to cycle through all pretrain data."""
        self._pretrain_batch_count += 1

        if self._pretrain_batch_count % self._pretrain_refresh_interval != 0:
            return

        if not hasattr(self, "_pretrain_chunk_paths") or not self._pretrain_chunk_paths:
            return

        num_chunks = len(self._pretrain_chunk_paths)
        if num_chunks <= len(self._pretrain_loaded_chunk_indices):
            return  # All chunks already loaded

        # Replace oldest chunk with a new random one
        loaded_set = set(self._pretrain_loaded_chunk_indices)
        available = [i for i in range(num_chunks) if i not in loaded_set]

        if available:
            # Remove oldest, add new random chunk
            self._pretrain_loaded_chunk_indices.pop(0)
            new_chunk = self._rng.choice(available)
            self._pretrain_loaded_chunk_indices.append(new_chunk)
            self._load_pretrain_chunks()

    def _sample_mixed_batch(self, batch_size: int) -> tuple:
        """
        Sample a mixed batch from self-play buffer and pretrain data.

        Returns:
            Tuple of (states, policies, values) numpy arrays.
        """
        # Periodically refresh chunks to cycle through all pretrain data
        self._maybe_refresh_pretrain_chunks()

        # Calculate split
        pretrain_count = int(batch_size * self._pretrain_mix_ratio)
        selfplay_count = batch_size - pretrain_count

        # Sample from self-play buffer
        sp_states, sp_policies, sp_values = self._buffer.sample(selfplay_count)

        # Sample from pretrain data
        if pretrain_count > 0 and self._pretrain_states is not None:
            indices = self._rng.choice(
                len(self._pretrain_states), pretrain_count, replace=False
            )

            pt_states = self._pretrain_states[indices]
            pt_values = self._pretrain_values[indices]

            # Convert policy indices to policy vectors with label smoothing
            # This makes pretrain targets softer, closer to MCTS distributions
            label_smoothing = self.config.pretrain_label_smoothing
            smoothing_value = label_smoothing / MOVE_ENCODING_SIZE
            pt_policies = np.full(
                (pretrain_count, MOVE_ENCODING_SIZE), smoothing_value, dtype=np.float32
            )
            for i, idx in enumerate(self._pretrain_policy_indices[indices]):
                if idx < MOVE_ENCODING_SIZE:
                    pt_policies[i, idx] = 1.0 - label_smoothing + smoothing_value

            # Concatenate
            states = np.concatenate([sp_states, pt_states], axis=0)
            policies = np.concatenate([sp_policies, pt_policies], axis=0)
            values = np.concatenate([sp_values, pt_values], axis=0)

            # Shuffle to mix the batches
            shuffle_idx = self._rng.permutation(batch_size)
            states = states[shuffle_idx]
            policies = policies[shuffle_idx]
            values = values[shuffle_idx]
        else:
            states, policies, values = sp_states, sp_policies, sp_values

        return states, policies, values

    def _run_arena(
        self,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> ArenaStats:
        """Run full arena evaluation against Random, Best, and Old models."""
        # Select an old model for anti-forgetting check (if available)
        old_network = None
        old_iteration = None
        if self._old_checkpoints:
            old_iteration, old_path = self._select_old_checkpoint()
            if old_path and os.path.exists(old_path):
                try:
                    old_network = DualHeadNetwork.load(old_path)
                except Exception:
                    old_network = None

        # Run full evaluation
        stats = self.arena.run_full_evaluation(
            current_network=self.network,
            best_network=self._best_network,
            old_network=old_network,
            pretrained_network=self._pretrained_network,
            best_iteration=self._best_iteration,
            old_iteration=old_iteration,
            callback=callback,
            promotion_threshold=self.config.win_threshold,
            veto_threshold=self.config.veto_threshold,
        )

        # Handle promotion
        if stats.is_new_best and not stats.veto:
            # Save as new best
            self._best_network = DualHeadNetwork(
                num_filters=self.network.num_filters,
                num_residual_blocks=self.network.num_residual_blocks,
            )
            self._best_network.load_state_dict(self.network.state_dict())
            self._best_iteration = self._iteration

            # Reset escalation counters on successful promotion
            self._consecutive_vetoes = 0
            self._exploration_ratio = 0.0
            self._exploration_remaining = 0

            # Add current to old checkpoints list
            checkpoint_path = (
                self._checkpoint_manager.checkpoint_dir
                / f"iteration_{self._iteration}_network.pt"
            )
            self._old_checkpoints.append((self._iteration, str(checkpoint_path)))

        # Handle veto with recovery actions
        if stats.veto:
            self._handle_veto(stats, callback)

        if callback:
            callback(
                {
                    "phase": "arena_complete",
                    "elo": stats.elo,
                    "best_elo": stats.best_elo,
                    "is_new_best": stats.is_new_best,
                    "veto": stats.veto,
                    "veto_reason": stats.veto_reason,
                    "vs_random": {
                        "wins": stats.vs_random.wins if stats.vs_random else 0,
                        "losses": stats.vs_random.losses if stats.vs_random else 0,
                        "draws": stats.vs_random.draws if stats.vs_random else 0,
                        "score": stats.vs_random.score if stats.vs_random else 0,
                        "avg_time": stats.vs_random.avg_time if stats.vs_random else 0,
                    },
                    "vs_mcts": {
                        "wins": stats.vs_mcts.wins if stats.vs_mcts else 0,
                        "losses": stats.vs_mcts.losses if stats.vs_mcts else 0,
                        "draws": stats.vs_mcts.draws if stats.vs_mcts else 0,
                        "score": stats.vs_mcts.score if stats.vs_mcts else 0,
                        "avg_time": stats.vs_mcts.avg_time if stats.vs_mcts else 0,
                    },
                    "vs_best": (
                        {
                            "wins": stats.vs_best.wins if stats.vs_best else 0,
                            "losses": stats.vs_best.losses if stats.vs_best else 0,
                            "draws": stats.vs_best.draws if stats.vs_best else 0,
                            "score": stats.vs_best.score if stats.vs_best else 0,
                            "from_iteration": (
                                stats.vs_best.from_iteration if stats.vs_best else None
                            ),
                            "avg_time": stats.vs_best.avg_time if stats.vs_best else 0,
                        }
                        if stats.vs_best
                        else None
                    ),
                    "vs_old": (
                        {
                            "wins": stats.vs_old.wins if stats.vs_old else 0,
                            "losses": stats.vs_old.losses if stats.vs_old else 0,
                            "draws": stats.vs_old.draws if stats.vs_old else 0,
                            "score": stats.vs_old.score if stats.vs_old else 0,
                            "from_iteration": (
                                stats.vs_old.from_iteration if stats.vs_old else None
                            ),
                            "avg_time": stats.vs_old.avg_time if stats.vs_old else 0,
                        }
                        if stats.vs_old
                        else None
                    ),
                    "vs_pretrained": (
                        {
                            "wins": (
                                stats.vs_pretrained.wins if stats.vs_pretrained else 0
                            ),
                            "losses": (
                                stats.vs_pretrained.losses if stats.vs_pretrained else 0
                            ),
                            "draws": (
                                stats.vs_pretrained.draws if stats.vs_pretrained else 0
                            ),
                            "score": (
                                stats.vs_pretrained.score if stats.vs_pretrained else 0
                            ),
                            "avg_time": (
                                stats.vs_pretrained.avg_time
                                if stats.vs_pretrained
                                else 0
                            ),
                        }
                        if stats.vs_pretrained
                        else None
                    ),
                }
            )

        return stats

    def _select_old_checkpoint(self) -> tuple:
        """Select an old checkpoint for anti-forgetting check.

        70% chance: milestone (iter 5, 10, 25, 50, 100...)
        30% chance: recent model
        """
        import random as rand

        if not self._old_checkpoints:
            return None, None

        # Separate milestones and recent
        milestones = [(i, p) for i, p in self._old_checkpoints if i == 1 or i % 5 == 0]
        recent = self._old_checkpoints[-5:]  # Last 5

        if rand.random() < 0.7 and milestones:
            return rand.choice(milestones)
        elif recent:
            return rand.choice(recent)
        elif milestones:
            return rand.choice(milestones)
        else:
            return rand.choice(self._old_checkpoints)

    def _decay_learning_rate(self) -> None:
        """Apply learning rate decay."""
        for param_group in self._optimizer.param_groups:
            new_lr = max(
                param_group["lr"] * self.config.lr_decay,
                self.config.min_learning_rate,
            )
            param_group["lr"] = new_lr

    def _save_checkpoint(self) -> None:
        """Save training checkpoint with automatic cleanup."""
        # Training state
        state = {
            "iteration": self._iteration,
            "optimizer_state": self._optimizer.state_dict(),
            "learning_rate": self._optimizer.param_groups[0]["lr"],
        }
        if self._scaler is not None:
            state["scaler_state"] = self._scaler.state_dict()

        # Add training metrics (compatible with show_losses.py)
        if self._last_training_stats:
            state["train_loss"] = self._last_training_stats.get("avg_total_loss")
            state["train_policy"] = self._last_training_stats.get("avg_policy_loss")
            state["train_value"] = self._last_training_stats.get("avg_value_loss")

        # Save with checkpoint manager (handles cleanup)
        self._checkpoint_manager.save_training_checkpoint(
            iteration=self._iteration,
            network=self.network,
            state=state,
            buffer=self._buffer if len(self._buffer) > 0 else None,
        )

    def load_checkpoint(self, name: Optional[str] = None) -> bool:
        """
        Load a training checkpoint.

        Args:
            name: Checkpoint name (e.g., "iteration_5"), or None for latest.

        Returns:
            True if checkpoint was loaded successfully.
        """
        # Get iteration number
        if name is None:
            iteration = self._checkpoint_manager.get_latest_iteration()
            if iteration == 0:
                return False
        elif name.startswith("iteration_"):
            try:
                iteration = int(name.split("_")[1])
            except (IndexError, ValueError):
                return False
        else:
            return False

        # Load using checkpoint manager
        checkpoint = self._checkpoint_manager.load_training_checkpoint(
            iteration=iteration,
            load_buffer=True,
        )

        if checkpoint is None:
            return False

        # Load network
        self.network = DualHeadNetwork.load(checkpoint["network_path"])
        self._iteration = checkpoint["iteration"]

        # Recreate optimizer with new network's parameters
        # This is critical: the old optimizer points to the old network's params
        state = checkpoint.get("state")
        lr = self.config.learning_rate
        if state and "learning_rate" in state:
            lr = state["learning_rate"]

        self._optimizer = optim.AdamW(
            self.network.parameters(),
            lr=lr,
            weight_decay=self.config.weight_decay,
        )

        # Load optimizer state if available
        if state and "optimizer_state" in state:
            try:
                self._optimizer.load_state_dict(state["optimizer_state"])
            except Exception:
                pass  # Optimizer state may not match

        # Note: Don't restore scaler state - it can cause issues after resume
        # The scaler will auto-adjust its scale factor during training
        if self._scaler:
            self._scaler = GradScaler("cuda")  # Fresh scaler

        # Load buffer
        buffer = checkpoint.get("buffer")
        if buffer is not None:
            self._buffer = buffer

        return True

    def resume_or_start(self) -> int:
        """
        Resume from latest checkpoint or start fresh.

        Returns:
            Starting iteration number (0 if fresh start).
        """
        if self.load_checkpoint():
            print(f"Resumed from iteration {self._iteration}")
            return self._iteration
        else:
            print("Starting fresh training")
            return 0

    def train(
        self,
        num_iterations: int,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> list[dict]:
        """
        Run multiple training iterations.

        Args:
            num_iterations: Number of iterations to run.
            callback: Optional progress callback.

        Returns:
            List of iteration statistics.
        """
        all_stats = []

        for _ in range(num_iterations):
            stats = self.train_iteration(callback)
            all_stats.append(stats)

        return all_stats
