"""
Configuration management for NeuralMate2.

Provides unified JSON configuration for pretraining and self-play training.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class NetworkConfig(object):
    """Neural network architecture configuration."""

    num_filters: int = 128
    num_residual_blocks: int = 8
    # Fixed at 3 (4 positions total)
    history_length: int = 3


@dataclass
class PretrainingConfig(object):
    """Pretraining configuration."""

    pgn_path: str = "data/lichess_elite_2020-08.pgn"
    # Multiple PGN files (overrides pgn_path)
    pgn_paths: list[str] | None = None
    output_path: str = "models/pretrained.pt"

    # Directory for chunk files
    chunks_dir: str = "data/chunks"
    # Number of positions per chunk file
    chunk_size: int = 20000
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 0.001
    validation_split: float = 0.1
    patience: int = 10
    min_elo: int = 2200
    max_games: int | None = 30000
    # Use all positions from max_games
    max_positions: int | None = None
    skip_first_n_moves: int = 8
    # Weight for value loss to prevent collapse
    value_loss_weight: float = 5.0
    # Entropy bonus to encourage policy diversity
    entropy_coefficient: float = 0.01
    # Label smoothing factor for cross-entropy loss
    label_smoothing: float = 0.05
    # Number of prefetch threads for chunk loading
    prefetch_workers: int = 2
    # Accumulate gradients over N steps (effective batch = batch_size * N)
    gradient_accumulation_steps: int = 1
    # Training dynamics (prevent catastrophic forgetting)
    # AdamW weight decay
    weight_decay: float = 1e-4
    # Gradient clipping max norm
    gradient_clip_norm: float = 1.0
    # ReduceLROnPlateau decay factor
    lr_decay_factor: float = 0.7
    # Epochs before LR decay triggers
    lr_decay_patience: int = 5
    # Minimum LR floor
    min_learning_rate: float = 1e-5
    # Keep last N checkpoints
    checkpoint_keep_last: int = 10

    # Anti-forgetting: Tactical Replay Buffer
    # Enable tactical position replay
    tactical_replay_enabled: bool = True
    # 20% of batch from replay buffer
    tactical_replay_ratio: float = 0.20
    # Min weight to store in buffer
    tactical_replay_threshold: float = 1.5
    # Max positions in buffer (35/chunk)
    tactical_replay_capacity: int = 200000

    # Anti-forgetting: Knowledge Distillation
    # Enable distillation from teacher network
    teacher_enabled: bool = True
    # Path to teacher network (e.g., epoch 1)
    teacher_path: str | None = None
    # Weight for soft loss (0.6 = 60% soft, 40% hard)
    teacher_alpha: float = 0.6
    # Softmax temperature for distillation
    teacher_temperature: float = 2.0

    # Anti-forgetting: EWC (Elastic Weight Consolidation)
    # Enable EWC regularization
    ewc_enabled: bool = True
    # EWC regularization strength
    ewc_lambda: float = 0.4
    # First epoch to apply EWC (after Fisher computed)
    ewc_start_epoch: int = 2
    # Samples for Fisher Information estimation
    ewc_fisher_samples: int = 10000

    # Testing
    # Limit chunks per epoch (for quick testing)
    max_chunks: int | None = None

    def get_pgn_paths(self) -> list[str]:
        """Get all PGN paths as a list (handles both legacy and new format)."""
        if self.pgn_paths:
            return self.pgn_paths
        return [self.pgn_path] if self.pgn_path else []


@dataclass
class TrainingConfig(object):
    """Self-play training configuration."""

    checkpoint_path: str = "checkpoints"
    iterations: int = 100
    games_per_iteration: int = 100
    num_simulations: int = 100
    # Batch size for MCTS GPU inference
    mcts_batch_size: int = 8
    batch_size: int = 256
    learning_rate: float = 0.01
    lr_decay: float = 0.95
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 3
    patience: int = 10
    buffer_size: int = 500000
    min_buffer_size: int = 5000
    recent_weight: float = 0.8
    arena_interval: int = 5
    arena_games: int = 20
    arena_simulations: int = 100
    # Win rate to replace best model
    win_threshold: float = 0.55
    max_moves: int = 200
    temperature_moves: int = 30
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    # Exploration constant for MCTS
    c_puct: float = 1.5
    # Must match NetworkConfig.history_length
    history_length: int = 3
    # Min score vs old/pretrained to avoid veto
    veto_threshold: float = 0.35
    # Save checkpoint every N iterations
    checkpoint_interval: int = 1
    # Path to pretrained model for arena comparison
    pretrained_path: str | None = None
    # Data mixing to prevent catastrophic forgetting
    # Ratio of pretrain data in each batch (0.4 = 40% pretrain, 60% self-play)
    pretrain_mix_ratio: float = 0.0
    # Path to pretrain chunks (same as pretraining.chunks_dir)
    pretrain_chunks_dir: str = "data/chunks"
    # Label smoothing for pretrain targets (0.1 = 10% smoothing)
    pretrain_label_smoothing: float = 0.1
    # Number of pretrain chunks to keep in memory
    pretrain_chunks_loaded: int = 15
    # Weight for KL divergence loss to pretrained model (0.1 recommended)
    kl_loss_weight: float = 0.0
    # Adaptive KL divergence control
    # Target KL divergence threshold
    kl_target: float = 0.15
    # Base KL weight when KL < target
    kl_weight_base: float = 0.1
    # Maximum KL weight when KL >> target
    kl_weight_max: float = 2.0
    # How aggressively weight scales above target
    kl_adaptive_factor: float = 10.0
    # Early warning thresholds
    # Trigger warning and boost pretrain mix
    kl_warning_threshold: float = 0.20
    # Force immediate arena evaluation
    kl_critical_threshold: float = 0.30
    # Veto recovery settings
    # Purge this ratio of buffer on veto
    veto_buffer_purge_ratio: float = 0.25
    # Iterations to boost pretrain mix after veto
    veto_recovery_iterations: int = 3
    # Self-play quality
    # Use best network for self-play (prevents buffer pollution)
    use_best_for_selfplay: bool = True
    # Veto escalation (prevents infinite rollback loops)
    # Reduce LR by this factor after 2+ vetoes
    veto_escalation_lr_factor: float = 0.5
    # Use current network for this ratio of games after 3+ vetoes
    veto_exploration_ratio: float = 0.3
    # Purge this ratio of buffer after 4+ vetoes
    veto_critical_purge_ratio: float = 0.5


@dataclass
class Config(object):
    """Main configuration container."""

    pretraining: PretrainingConfig = field(default_factory=PretrainingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)

    @classmethod
    def load(cls, path: str) -> Config:
        """
        Load configuration from JSON file.

        Args:
            path: Path to JSON config file.

        Returns:
            Config object with loaded values.
        """
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        config: Config = cls()

        # Load pretraining config
        if "pretraining" in data:
            for key, value in data["pretraining"].items():
                if hasattr(config.pretraining, key):
                    # Handle pgn_path as list (new format) or string (legacy)
                    if key == "pgn_path" and isinstance(value, list):
                        config.pretraining.pgn_paths = value
                        config.pretraining.pgn_path = value[0] if value else ""
                    else:
                        setattr(config.pretraining, key, value)

        # Load training config
        if "training" in data:
            for key, value in data["training"].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)

        # Load network config
        if "network" in data:
            for key, value in data["network"].items():
                if hasattr(config.network, key):
                    setattr(config.network, key, value)

        return config

    def save(self, path: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Path to save JSON file.
        """
        data: dict[str, dict[str, Any]] = {
            "pretraining": asdict(self.pretraining),
            "training": asdict(self.training),
            "network": asdict(self.network),
        }

        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Convert to dictionary."""
        return {
            "pretraining": asdict(self.pretraining),
            "training": asdict(self.training),
            "network": asdict(self.network),
        }

    @classmethod
    def default(cls) -> Config:
        """Create default configuration."""
        return cls()

    def print_summary(self, section: str | None = None) -> None:
        """Print configuration summary."""
        if section is None or section == "pretraining":
            print("\n[Pretraining]")
            for key, value in asdict(self.pretraining).items():
                print(f"  {key}: {value}")

        if section is None or section == "training":
            print("\n[Training]")
            for key, value in asdict(self.training).items():
                print(f"  {key}: {value}")

        if section is None or section == "network":
            print("\n[Network]")
            for key, value in asdict(self.network).items():
                print(f"  {key}: {value}")


def generate_default_config() -> str:
    """Generate default config as JSON string."""
    config: Config = Config.default()
    return json.dumps(config.to_dict(), indent=2)


def load_config(path: str) -> Config:
    """Convenience function to load config."""
    return Config.load(path)
