"""
Configuration management for NeuralMate2.

Provides unified JSON configuration for pretraining and self-play training.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""

    num_filters: int = 192
    num_residual_blocks: int = 12
    history_length: int = 3
    num_input_planes: int = 60  # Computed from history_length

    def __post_init__(self):
        # Compute input planes from history length
        # (history + 1) positions × 12 planes + 12 metadata planes (includes attack maps)
        self.num_input_planes = (self.history_length + 1) * 12 + 12


@dataclass
class PretrainingConfig:
    """Pretraining configuration."""

    pgn_path: str = "data/lichess_elite_2020-08.pgn"
    pgn_paths: Optional[List[str]] = None  # Multiple PGN files (overrides pgn_path)
    output_path: str = "models/pretrained.pt"

    chunks_dir: str = "data/chunks"  # Directory for chunk files
    chunk_size: int = 20000  # Number of positions per chunk file
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 0.001
    validation_split: float = 0.1
    patience: int = 10
    min_elo: int = 2200
    max_games: Optional[int] = 30000
    max_positions: Optional[int] = None  # Use all positions from max_games
    skip_first_n_moves: int = 8
    value_loss_weight: float = 5.0  # Weight for value loss to prevent collapse
    entropy_coefficient: float = 0.01  # Entropy bonus to encourage policy diversity
    prefetch_workers: int = 2  # Number of prefetch threads for chunk loading
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N steps (effective batch = batch_size * N)

    def get_pgn_paths(self) -> List[str]:
        """Get all PGN paths as a list (handles both legacy and new format)."""
        if self.pgn_paths:
            return self.pgn_paths
        return [self.pgn_path] if self.pgn_path else []


@dataclass
class TrainingConfig:
    """Self-play training configuration."""

    checkpoint_path: str = "checkpoints"
    iterations: int = 100
    games_per_iteration: int = 100
    num_simulations: int = 100
    mcts_batch_size: int = 16  # Batch size for MCTS GPU inference
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
    win_threshold: float = 0.55  # Win rate to replace best model
    max_moves: int = 200
    temperature_moves: int = 30
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    c_puct: float = 1.5  # Exploration constant for MCTS
    history_length: int = 3  # Must match NetworkConfig.history_length
    veto_threshold: float = 0.35  # Min score vs old/pretrained to avoid veto
    checkpoint_interval: int = 1  # Save checkpoint every N iterations
    pretrained_path: Optional[str] = None  # Path to pretrained model for arena comparison
    # Data mixing to prevent catastrophic forgetting
    pretrain_mix_ratio: float = 0.0  # Ratio of pretrain data in each batch (0.4 = 40% pretrain, 60% self-play)
    pretrain_chunks_dir: str = "data/chunks"  # Path to pretrain chunks (same as pretraining.chunks_dir)
    pretrain_label_smoothing: float = 0.1  # Label smoothing for pretrain targets (0.1 = 10% smoothing)
    pretrain_chunks_loaded: int = 15  # Number of pretrain chunks to keep in memory
    kl_loss_weight: float = 0.0  # Weight for KL divergence loss to pretrained model (0.1 recommended)
    # Adaptive KL divergence control
    kl_target: float = 0.15  # Target KL divergence threshold
    kl_weight_base: float = 0.1  # Base KL weight when KL < target
    kl_weight_max: float = 2.0  # Maximum KL weight when KL >> target
    kl_adaptive_factor: float = 10.0  # How aggressively weight scales above target
    # Early warning thresholds
    kl_warning_threshold: float = 0.20  # Trigger warning and boost pretrain mix
    kl_critical_threshold: float = 0.30  # Force immediate arena evaluation
    # Veto recovery settings
    veto_buffer_purge_ratio: float = 0.25  # Purge this ratio of buffer on veto
    veto_recovery_iterations: int = 3  # Iterations to boost pretrain mix after veto
    # Self-play quality
    use_best_for_selfplay: bool = True  # Use best network for self-play (prevents buffer pollution)
    # Veto escalation (prevents infinite rollback loops)
    veto_escalation_lr_factor: float = 0.5  # Reduce LR by this factor after 2+ vetoes
    veto_exploration_ratio: float = 0.3  # Use current network for this ratio of games after 3+ vetoes
    veto_critical_purge_ratio: float = 0.5  # Purge this ratio of buffer after 4+ vetoes


@dataclass
class Config:
    """Main configuration container."""

    pretraining: PretrainingConfig = field(default_factory=PretrainingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)

    @classmethod
    def load(cls, path: str) -> "Config":
        """
        Load configuration from JSON file.

        Args:
            path: Path to JSON config file.

        Returns:
            Config object with loaded values.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = cls()

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
            # Recompute input planes
            config.network.__post_init__()

        return config

    def save(self, path: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Path to save JSON file.
        """
        data = {
            "pretraining": asdict(self.pretraining),
            "training": asdict(self.training),
            "network": asdict(self.network),
        }

        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pretraining": asdict(self.pretraining),
            "training": asdict(self.training),
            "network": asdict(self.network),
        }

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls()

    def print_summary(self, section: Optional[str] = None) -> None:
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
    config = Config.default()
    return json.dumps(config.to_dict(), indent=2)


def load_config(path: str) -> Config:
    """Convenience function to load config."""
    return Config.load(path)
