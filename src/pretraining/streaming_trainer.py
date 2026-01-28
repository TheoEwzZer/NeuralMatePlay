"""
Streaming trainer that loads chunks incrementally during training.

This avoids loading all data into RAM at once, enabling training on datasets
larger than available memory.
"""

import concurrent.futures
import gc
import os
import random
import time
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

from .chunk_manager import ChunkManager

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphazero import DualHeadNetwork, get_device, supports_mixed_precision
from alphazero.checkpoint_manager import CheckpointManager
from alphazero.move_encoding import MOVE_ENCODING_SIZE


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
    # Formula: class = 1 - value (for discrete values)
    # But values might be continuous, so we use rounding
    wdl_targets = torch.round(1.0 - values).long()
    # Clamp to valid range [0, 2]
    return torch.clamp(wdl_targets, min=0, max=2)


def adaptive_label_smoothing(
    base_smoothing: float,
    weights: torch.Tensor,
    min_smoothing: float = 0.01,
    max_smoothing: float = 0.08,
) -> torch.Tensor:
    """
    Calculate adaptive label smoothing based on tactical weights.

    INVERSED: High weight → LOW smoothing (more confidence on tactical positions)
    Low weight → HIGH smoothing (regularization on simple positions)

    The intuition: tactical/rare positions (high weight) should have MORE confidence
    because these are the critical patterns the model must learn decisively.
    Simple positions can afford more smoothing/regularization.

    Args:
        base_smoothing: Base label smoothing value (e.g., 0.05)
        weights: Tactical weights tensor of shape (batch,), typically in [1.0, 5.0]
        min_smoothing: Minimum smoothing (floor)
        max_smoothing: Maximum smoothing (ceiling)

    Returns:
        Per-sample smoothing values of shape (batch,)

    Formula: smoothing = base / (1.0 + 0.5 * (weight - 1.0))
    Examples:
        weight=1.0 → smoothing=0.05 (base)
        weight=2.0 → smoothing=0.033
        weight=3.0 → smoothing=0.025
        weight=5.0 → smoothing=0.017
    """
    decay_factor = 1.0 / (1.0 + 0.5 * (weights - 1.0))
    adaptive = base_smoothing * decay_factor
    return torch.clamp(adaptive, min=min_smoothing, max=max_smoothing)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    KL divergence loss between student and teacher for knowledge distillation.

    Uses soft targets from teacher to transfer knowledge and prevent forgetting.

    Args:
        student_logits: Student model logits of shape (batch, num_classes)
        teacher_logits: Teacher model logits of shape (batch, num_classes)
        temperature: Softmax temperature (higher = softer distributions)

    Returns:
        Scalar KL divergence loss
    """
    student_log_probs = nn.functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=-1)
    return nn.functional.kl_div(
        student_log_probs, teacher_probs, reduction="batchmean"
    ) * (temperature**2)


class TacticalReplayBuffer:
    """
    Buffer for storing tactical/critical positions to prevent catastrophic forgetting.

    Positions with high tactical weights are stored and re-injected into training
    batches to ensure the model doesn't forget rare but important patterns.

    Uses proportional sampling: keeps top N positions from EACH chunk to ensure
    uniform coverage of the entire dataset.

    Optimized: Single contiguous array with doubling growth strategy.
    Starts small (1k) and doubles when full, up to max capacity.
    """

    def __init__(
        self,
        capacity: int,
        weight_threshold: float,
        num_chunks: int,
    ):
        """
        Initialize replay buffer with fixed pre-allocation.

        Args:
            capacity: Maximum number of positions to store
            weight_threshold: Minimum weight for a position to be stored
            num_chunks: Total number of chunks (for proportional allocation)
        """
        self.capacity = capacity
        self.threshold = weight_threshold
        self.positions_per_chunk = max(1, capacity // max(1, num_chunks))

        # Pre-allocate full capacity upfront (no growth = no warmup slowdown)
        self._allocated = capacity
        self.states = np.zeros((capacity, 72, 8, 8), dtype=np.float32)
        self.policies = np.zeros(capacity, dtype=np.int32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.weights = np.zeros(capacity, dtype=np.float32)

        self._size = 0  # Current number of positions in buffer

        # Reusable cache (avoid GPU memory fragmentation)
        self._chunk_cache = None
        self._chunk_cache_idx = 0

        # GPU cache pré-alloué (créé au premier appel de pre_sample_for_chunk)
        self._gpu_cache_states = None
        self._gpu_cache_policies = None
        self._gpu_cache_values = None
        self._gpu_cache_weights = None
        self._gpu_cache_size = 0
        self._gpu_cache_device = None
        self._chunk_cache_valid_size = 0

        # Random number generator
        self._rng = np.random.default_rng(42)

    def add_batch(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        weights: np.ndarray,
    ) -> int:
        """
        Add top N positions from a chunk based on tactical weight.

        Only keeps the positions with highest weights (most tactical/critical).
        This ensures uniform coverage across ALL chunks in the dataset.

        Args:
            states: State arrays of shape (batch, planes, 8, 8)
            policies: Policy indices of shape (batch,)
            values: Value targets of shape (batch,)
            weights: Tactical weights of shape (batch,)

        Returns:
            Number of positions added
        """
        # Filter by threshold
        mask = weights >= self.threshold
        if not mask.any():
            return 0

        # Get indices of tactical positions
        tactical_indices = np.nonzero(mask)[0]

        # Sort by weight (descending) and take top N
        tactical_weights = weights[tactical_indices]
        sorted_order = np.argsort(tactical_weights)[::-1]  # Descending
        top_indices = tactical_indices[sorted_order[: self.positions_per_chunk]]

        # Vectorized batch insert
        n_to_add = min(len(top_indices), self.capacity - self._size)
        if n_to_add > 0:
            insert_indices = top_indices[:n_to_add]
            end_idx = self._size + n_to_add
            self.states[self._size : end_idx] = states[insert_indices]
            self.policies[self._size : end_idx] = policies[insert_indices]
            self.values[self._size : end_idx] = values[insert_indices]
            self.weights[self._size : end_idx] = weights[insert_indices]
            self._size = end_idx

        return n_to_add

    def sample(
        self, n: int, device: torch.device
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Sample n positions from the buffer.

        Args:
            n: Number of positions to sample
            device: Device to place tensors on

        Returns:
            Tuple of (states, policy_indices, values, weights) or None if empty
        """
        if self._size == 0:
            return None

        n = min(n, self._size)
        indices = self._rng.choice(self._size, n, replace=False)

        # Direct numpy fancy indexing (fast, contiguous memory)
        sampled_states = torch.from_numpy(self.states[indices].copy()).to(
            device, non_blocking=True
        )

        sampled_policies = (
            torch.from_numpy(self.policies[indices].copy())
            .long()
            .to(device, non_blocking=True)
        )

        sampled_values = torch.from_numpy(self.values[indices].copy()).to(
            device, non_blocking=True
        )

        sampled_weights = torch.from_numpy(self.weights[indices].copy()).to(
            device, non_blocking=True
        )

        return sampled_states, sampled_policies, sampled_values, sampled_weights

    def _ensure_gpu_cache(self, size: int, device: torch.device) -> None:
        """Alloue ou réalloue le cache GPU si nécessaire."""
        if (
            self._gpu_cache_states is not None
            and self._gpu_cache_size >= size
            and self._gpu_cache_device == device
        ):
            return  # Cache déjà OK

        # Libérer l'ancien cache
        if self._gpu_cache_states is not None:
            del self._gpu_cache_states, self._gpu_cache_policies
            del self._gpu_cache_values, self._gpu_cache_weights
            torch.cuda.empty_cache()

        # Allouer avec 20% de marge
        alloc_size = int(size * 1.2)
        self._gpu_cache_states = torch.empty(
            (alloc_size, 72, 8, 8), dtype=torch.float32, device=device
        )
        self._gpu_cache_policies = torch.empty(
            alloc_size, dtype=torch.long, device=device
        )
        self._gpu_cache_values = torch.empty(
            alloc_size, dtype=torch.float32, device=device
        )
        self._gpu_cache_weights = torch.empty(
            alloc_size, dtype=torch.float32, device=device
        )
        self._gpu_cache_size = alloc_size
        self._gpu_cache_device = device

    def pre_sample_for_chunk(self, total_samples: int, device: torch.device) -> None:
        """
        Pre-sample avec tensors GPU persistants (temps constant).

        Uses CONTIGUOUS sampling for cache efficiency: picks a random start
        position and reads a contiguous block. Much faster than scattered indices.

        Args:
            total_samples: Total number of samples needed for the chunk
            device: Device to place tensors on
        """
        self._chunk_cache = None
        self._chunk_cache_idx = 0
        self._chunk_cache_valid_size = 0

        if self._size == 0 or total_samples <= 0:
            return

        total_samples = min(total_samples, self._size)

        # S'assurer que le cache GPU est alloué
        self._ensure_gpu_cache(total_samples, device)

        # Random sampling: diversité maximale (essentiel pour replay buffer!)
        indices = self._rng.choice(self._size, size=total_samples, replace=False)

        # Copier dans le cache GPU pré-alloué (PAS de nouvelle allocation)
        self._gpu_cache_states[:total_samples].copy_(
            torch.from_numpy(self.states[indices])
        )
        self._gpu_cache_policies[:total_samples].copy_(
            torch.from_numpy(self.policies[indices].astype(np.int64))
        )
        self._gpu_cache_values[:total_samples].copy_(
            torch.from_numpy(self.values[indices])
        )
        self._gpu_cache_weights[:total_samples].copy_(
            torch.from_numpy(self.weights[indices])
        )

        # Créer les vues (pas de copie)
        self._chunk_cache = (
            self._gpu_cache_states[:total_samples],
            self._gpu_cache_policies[:total_samples],
            self._gpu_cache_values[:total_samples],
            self._gpu_cache_weights[:total_samples],
        )
        self._chunk_cache_valid_size = total_samples

    def get_batch_from_cache(self, n: int) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Get next batch from pre-sampled cache (O(1), pas de transfer).

        Args:
            n: Number of positions to get

        Returns:
            Tuple of (states, policy_indices, values, weights) or None if cache empty
        """
        if self._chunk_cache is None:
            return None

        cache_size = self._chunk_cache_valid_size

        if self._chunk_cache_idx >= cache_size:
            return None

        end_idx = min(self._chunk_cache_idx + n, cache_size)
        batch = (
            self._chunk_cache[0][self._chunk_cache_idx : end_idx],
            self._chunk_cache[1][self._chunk_cache_idx : end_idx],
            self._chunk_cache[2][self._chunk_cache_idx : end_idx],
            self._chunk_cache[3][self._chunk_cache_idx : end_idx],
        )
        self._chunk_cache_idx = end_idx
        return batch

    def save(self, path: str) -> None:
        """Save replay buffer to disk."""
        if self._size == 0:
            return
        np.savez_compressed(
            path,
            states=self.states[: self._size],
            policies=self.policies[: self._size],
            values=self.values[: self._size],
            weights=self.weights[: self._size],
        )

    def load(self, path: str) -> bool:
        """Load replay buffer from disk. Returns True if loaded."""
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path)
            n = len(data["states"])
            n = min(n, self.capacity)
            self.states[:n] = data["states"][:n]
            self.policies[:n] = data["policies"][:n]
            self.values[:n] = data["values"][:n]
            self.weights[:n] = data["weights"][:n]
            self._size = n
            return True
        except Exception as e:
            print(f"  WARNING: Failed to load replay buffer: {e}")
            return False

    def __len__(self) -> int:
        return self._size


class EWCRegularizer:
    """
    Elastic Weight Consolidation (EWC) regularizer to prevent catastrophic forgetting.

    After completing an epoch, computes Fisher Information Matrix diagonal to identify
    important weights. Then penalizes changes to these weights in subsequent epochs.
    """

    def __init__(self, lambda_ewc: float = 0.4):
        """
        Initialize EWC regularizer.

        Args:
            lambda_ewc: Regularization strength (higher = more protection)
        """
        self.lambda_ewc = lambda_ewc
        self.fisher: dict = {}
        self.optimal_params: dict = {}
        self._computed = False

    def compute_fisher(
        self,
        network: nn.Module,
        dataloader_fn,
        num_samples: int = 10000,
        device: torch.device = None,
    ) -> None:
        """
        Compute Fisher Information Matrix diagonal after training epoch.

        Args:
            network: Neural network to compute Fisher for
            dataloader_fn: Function that yields batches (states, policies, values, weights)
            num_samples: Number of samples to use for estimation
            device: Device for computation
        """
        network.eval()

        # Initialize Fisher and store optimal params
        for name, param in network.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param)
                self.optimal_params[name] = param.clone().detach()

        samples_processed = 0
        for states, policy_indices, values, weights in dataloader_fn():
            if samples_processed >= num_samples:
                break

            batch_size = states.shape[0]

            # Forward pass
            network.zero_grad()
            pred_policies, _, _ = network(states)

            # Compute log probability of actual moves (supervised signal)
            log_probs = nn.functional.log_softmax(pred_policies, dim=-1)
            selected_log_probs = log_probs.gather(
                1, policy_indices.unsqueeze(1)
            ).squeeze(1)

            # Weight by tactical importance
            weighted_log_prob = (selected_log_probs * weights).sum()

            # Backward to get gradients
            weighted_log_prob.backward()

            # Accumulate squared gradients (Fisher diagonal)
            for name, param in network.named_parameters():
                if param.grad is not None and name in self.fisher:
                    self.fisher[name] += param.grad.pow(2) * batch_size

            samples_processed += batch_size

        # Normalize by number of samples
        for name in self.fisher:
            self.fisher[name] /= max(1, samples_processed)

        self._computed = True
        network.train()

        print(f"  EWC Fisher computed on {samples_processed} samples")

    def penalty(self, network: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty: sum of F_i * (theta_i - theta*_i)^2

        Args:
            network: Current network

        Returns:
            EWC penalty loss (scalar tensor)
        """
        if not self._computed:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0, device=next(network.parameters()).device)
        for name, param in network.named_parameters():
            if name in self.fisher:
                penalty = (
                    penalty
                    + (
                        self.fisher[name] * (param - self.optimal_params[name]).pow(2)
                    ).sum()
                )

        return self.lambda_ewc * penalty

    def save(self, path: str) -> None:
        """Save Fisher and optimal params to file."""
        torch.save(
            {
                "fisher": self.fisher,
                "optimal_params": self.optimal_params,
                "lambda_ewc": self.lambda_ewc,
                "computed": self._computed,
            },
            path,
        )

    def load(self, path: str) -> bool:
        """Load Fisher and optimal params from file. Returns True if successful."""
        if not os.path.exists(path):
            return False
        try:
            data = torch.load(path, map_location="cpu")
            self.fisher = data["fisher"]
            self.optimal_params = data["optimal_params"]
            self.lambda_ewc = data.get("lambda_ewc", self.lambda_ewc)
            self._computed = data.get("computed", True)
            return True
        except Exception as e:
            print(f"  Warning: Failed to load EWC state: {e}")
            return False

    @property
    def is_computed(self) -> bool:
        return self._computed


def adaptive_cross_entropy_with_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing_values: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss with per-sample adaptive label smoothing.

    Standard cross_entropy with label_smoothing applies uniform smoothing.
    This function allows different smoothing per sample based on position complexity.

    Formula per sample:
        loss = (1 - s) * (-log_prob[true_class]) + s * mean(-log_prob)
             = -log_prob[true_class] + s * (log_prob[true_class] - mean(log_prob))

    where s is the per-sample smoothing value.

    Args:
        logits: Model predictions of shape (batch, num_classes)
        targets: One-hot encoded targets of shape (batch, num_classes) OR
                 class indices of shape (batch,)
        smoothing_values: Per-sample smoothing values of shape (batch,)

    Returns:
        Per-sample cross-entropy losses of shape (batch,)
    """
    # Convert one-hot to indices if needed
    if targets.dim() == 2:
        # targets is one-hot encoded
        target_indices = targets.argmax(dim=1)
    else:
        target_indices = targets

    # Compute log softmax for numerical stability
    log_probs = nn.functional.log_softmax(logits, dim=-1)

    # For each sample, compute smoothed loss:
    # loss = (1 - smoothing) * -log_prob[true_class] + smoothing * mean(-log_prob)
    # Equivalently: loss = -log_prob[true_class] + smoothing * (log_prob[true_class] - mean(log_prob))

    # Get log prob of true class
    true_class_log_prob = log_probs.gather(1, target_indices.unsqueeze(1)).squeeze(1)

    # Mean of all log probs (for uniform distribution part)
    mean_log_prob = log_probs.mean(dim=1)

    # Smoothed loss per sample
    # (1 - s) * (-true_log_prob) + s * (-mean_log_prob)
    # = -true_log_prob + s * true_log_prob - s * mean_log_prob
    # = -true_log_prob + s * (true_log_prob - mean_log_prob)
    losses = -true_class_log_prob + smoothing_values * (
        true_class_log_prob - mean_log_prob
    )

    return losses


class SafeGradScaler(GradScaler):
    """
    GradScaler with maximum scale cap to prevent overflow in float16.

    Standard GradScaler can grow unbounded (up to 2^24 or higher), which
    can cause overflow in float16 computations. This version caps the scale
    at a safe maximum value.

    Key safety features:
    - Caps scale at max_scale to prevent float16 overflow
    - Tracks NaN occurrences for monitoring
    - Provides reset_on_nan() to recover from NaN states
    """

    def __init__(
        self,
        device_type: str = "cuda",
        init_scale: float = 2**10,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        max_scale: float = 2**16,
        enabled: bool = True,
    ):
        super().__init__(
            device_type,
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        self._max_scale = max_scale
        self._init_scale = init_scale
        self._nan_count = 0
        self._total_batches = 0

    def update(self, new_scale: Optional[float] = None) -> None:
        """Update scale with cap at max_scale."""
        super().update(new_scale)

        # Cap the scale at max_scale to prevent float16 overflow
        # The scale is stored internally and accessed via get_scale()
        current_scale = self.get_scale()
        if current_scale > self._max_scale:
            # Use the internal tensor if available, otherwise set via state_dict
            if hasattr(self, "_scale") and self._scale is not None:
                self._scale.fill_(self._max_scale)
            else:
                # Fallback: reload state with capped scale
                state = self.state_dict()
                state["scale"] = torch.tensor(self._max_scale)
                self.load_state_dict(state)

        self._total_batches += 1

    def reset_on_nan(self) -> None:
        """Reset scale when NaN is detected to recover from bad state."""
        self._nan_count += 1
        # Reset to a safe lower scale (half of init_scale)
        safe_scale = self._init_scale / 2

        # Access internal scale tensor properly
        if hasattr(self, "_scale") and self._scale is not None:
            self._scale.fill_(safe_scale)
        else:
            # Fallback: reload state with reset scale
            state = self.state_dict()
            state["scale"] = torch.tensor(safe_scale)
            self.load_state_dict(state)

        # Reset internal growth tracker if it exists (PyTorch version dependent)
        if hasattr(self, "_growth_tracker") and self._growth_tracker is not None:
            if isinstance(self._growth_tracker, torch.Tensor):
                self._growth_tracker.zero_()
            else:
                self._growth_tracker = 0

    def get_nan_count(self) -> int:
        """Return the number of NaN occurrences."""
        return self._nan_count

    @property
    def nan_rate(self) -> float:
        """Return the rate of NaN occurrences."""
        if self._total_batches == 0:
            return 0.0
        return self._nan_count / self._total_batches


class StreamingTrainer:
    """
    Trainer that loads and trains on chunks sequentially.

    Advantages:
    - Memory efficient: only one chunk loaded at a time
    - Works with arbitrarily large datasets
    - Shuffles both chunk order and within-chunk order
    """

    def __init__(
        self,
        network: DualHeadNetwork,
        chunks_dir: str,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        validation_split: float = 0.1,
        patience: int = 5,
        output_path: str = "models/pretrained.pt",
        verbose: bool = True,
        resume_from: Optional[str] = None,
        value_loss_weight: float = 5.0,
        entropy_coefficient: float = 0.01,
        label_smoothing: float = 0.05,
        prefetch_workers: int = 2,
        gradient_accumulation_steps: int = 1,
        # Training dynamics (prevent catastrophic forgetting)
        weight_decay: float = 1e-4,
        gradient_clip_norm: float = 1.0,
        lr_decay_factor: float = 0.7,
        lr_decay_patience: int = 5,
        min_learning_rate: float = 1e-5,
        checkpoint_keep_last: int = 10,
        # Anti-forgetting: Tactical Replay Buffer
        tactical_replay_enabled: bool = True,
        tactical_replay_ratio: float = 0.25,
        tactical_replay_threshold: float = 1.5,
        tactical_replay_capacity: int = 100000,
        # Anti-forgetting: Knowledge Distillation
        teacher_enabled: bool = True,
        teacher_path: Optional[str] = None,
        teacher_alpha: float = 0.6,
        teacher_temperature: float = 2.0,
        # Anti-forgetting: EWC
        ewc_enabled: bool = True,
        ewc_lambda: float = 0.4,
        ewc_start_epoch: int = 2,
        ewc_fisher_samples: int = 10000,
        # Testing
        max_chunks: Optional[int] = None,
    ):
        """
        Initialize streaming trainer.

        Args:
            network: Neural network to train.
            chunks_dir: Directory containing chunk files.
            batch_size: Training batch size.
            learning_rate: Initial learning rate.
            validation_split: Fraction of chunks for validation.
            patience: Early stopping patience (epochs without improvement).
            output_path: Path to save trained model.
            verbose: Print progress information.
            resume_from: Resume from checkpoint ('latest' or checkpoint number).
            value_loss_weight: Weight for value loss relative to policy loss (default 5.0).
                              Higher values give more importance to value head training.
            entropy_coefficient: Coefficient for entropy bonus to encourage policy diversity.
            label_smoothing: Label smoothing factor for cross-entropy loss (default 0.05).
            prefetch_workers: Number of background threads for chunk prefetching (default 2).
            gradient_accumulation_steps: Accumulate gradients over N steps before optimizer update.
            tactical_replay_enabled: Enable tactical replay buffer.
            tactical_replay_ratio: Ratio of batch from replay buffer (0.25 = 25%).
            tactical_replay_threshold: Min weight to store position in buffer.
            tactical_replay_capacity: Max positions in replay buffer.
            teacher_enabled: Enable knowledge distillation from teacher network.
            teacher_path: Path to teacher network checkpoint.
            teacher_alpha: Distillation loss weight (0.6 = 60% soft, 40% hard).
            teacher_temperature: Softmax temperature for distillation.
            ewc_enabled: Enable EWC regularization.
            ewc_lambda: EWC regularization strength.
            ewc_start_epoch: First epoch to apply EWC (after computing Fisher).
            ewc_fisher_samples: Number of samples for Fisher estimation.
        """
        self.network = network
        self.value_loss_weight = value_loss_weight
        self.entropy_coefficient = entropy_coefficient
        self.label_smoothing = label_smoothing
        self.prefetch_workers = prefetch_workers
        self.gradient_accumulation_steps = gradient_accumulation_steps
        # Training dynamics parameters
        self.weight_decay = weight_decay
        self.gradient_clip_norm = gradient_clip_norm
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_patience = lr_decay_patience
        self.min_learning_rate = min_learning_rate
        self.checkpoint_keep_last = checkpoint_keep_last

        # Anti-forgetting: Tactical Replay Buffer
        self.tactical_replay_enabled = tactical_replay_enabled
        self.tactical_replay_ratio = tactical_replay_ratio
        self.tactical_replay_capacity = tactical_replay_capacity
        self.tactical_replay_threshold = tactical_replay_threshold
        self.tactical_replay_buffer = None
        # Buffer will be initialized in train() after we know num_chunks

        # Anti-forgetting: Knowledge Distillation
        self.teacher_enabled = teacher_enabled
        self.teacher_alpha = teacher_alpha
        self.teacher_temperature = teacher_temperature
        self.teacher_network = None
        if teacher_enabled and teacher_path and os.path.exists(teacher_path):
            try:
                self.teacher_network = DualHeadNetwork.load(teacher_path)
                self.teacher_network.eval()
                print(f"Teacher network loaded from: {teacher_path}")
            except Exception as e:
                print(f"Warning: Failed to load teacher network: {e}")
                self.teacher_enabled = False

        # Anti-forgetting: EWC
        self.ewc_enabled = ewc_enabled
        self.ewc_start_epoch = ewc_start_epoch
        self.ewc_fisher_samples = ewc_fisher_samples
        self.ewc_regularizer = (
            EWCRegularizer(lambda_ewc=ewc_lambda) if ewc_enabled else None
        )
        self.ewc_path = os.path.join(
            os.path.dirname(output_path) or "models", "ewc_state.pt"
        )
        self.replay_buffer_path = os.path.join(
            os.path.dirname(output_path) or "models", "replay_buffer.npz"
        )
        self.chunks_dir = chunks_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.patience = patience
        self.output_path = output_path
        self.verbose = verbose
        self.max_chunks = max_chunks

        # Get device and AMP settings
        self.device = get_device()
        self.use_amp = supports_mixed_precision()

        # Use SafeGradScaler with capped scale to prevent float16 overflow
        # With gradient accumulation, gradients accumulate in float16 before unscaling
        # So we need a lower max_scale to prevent overflow during accumulation
        # Formula: max_scale = 2^15 / gradient_accumulation_steps
        safe_max_scale = 2**15 // max(1, gradient_accumulation_steps)
        self.scaler = (
            SafeGradScaler(
                "cuda",
                init_scale=2**8,  # Start lower at 256 (safer with grad accum)
                growth_interval=2000,  # Grow every 2000 successful batches
                max_scale=safe_max_scale,  # Dynamic cap based on grad accum
                enabled=True,
            )
            if self.use_amp
            else None
        )

        # Setup optimizer
        self.optimizer = optim.AdamW(
            network.parameters(),
            lr=learning_rate,
            weight_decay=self.weight_decay,
        )

        # Get chunk paths
        self.chunk_paths = list(ChunkManager.iter_chunk_paths(chunks_dir))
        if not self.chunk_paths:
            raise RuntimeError(f"No chunks found in {chunks_dir}")

        # Split chunks into train/val
        n_val = max(1, int(len(self.chunk_paths) * validation_split))
        random.shuffle(self.chunk_paths)
        self.val_chunks = self.chunk_paths[:n_val]
        self.train_chunks = self.chunk_paths[n_val:]

        # Limit chunks for quick testing
        if self.max_chunks is not None:
            self.train_chunks = self.train_chunks[: self.max_chunks]
            self.val_chunks = self.val_chunks[: max(1, self.max_chunks // 10)]

        if verbose:
            print(f"Chunks: {len(self.train_chunks)} train, {len(self.val_chunks)} val")

        # Checkpoint manager
        checkpoint_dir = os.path.dirname(output_path) or "models"
        checkpoint_name = os.path.splitext(os.path.basename(output_path))[0]
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir,
            keep_last_n=self.checkpoint_keep_last,
            verbose=False,
        )
        self.checkpoint_name = checkpoint_name

        # Random number generator
        self._rng = np.random.default_rng(42)

        # Training state
        self.best_val_loss = float("inf")
        self.best_count = 0
        self.epochs_without_improvement = 0
        self.start_epoch = 0
        self._scheduler_state_to_restore = None

        # Intra-epoch resume state
        self._resume_chunk_idx = None
        self._resume_train_chunks_order = None
        self._resume_rng_state = None
        self._resume_epoch_metrics = None

        # Handle resume
        if resume_from is not None:
            self._load_checkpoint(resume_from)

    def _load_checkpoint(self, resume_from: str) -> None:
        """Load checkpoint for resume."""
        # Parse resume_from
        if resume_from.lower() == "latest":
            # Try to load latest checkpoint first (intra-epoch)
            latest_checkpoint = self.checkpoint_manager.load_latest_checkpoint(
                self.checkpoint_name
            )
            if latest_checkpoint is not None:
                state = latest_checkpoint.get("state")
                if state and "chunk_idx" in state:
                    # Intra-epoch checkpoint found
                    print("Found latest checkpoint with intra-epoch state")
                    self.network = DualHeadNetwork.load(
                        latest_checkpoint["network_path"]
                    )
                    print(f"Loaded network from: {latest_checkpoint['network_path']}")

                    # Recreate optimizer
                    self.optimizer = optim.AdamW(
                        self.network.parameters(),
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay,
                    )

                    # Recreate scaler
                    if self.use_amp:
                        safe_max_scale = 2**15 // max(
                            1, self.gradient_accumulation_steps
                        )
                        self.scaler = SafeGradScaler(
                            "cuda",
                            init_scale=2**8,
                            growth_interval=2000,
                            max_scale=safe_max_scale,
                            enabled=True,
                        )

                    # Load optimizer state if available
                    if "optimizer_state" in state:
                        try:
                            self.optimizer.load_state_dict(state["optimizer_state"])
                        except (ValueError, RuntimeError):
                            pass

                    # Restore intra-epoch state
                    self.start_epoch = state.get("epoch", 0)
                    self._resume_chunk_idx = state.get("chunk_idx")
                    self._resume_train_chunks_order = state.get("train_chunks_order")
                    self._resume_rng_state = state.get("rng_state")
                    self._resume_epoch_metrics = {
                        "total_policy_loss": state.get("total_policy_loss", 0.0),
                        "total_value_loss": state.get("total_value_loss", 0.0),
                        "total_batches": state.get("total_batches", 0),
                    }

                    # Restore RNG state if available
                    # Note: We use a fixed seed per epoch, so RNG state is deterministic
                    # Just recreate RNG with epoch-based seed
                    if self._resume_rng_state is not None:
                        # RNG state is saved but we'll use epoch-based seed for consistency
                        pass

                    if "best_val_loss" in state:
                        self.best_val_loss = state["best_val_loss"]
                    if "best_count" in state:
                        self.best_count = state["best_count"]
                    if "epochs_without_improvement" in state:
                        self.epochs_without_improvement = state[
                            "epochs_without_improvement"
                        ]
                    if "scheduler_state" in state:
                        self._scheduler_state_to_restore = state["scheduler_state"]

                    print(
                        f"Resuming from epoch {self.start_epoch + 1}, chunk {self._resume_chunk_idx + 1}"
                    )
                    return

            # Fallback to best numbered checkpoint
            best_count = None
        else:
            try:
                best_count = int(resume_from)
            except ValueError:
                print(f"Invalid resume value: {resume_from}")
                print("Use 'latest' or a checkpoint number")
                return

        # Load best numbered checkpoint
        checkpoint = self.checkpoint_manager.load_best_numbered_checkpoint(
            self.checkpoint_name, best_count
        )

        if checkpoint is None:
            print("No checkpoint found to resume from")
            return

        # Load network weights
        self.network = DualHeadNetwork.load(checkpoint["network_path"])
        print(f"Loaded network from: {checkpoint['network_path']}")

        # Recreate optimizer with new network parameters
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Recreate scaler (don't load old state - causes "No inf checks" error)
        # Use same safe parameters as __init__ - account for gradient accumulation
        if self.use_amp:
            safe_max_scale = 2**15 // max(1, self.gradient_accumulation_steps)
            self.scaler = SafeGradScaler(
                "cuda",
                init_scale=2**8,  # Lower start (safer with grad accum)
                growth_interval=2000,
                max_scale=safe_max_scale,  # Dynamic cap based on grad accum
                enabled=True,
            )

        # Load training state
        state = checkpoint.get("state")
        if state:
            if "optimizer_state" in state:
                try:
                    self.optimizer.load_state_dict(state["optimizer_state"])
                except (ValueError, RuntimeError) as e:
                    # Optimizer state mismatch (e.g., network architecture changed)
                    print(f"  Warning: Could not load optimizer state ({e})")
                    print(
                        "  Using fresh optimizer (this is normal after architecture changes)"
                    )
            if "best_val_loss" in state:
                self.best_val_loss = state["best_val_loss"]
            if "best_count" in state:
                self.best_count = state["best_count"]
            if "epochs_without_improvement" in state:
                self.epochs_without_improvement = state["epochs_without_improvement"]
            # Note: Don't load scaler state - it causes issues with new optimizer

            # Store scheduler state to restore after scheduler is created in train()
            if "scheduler_state" in state:
                self._scheduler_state_to_restore = state["scheduler_state"]

            # Resume from next epoch after the saved one
            if "epoch" in state:
                self.start_epoch = state["epoch"] + 1
            else:
                # Old checkpoint without epoch - estimate from best_count
                self.start_epoch = self.best_count
                print("(Old checkpoint format - estimating epoch from best_count)")

            print(f"Resuming from epoch {self.start_epoch + 1}")
            print(f"Best val loss so far: {self.best_val_loss:.4f}")
            print(f"Best count: {self.best_count}")
            print(f"Epochs without improvement: {self.epochs_without_improvement}")

    def train(self, epochs: int) -> DualHeadNetwork:
        """
        Train for specified number of epochs.

        Args:
            epochs: Number of training epochs.

        Returns:
            Trained network.
        """
        self.network.to(self.device)

        # Initialize Tactical Replay Buffer now that we know num_chunks
        if self.tactical_replay_enabled and self.tactical_replay_buffer is None:
            num_train_chunks = len(self.train_chunks)
            self.tactical_replay_buffer = TacticalReplayBuffer(
                capacity=self.tactical_replay_capacity,
                weight_threshold=self.tactical_replay_threshold,
                num_chunks=num_train_chunks,
            )
            positions_per_chunk = self.tactical_replay_buffer.positions_per_chunk
            print(
                f"Replay Buffer: {self.tactical_replay_capacity:,} capacity, {positions_per_chunk} positions/chunk"
            )

            # Load replay buffer from disk if available
            if self.tactical_replay_buffer.load(self.replay_buffer_path):
                print(
                    f"Loaded replay buffer from: {self.replay_buffer_path} ({len(self.tactical_replay_buffer):,} positions)"
                )

        # Move teacher network to device if enabled
        if self.teacher_enabled and self.teacher_network is not None:
            self.teacher_network.to(self.device)
            self.teacher_network.eval()

        # Load EWC state if available (from previous training run)
        if self.ewc_enabled and self.ewc_regularizer is not None:
            if self.ewc_regularizer.load(self.ewc_path):
                # Move Fisher matrices and optimal params to device
                for name in self.ewc_regularizer.fisher:
                    self.ewc_regularizer.fisher[name] = self.ewc_regularizer.fisher[
                        name
                    ].to(self.device)
                    self.ewc_regularizer.optimal_params[name] = (
                        self.ewc_regularizer.optimal_params[name].to(self.device)
                    )
                print(f"Loaded EWC state from: {self.ewc_path}")

        # Enable TensorFloat32 for faster matmul on Ampere+ GPUs (RTX 30xx, 40xx)
        torch.set_float32_matmul_precision("high")

        # Create scheduler - ReduceLROnPlateau for adaptive learning
        # Reduces LR when val loss doesn't improve for lr_decay_patience epochs
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.lr_decay_factor,
            patience=self.lr_decay_patience,
            min_lr=self.min_learning_rate,
        )

        # Restore scheduler state if resuming
        if (
            hasattr(self, "_scheduler_state_to_restore")
            and self._scheduler_state_to_restore
        ):
            self.scheduler.load_state_dict(self._scheduler_state_to_restore)
            old_lr = self.optimizer.param_groups[0]["lr"]
            # Force the new learning rate from config (override checkpoint)
            if old_lr != self.learning_rate:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate
                print(
                    f"Restored scheduler state, LR overridden: {old_lr:.6f} → {self.learning_rate:.6f}"
                )
            else:
                print(f"Restored scheduler state (LR: {old_lr:.6f})")
            self._scheduler_state_to_restore = None

        print("\nStarting streaming training...")
        print(f"Device: {self.device}, Mixed precision: {self.use_amp}")

        # Print anti-forgetting features status
        print("\nAnti-forgetting features:")
        if self.tactical_replay_enabled:
            print(
                f"  - Tactical Replay Buffer: ENABLED (ratio={self.tactical_replay_ratio}, starts epoch 2)"
            )
        else:
            print("  - Tactical Replay Buffer: DISABLED")
        if self.teacher_enabled and self.teacher_network is not None:
            print(
                f"  - Knowledge Distillation: ENABLED (alpha={self.teacher_alpha}, T={self.teacher_temperature})"
            )
        else:
            print("  - Knowledge Distillation: DISABLED")
        if self.ewc_enabled:
            ewc_status = (
                "ACTIVE"
                if self.ewc_regularizer.is_computed
                else f"starts epoch {self.ewc_start_epoch}"
            )
            print(f"  - EWC Regularization: ENABLED ({ewc_status})")
        else:
            print("  - EWC Regularization: DISABLED")

        if self.start_epoch > 0:
            print(f"\nResuming from epoch {self.start_epoch + 1}/{epochs}")

        for epoch in range(self.start_epoch, epochs):
            epoch_start = time.time()

            # Clear memory between epochs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Training phase
            train_loss, train_policy_loss, train_value_loss = self._train_epoch(
                epoch, epochs
            )

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            val_loss, val_policy_loss, val_value_loss = self._validate(epoch, epochs)

            # Early stop on NaN validation loss
            if np.isnan(val_loss):
                print("\n  CRITICAL: Validation loss is NaN, stopping training!")
                print(f"  Last good checkpoint saved at epoch {self.best_count}")
                print(
                    "  Recommendation: Resume from best checkpoint with lower learning rate"
                )
                break

            # Update scheduler based on validation loss
            self.scheduler.step(val_loss)

            # Print epoch summary with validation
            print(f"\rEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
            print(
                f"  Train - Loss: {train_loss:.4f} (P: {train_policy_loss:.4f}, V: {train_value_loss:.4f})"
            )
            print(
                f"  Val   - Loss: {val_loss:.4f} (P: {val_policy_loss:.4f}, V: {val_value_loss:.4f})"
            )
            print(f"  LR: {current_lr:.6f}")

            # Print anti-forgetting stats
            if self.tactical_replay_enabled and self.tactical_replay_buffer:
                print(
                    f"  Replay buffer: {len(self.tactical_replay_buffer)}/{self.tactical_replay_buffer.capacity} positions"
                )

            # Compute EWC Fisher after first epoch (or ewc_start_epoch - 1)
            if (
                self.ewc_enabled
                and self.ewc_regularizer is not None
                and not self.ewc_regularizer.is_computed
                and epoch + 1 == self.ewc_start_epoch - 1
            ):
                print("  Computing EWC Fisher Information Matrix...")

                def fisher_dataloader():
                    """Generator for Fisher computation using tactical positions."""
                    for chunk_path in self.train_chunks[:10]:  # Use first 10 chunks
                        chunk = ChunkManager.load_chunk(chunk_path)
                        states = (
                            torch.from_numpy(chunk["states"]).float().to(self.device)
                        )
                        policy_indices = (
                            torch.from_numpy(chunk["policy_indices"])
                            .long()
                            .to(self.device)
                        )
                        values = (
                            torch.from_numpy(chunk["values"]).float().to(self.device)
                        )
                        weights = (
                            torch.from_numpy(chunk["weights"]).float().to(self.device)
                        )
                        # Only use tactical positions (high weight)
                        mask = weights >= 1.5
                        if mask.any():
                            for i in range(0, mask.sum().item(), self.batch_size):
                                end = min(i + self.batch_size, mask.sum().item())
                                indices = torch.where(mask)[0][i:end]
                                yield (
                                    states[indices],
                                    policy_indices[indices],
                                    values[indices],
                                    weights[indices],
                                )

                self.ewc_regularizer.compute_fisher(
                    self.network,
                    fisher_dataloader,
                    num_samples=self.ewc_fisher_samples,
                    device=self.device,
                )
                self.ewc_regularizer.save(self.ewc_path)
                print(f"  EWC state saved to: {self.ewc_path}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_count += 1
                self.epochs_without_improvement = 0
                self._save_checkpoint(
                    epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_policy=train_policy_loss,
                    train_value=train_value_loss,
                    val_policy=val_policy_loss,
                    val_value=val_value_loss,
                )
                print(f"  New best model saved! (#{self.best_count})")
            else:
                self.epochs_without_improvement += 1
                print(
                    f"  No improvement ({self.epochs_without_improvement}/{self.patience})"
                )

            # Save replay buffer to disk
            if (
                self.tactical_replay_enabled
                and self.tactical_replay_buffer is not None
                and len(self.tactical_replay_buffer) > 0
            ):
                self.tactical_replay_buffer.save(self.replay_buffer_path)
                print(
                    f"  Replay buffer saved: {len(self.tactical_replay_buffer):,} positions"
                )

            print()

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping: no improvement for {self.patience} epochs")
                break

        print(f"Training complete! Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total improvements: {self.best_count}")

        # Load and return best model
        best_path = f"{os.path.dirname(self.output_path) or 'models'}/{self.checkpoint_name}_best_network.pt"
        if os.path.exists(best_path):
            return DualHeadNetwork.load(best_path)
        return self.network

    def _train_epoch(self, epoch: int, total_epochs: int) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.network.train()

        # Check if we need to resume from a specific chunk
        start_chunk_idx = 0
        if self._resume_chunk_idx is not None and epoch == self.start_epoch:
            # Resume from saved chunk
            start_chunk_idx = self._resume_chunk_idx + 1  # Resume from next chunk
            if self._resume_train_chunks_order is not None:
                # Use saved chunk order
                train_chunks = self._resume_train_chunks_order
            print(
                f"Resuming epoch {epoch + 1} from chunk {start_chunk_idx + 1}/{len(train_chunks)}"
            )
            total_policy_loss = self._resume_epoch_metrics.get("total_policy_loss", 0.0)
            total_value_loss = self._resume_epoch_metrics.get("total_value_loss", 0.0)
            total_batches = self._resume_epoch_metrics.get("total_batches", 0)
            # Clear resume state after using it
            self._resume_chunk_idx = None
            self._resume_train_chunks_order = None
            self._resume_rng_state = None
            self._resume_epoch_metrics = None
        else:
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_batches = 0

        # Shuffle chunk order with fixed seed based on epoch for reproducibility
        train_chunks = self.train_chunks.copy()
        # Use Python's random for chunk shuffle (deterministic per epoch)
        rng_epoch = random.Random(42 + epoch)  # Fixed seed per epoch
        rng_epoch.shuffle(train_chunks)

        # Set numpy RNG with epoch-based seed for consistency
        # Note: We save RNG state in checkpoints but use epoch-based seed for reproducibility
        # The within-chunk shuffle may be slightly different on resume, but that's acceptable
        self._rng = np.random.default_rng(42 + epoch)

        epoch_start_time = time.time()

        # Use prefetching to load next chunk while GPU processes current chunk
        # Also use executor for async checkpoint saving
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.prefetch_workers + 1  # +1 for checkpoint saving
        ) as executor:
            prefetch_future = None
            checkpoint_future = None

            for chunk_idx, chunk_path in enumerate(train_chunks):
                # Skip chunks if resuming
                if chunk_idx < start_chunk_idx:
                    continue
                # Get current chunk (from prefetch or load directly)
                try:
                    if prefetch_future is not None:
                        chunk = prefetch_future.result()
                    else:
                        chunk = ChunkManager.load_chunk(chunk_path)
                    states = chunk["states"]
                    policy_indices = chunk["policy_indices"]
                    values = chunk["values"]
                    weights = chunk["weights"]
                except MemoryError:
                    # Memory fragmentation - force cleanup and retry
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    chunk = ChunkManager.load_chunk(chunk_path)
                    states = chunk["states"]
                    policy_indices = chunk["policy_indices"]
                    values = chunk["values"]
                    weights = chunk["weights"]

                # Start prefetching next chunk in background
                if chunk_idx + 1 < len(train_chunks):
                    prefetch_future = executor.submit(
                        ChunkManager.load_chunk, train_chunks[chunk_idx + 1]
                    )
                else:
                    prefetch_future = None

                # Shuffle within chunk
                indices = self._rng.permutation(len(states))
                states = states[indices]
                policy_indices = policy_indices[indices]
                values = values[indices]
                weights = weights[indices]

                # Add tactical positions to replay buffer
                if (
                    self.tactical_replay_enabled
                    and self.tactical_replay_buffer is not None
                ):
                    self.tactical_replay_buffer.add_batch(
                        states, policy_indices, values, weights
                    )

                # Create mini-batches with gradient accumulation
                n_batches = len(states) // self.batch_size
                accum_steps = self.gradient_accumulation_steps

                # Pre-sample replay buffer for entire chunk (1 call vs 31 calls)
                # Only at epoch 2+ (epoch > 0) - epoch 1 just fills the buffer
                if (
                    self.tactical_replay_enabled
                    and self.tactical_replay_buffer is not None
                    and len(self.tactical_replay_buffer) > 0
                    and epoch > 0
                ):
                    replay_size_per_batch = int(
                        self.batch_size * self.tactical_replay_ratio
                    )
                    total_replay_samples = replay_size_per_batch * n_batches
                    self.tactical_replay_buffer.pre_sample_for_chunk(
                        total_replay_samples, self.device
                    )

                # Zero gradients at the start of each chunk
                self.optimizer.zero_grad()

                for batch_idx in range(n_batches):
                    start = batch_idx * self.batch_size
                    end = start + self.batch_size

                    # Get batch (no .copy() needed - PyTorch copies on .float()/.long())
                    # non_blocking=True for async CPU->GPU transfer
                    batch_states = (
                        torch.from_numpy(states[start:end])
                        .float()
                        .to(self.device, non_blocking=True)
                    )
                    batch_values = (
                        torch.from_numpy(values[start:end])
                        .float()
                        .to(self.device, non_blocking=True)
                    )

                    # Reconstruct one-hot policies using scatter (faster & safer)
                    batch_size_actual = end - start
                    batch_policy_indices = (
                        torch.from_numpy(policy_indices[start:end])
                        .long()
                        .to(self.device, non_blocking=True)
                    )
                    batch_policies = torch.zeros(
                        batch_size_actual, MOVE_ENCODING_SIZE, device=self.device
                    )
                    batch_policies.scatter_(1, batch_policy_indices.unsqueeze(1), 1.0)

                    # Tactical weights for this batch
                    batch_weights = (
                        torch.from_numpy(weights[start:end])
                        .float()
                        .to(self.device, non_blocking=True)
                    )

                    # Mix with replay buffer if enabled (20% from buffer by default)
                    # Only replay starting epoch 2 (epoch > 0) - epoch 1 just fills the buffer
                    # Uses pre-sampled cache for O(1) access (no CPU/GPU transfer per batch)
                    if (
                        self.tactical_replay_enabled
                        and self.tactical_replay_buffer is not None
                        and len(self.tactical_replay_buffer) > 0
                        and epoch
                        > 0  # Don't replay during epoch 1 - nothing to "remember" yet
                    ):
                        replay_size = int(
                            batch_size_actual * self.tactical_replay_ratio
                        )
                        if replay_size > 0:
                            replay_data = (
                                self.tactical_replay_buffer.get_batch_from_cache(
                                    replay_size
                                )
                            )
                            if replay_data is not None:
                                r_states, r_policies_idx, r_values, r_weights = (
                                    replay_data
                                )
                                # Reconstruct one-hot for replay policies
                                r_policies = torch.zeros(
                                    len(r_policies_idx),
                                    MOVE_ENCODING_SIZE,
                                    device=self.device,
                                )
                                r_policies.scatter_(1, r_policies_idx.unsqueeze(1), 1.0)
                                # Concatenate with current batch
                                batch_states = torch.cat(
                                    [batch_states, r_states], dim=0
                                )
                                batch_policies = torch.cat(
                                    [batch_policies, r_policies], dim=0
                                )
                                batch_values = torch.cat(
                                    [batch_values, r_values], dim=0
                                )
                                batch_policy_indices = torch.cat(
                                    [batch_policy_indices, r_policies_idx], dim=0
                                )
                                batch_weights = torch.cat(
                                    [batch_weights, r_weights], dim=0
                                )
                                batch_size_actual = len(batch_states)

                    # Check if this is the last batch in accumulation cycle or chunk
                    is_accumulation_step = (batch_idx + 1) % accum_steps == 0
                    is_last_batch = batch_idx == n_batches - 1

                    # Skip batches with NaN/Inf inputs (corrupted data)
                    if (
                        torch.isnan(batch_states).any()
                        or torch.isinf(batch_states).any()
                        or torch.isnan(batch_values).any()
                        or torch.isinf(batch_values).any()
                    ):
                        continue

                    # Clamp values to [-1, 1] to prevent outliers causing large gradients
                    batch_values = torch.clamp(batch_values, min=-1.0, max=1.0)

                    if self.use_amp:
                        with autocast(device_type="cuda"):
                            pred_policies, pred_values, wdl_logits = self.network(
                                batch_states
                            )

                            # Clamp logits to prevent softmax overflow (|logit| > 88 causes overflow in float16)
                            pred_policies_clamped = torch.clamp(
                                pred_policies, min=-50, max=50
                            )

                            # Compute adaptive label smoothing based on tactical weights
                            # Higher weights (rare/tactical positions) get more smoothing
                            smoothing_values = adaptive_label_smoothing(
                                self.label_smoothing, batch_weights
                            )

                            # Policy loss with adaptive smoothing and tactical weighting
                            policy_loss_unreduced = (
                                adaptive_cross_entropy_with_smoothing(
                                    pred_policies_clamped,
                                    batch_policies,
                                    smoothing_values,
                                )
                            )
                            policy_loss = (policy_loss_unreduced * batch_weights).mean()

                            # Value loss: WDL cross-entropy with adaptive smoothing
                            wdl_targets = values_to_wdl_targets(batch_values)
                            wdl_logits_clamped = torch.clamp(
                                wdl_logits, min=-50, max=50
                            )
                            value_loss_unreduced = (
                                adaptive_cross_entropy_with_smoothing(
                                    wdl_logits_clamped,
                                    wdl_targets,
                                    smoothing_values,
                                )
                            )
                            value_loss = (value_loss_unreduced * batch_weights).mean()

                            # Compute entropy bonus for policy diversity (numerically stable)
                            # Use log_softmax which is more stable than softmax + log
                            log_probs = nn.functional.log_softmax(
                                pred_policies_clamped, dim=-1
                            )
                            probs = torch.exp(log_probs)
                            # Clamp log_probs to prevent -inf * 0 = NaN
                            log_probs_safe = torch.clamp(log_probs, min=-100)
                            entropy = -(probs * log_probs_safe).sum(dim=-1).mean()
                            # Clamp entropy to a safe range
                            entropy = torch.clamp(entropy, min=0.0, max=20.0)

                            # Adaptive entropy coefficient: more entropy for tactical positions
                            # This encourages exploration of alternative moves in complex positions
                            adaptive_entropy_coef = self.entropy_coefficient * (
                                1.0 + 0.3 * (batch_weights.mean() - 1.0)
                            )

                            # Combined base loss with adaptive entropy bonus
                            hard_loss = (
                                policy_loss
                                + self.value_loss_weight * value_loss
                                - adaptive_entropy_coef * entropy
                            )

                            # Knowledge Distillation loss
                            distill_loss = torch.tensor(0.0, device=self.device)
                            if (
                                self.teacher_enabled
                                and self.teacher_network is not None
                            ):
                                with torch.no_grad():
                                    teacher_policies, _, teacher_wdl = (
                                        self.teacher_network(batch_states)
                                    )
                                    teacher_policies_clamped = torch.clamp(
                                        teacher_policies, min=-50, max=50
                                    )
                                # Policy distillation
                                distill_loss = distillation_loss(
                                    pred_policies_clamped,
                                    teacher_policies_clamped,
                                    temperature=self.teacher_temperature,
                                )
                                # Blend hard and soft losses
                                loss = (
                                    1.0 - self.teacher_alpha
                                ) * hard_loss + self.teacher_alpha * distill_loss
                            else:
                                loss = hard_loss

                            # EWC regularization (protects important weights)
                            ewc_penalty = torch.tensor(0.0, device=self.device)
                            if (
                                self.ewc_enabled
                                and self.ewc_regularizer is not None
                                and self.ewc_regularizer.is_computed
                                and epoch + 1 >= self.ewc_start_epoch
                            ):
                                ewc_penalty = self.ewc_regularizer.penalty(self.network)
                                loss = loss + ewc_penalty

                            # Scale loss for gradient accumulation
                            loss = loss / accum_steps

                            # Final safety check - clamp total loss
                            loss = torch.clamp(loss, min=-100, max=100)

                        # Skip backward if loss is NaN/Inf (prevents corrupting gradients)
                        if torch.isnan(loss) or torch.isinf(loss):
                            self.optimizer.zero_grad()
                            self.scaler.reset_on_nan()
                            continue

                        self.scaler.scale(loss).backward()

                        # Only step optimizer at accumulation boundaries or end of chunk
                        if is_accumulation_step or is_last_batch:
                            self.scaler.unscale_(self.optimizer)

                            # Check for NaN/Inf gradients before optimizer step
                            has_bad_grad = False
                            bad_param_name = None
                            for name, param in self.network.named_parameters():
                                if param.grad is not None:
                                    if (
                                        torch.isnan(param.grad).any()
                                        or torch.isinf(param.grad).any()
                                    ):
                                        has_bad_grad = True
                                        bad_param_name = name
                                        break

                            if has_bad_grad:
                                self.optimizer.zero_grad()
                                self.scaler.reset_on_nan()
                                self.scaler.update()
                            else:
                                # Gradient clipping to prevent explosions
                                nn.utils.clip_grad_norm_(
                                    self.network.parameters(),
                                    max_norm=self.gradient_clip_norm,
                                )
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                                self.optimizer.zero_grad()
                    else:
                        pred_policies, pred_values, wdl_logits = self.network(
                            batch_states
                        )

                        # Clamp logits to prevent softmax overflow
                        pred_policies_clamped = torch.clamp(
                            pred_policies, min=-50, max=50
                        )

                        # Compute adaptive label smoothing based on tactical weights
                        # Higher weights (rare/tactical positions) get more smoothing
                        smoothing_values = adaptive_label_smoothing(
                            self.label_smoothing, batch_weights
                        )

                        # Policy loss with adaptive smoothing and tactical weighting
                        policy_loss_unreduced = adaptive_cross_entropy_with_smoothing(
                            pred_policies_clamped,
                            batch_policies,
                            smoothing_values,
                        )
                        policy_loss = (policy_loss_unreduced * batch_weights).mean()

                        # Value loss: WDL cross-entropy with adaptive smoothing
                        wdl_targets = values_to_wdl_targets(batch_values)
                        wdl_logits_clamped = torch.clamp(wdl_logits, min=-50, max=50)
                        value_loss_unreduced = adaptive_cross_entropy_with_smoothing(
                            wdl_logits_clamped,
                            wdl_targets,
                            smoothing_values,
                        )
                        value_loss = (value_loss_unreduced * batch_weights).mean()

                        # Compute entropy bonus for policy diversity (numerically stable)
                        # Use log_softmax which is more stable than softmax + log
                        log_probs = nn.functional.log_softmax(
                            pred_policies_clamped, dim=-1
                        )
                        probs = torch.exp(log_probs)
                        # Clamp log_probs to prevent -inf * 0 = NaN
                        log_probs_safe = torch.clamp(log_probs, min=-100)
                        entropy = -(probs * log_probs_safe).sum(dim=-1).mean()
                        # Clamp entropy to a safe range
                        entropy = torch.clamp(entropy, min=0.0, max=20.0)

                        # Adaptive entropy coefficient: more entropy for tactical positions
                        # This encourages exploration of alternative moves in complex positions
                        adaptive_entropy_coef = self.entropy_coefficient * (
                            1.0 + 0.3 * (batch_weights.mean() - 1.0)
                        )

                        # Combined base loss with adaptive entropy bonus
                        hard_loss = (
                            policy_loss
                            + self.value_loss_weight * value_loss
                            - adaptive_entropy_coef * entropy
                        )

                        # Knowledge Distillation loss
                        distill_loss = torch.tensor(0.0, device=self.device)
                        if self.teacher_enabled and self.teacher_network is not None:
                            with torch.no_grad():
                                teacher_policies, _, teacher_wdl = self.teacher_network(
                                    batch_states
                                )
                                teacher_policies_clamped = torch.clamp(
                                    teacher_policies, min=-50, max=50
                                )
                            # Policy distillation
                            distill_loss = distillation_loss(
                                pred_policies_clamped,
                                teacher_policies_clamped,
                                temperature=self.teacher_temperature,
                            )
                            # Blend hard and soft losses
                            loss = (
                                1.0 - self.teacher_alpha
                            ) * hard_loss + self.teacher_alpha * distill_loss
                        else:
                            loss = hard_loss

                        # EWC regularization (protects important weights)
                        ewc_penalty = torch.tensor(0.0, device=self.device)
                        if (
                            self.ewc_enabled
                            and self.ewc_regularizer is not None
                            and self.ewc_regularizer.is_computed
                            and epoch + 1 >= self.ewc_start_epoch
                        ):
                            ewc_penalty = self.ewc_regularizer.penalty(self.network)
                            loss = loss + ewc_penalty

                        # Scale loss for gradient accumulation
                        loss = loss / accum_steps

                        # Final safety check - clamp total loss
                        loss = torch.clamp(loss, min=-100, max=100)

                        # Skip backward if loss is NaN/Inf (prevents corrupting gradients)
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(
                                "\n  WARNING: Loss is NaN/Inf (no AMP), skipping batch"
                            )
                            self.optimizer.zero_grad()
                            continue

                        loss.backward()

                        # Only step optimizer at accumulation boundaries or end of chunk
                        if is_accumulation_step or is_last_batch:
                            # Check for NaN/Inf gradients before optimizer step
                            has_bad_grad = False
                            bad_param_name = None
                            for name, param in self.network.named_parameters():
                                if param.grad is not None:
                                    if (
                                        torch.isnan(param.grad).any()
                                        or torch.isinf(param.grad).any()
                                    ):
                                        has_bad_grad = True
                                        bad_param_name = name
                                        break

                            if has_bad_grad:
                                print(
                                    f"\n  WARNING: NaN/Inf gradient in {bad_param_name}, skipping batch"
                                )
                                self.optimizer.zero_grad()
                            else:
                                # Gradient clipping to prevent explosions
                                nn.utils.clip_grad_norm_(
                                    self.network.parameters(),
                                    max_norm=self.gradient_clip_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()

                    # Skip NaN losses (use unscaled loss for logging)
                    p_loss = policy_loss.item()
                    v_loss = value_loss.item()
                    if not (np.isnan(p_loss) or np.isnan(v_loss)):
                        total_policy_loss += p_loss
                        total_value_loss += v_loss
                        total_batches += 1

                    # Monitor value head health (detect collapse early)
                    if total_batches % 500 == 0:
                        value_std = pred_values.std().item()
                        if value_std < 0.1:
                            print(
                                f"\n  WARNING: Value head may be collapsing (std={value_std:.4f})"
                            )

                # Progress update with average time per chunk
                # Use chunks processed since resume, not total chunk index
                elapsed = time.time() - epoch_start_time
                chunks_processed = chunk_idx - start_chunk_idx + 1
                avg_time_per_chunk = elapsed / chunks_processed
                eta_seconds = avg_time_per_chunk * (len(train_chunks) - chunk_idx - 1)
                eta_h = int(eta_seconds // 3600)
                eta_min = int((eta_seconds % 3600) // 60)
                eta_sec = int(eta_seconds % 60)
                eta_str = (
                    f"{eta_h}h{eta_min:02d}m{eta_sec:02d}s"
                    if eta_h > 0
                    else f"{eta_min}m{eta_sec:02d}s"
                )
                pct = (chunk_idx + 1) * 100 // len(train_chunks)

                # Build progress line with optional buffer info
                progress_line = (
                    f"\rEpoch {epoch+1}/{total_epochs} train {pct}% (chunk {chunk_idx+1}/{len(train_chunks)}) "
                    f"[{avg_time_per_chunk:.1f}s/chunk, ETA {eta_str}]"
                )

                # Show buffer progress only when not full
                if (
                    self.tactical_replay_enabled
                    and self.tactical_replay_buffer is not None
                    and len(self.tactical_replay_buffer)
                    < self.tactical_replay_buffer.capacity
                ):
                    buf_size = len(self.tactical_replay_buffer)
                    buf_cap = self.tactical_replay_buffer.capacity
                    buf_pct = buf_size * 100 // buf_cap
                    progress_line += f" Buffer: {buf_size:,}/{buf_cap:,} ({buf_pct}%)"

                print(progress_line + "   ", end="", flush=True)

                # Save checkpoint every 10 chunks (and at least 60 seconds apart)
                should_save_checkpoint = (chunk_idx + 1) % 10 == 0 or chunk_idx == len(
                    train_chunks
                ) - 1

                if should_save_checkpoint:
                    # Wait for previous checkpoint save to complete if still running
                    if checkpoint_future is not None:
                        try:
                            checkpoint_future.result(timeout=5)  # Wait max 5 seconds
                        except concurrent.futures.TimeoutError:
                            pass  # Continue even if previous save is slow

                    # Save checkpoint asynchronously
                    checkpoint_future = executor.submit(
                        self._save_latest_checkpoint_async,
                        epoch,
                        chunk_idx,
                        train_chunks,
                        total_policy_loss,
                        total_value_loss,
                        total_batches,
                    )
                    last_checkpoint_time = time.time()

                # Cleanup chunk data to prevent memory fragmentation
                del chunk, states, policy_indices, values, indices
                if chunk_idx % 5 == 0:  # Frequent lightweight cleanup
                    gc.collect(0)  # Generation 0 only (10x faster than full gc)

            # Wait for final checkpoint save to complete
            if checkpoint_future is not None:
                try:
                    checkpoint_future.result(timeout=10)
                except concurrent.futures.TimeoutError:
                    pass

        avg_policy = total_policy_loss / max(1, total_batches)
        avg_value = total_value_loss / max(1, total_batches)
        return avg_policy + avg_value, avg_policy, avg_value

    def _validate(self, epoch: int, total_epochs: int) -> Tuple[float, float, float]:
        """Run validation on validation chunks with prefetching."""
        self.network.eval()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_batches = 0
        val_start_time = time.time()

        with torch.inference_mode():
            # Use prefetching to load next chunk while GPU processes current chunk
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.prefetch_workers
            ) as executor:
                prefetch_future = None

                for chunk_idx, chunk_path in enumerate(self.val_chunks):
                    # Get current chunk (from prefetch or load directly)
                    try:
                        if prefetch_future is not None:
                            chunk = prefetch_future.result()
                        else:
                            chunk = ChunkManager.load_chunk(chunk_path)
                        states = chunk["states"]
                        policy_indices = chunk["policy_indices"]
                        values = chunk["values"]
                    except MemoryError:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        chunk = ChunkManager.load_chunk(chunk_path)
                        states = chunk["states"]
                        policy_indices = chunk["policy_indices"]
                        values = chunk["values"]

                    # Start prefetching next chunk in background
                    if chunk_idx + 1 < len(self.val_chunks):
                        prefetch_future = executor.submit(
                            ChunkManager.load_chunk, self.val_chunks[chunk_idx + 1]
                        )
                    else:
                        prefetch_future = None

                    n_batches = len(states) // self.batch_size
                    for batch_idx in range(n_batches):
                        start = batch_idx * self.batch_size
                        end = start + self.batch_size

                        batch_states = (
                            torch.from_numpy(states[start:end])
                            .float()
                            .to(self.device, non_blocking=True)
                        )
                        batch_values = (
                            torch.from_numpy(values[start:end])
                            .float()
                            .to(self.device, non_blocking=True)
                        )

                        batch_size_actual = end - start
                        batch_policy_indices = (
                            torch.from_numpy(policy_indices[start:end])
                            .long()
                            .to(self.device, non_blocking=True)
                        )
                        batch_policies = torch.zeros(
                            batch_size_actual, MOVE_ENCODING_SIZE, device=self.device
                        )
                        batch_policies.scatter_(
                            1, batch_policy_indices.unsqueeze(1), 1.0
                        )

                        pred_policies, _, wdl_logits = self.network(batch_states)
                        policy_loss = nn.functional.cross_entropy(
                            pred_policies,
                            batch_policies,
                            label_smoothing=self.label_smoothing,
                        )

                        # Value loss: WDL cross-entropy
                        wdl_targets = values_to_wdl_targets(batch_values)
                        value_loss = nn.functional.cross_entropy(
                            wdl_logits,
                            wdl_targets,
                            label_smoothing=self.label_smoothing,
                        )

                        # Skip NaN losses in validation
                        p_loss = policy_loss.item()
                        v_loss = value_loss.item()
                        if not (np.isnan(p_loss) or np.isnan(v_loss)):
                            total_policy_loss += p_loss
                            total_value_loss += v_loss
                            total_batches += 1
                        else:
                            print("\n  WARNING: NaN in validation batch, skipping")

                    # Progress update with average time per chunk
                    elapsed = time.time() - val_start_time
                    avg_time_per_chunk = elapsed / (chunk_idx + 1)
                    eta_seconds = avg_time_per_chunk * (
                        len(self.val_chunks) - chunk_idx - 1
                    )
                    eta_h = int(eta_seconds // 3600)
                    eta_min = int((eta_seconds % 3600) // 60)
                    eta_sec = int(eta_seconds % 60)
                    eta_str = (
                        f"{eta_h}h{eta_min:02d}m{eta_sec:02d}s"
                        if eta_h > 0
                        else f"{eta_min}m{eta_sec:02d}s"
                    )
                    pct = (chunk_idx + 1) * 100 // len(self.val_chunks)
                    print(
                        f"\rEpoch {epoch+1}/{total_epochs} val {pct}% (chunk {chunk_idx+1}/{len(self.val_chunks)}) "
                        f"[{avg_time_per_chunk:.1f}s/chunk, ETA {eta_str}]   ",
                        end="",
                        flush=True,
                    )

                    # Cleanup chunk data
                    del chunk, states, policy_indices, values

        avg_policy = total_policy_loss / max(1, total_batches)
        avg_value = total_value_loss / max(1, total_batches)
        return avg_policy + avg_value, avg_policy, avg_value

    def _save_latest_checkpoint_async(
        self,
        epoch: int,
        chunk_idx: int,
        train_chunks_order: List[str],
        total_policy_loss: float,
        total_value_loss: float,
        total_batches: int,
    ) -> None:
        """Save latest checkpoint asynchronously with intra-epoch state."""
        try:
            state = {
                "epoch": epoch,
                "chunk_idx": chunk_idx,
                "train_chunks_order": train_chunks_order,
                "total_policy_loss": total_policy_loss,
                "total_value_loss": total_value_loss,
                "total_batches": total_batches,
                "rng_state": self._rng.bit_generator.state,  # Save for reference, but epoch seed is primary
                "best_val_loss": self.best_val_loss,
                "best_count": self.best_count,
                "epochs_without_improvement": self.epochs_without_improvement,
                "optimizer_state": self.optimizer.state_dict(),
            }
            if hasattr(self, "scheduler") and self.scheduler is not None:
                state["scheduler_state"] = self.scheduler.state_dict()
            if self.scaler is not None:
                state["scaler_state"] = self.scaler.state_dict()

            self.checkpoint_manager.save_latest_checkpoint(
                self.checkpoint_name, self.network, state
            )
        except Exception as e:
            # Don't crash training if checkpoint save fails
            print(f"\n  WARNING: Failed to save latest checkpoint: {e}")

    def _save_checkpoint(
        self,
        epoch: int,
        train_loss: float = 0.0,
        val_loss: float = 0.0,
        train_policy: float = 0.0,
        train_value: float = 0.0,
        val_policy: float = 0.0,
        val_value: float = 0.0,
    ) -> None:
        """Save best model checkpoint."""
        state = {
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "best_count": self.best_count,
            "epochs_without_improvement": self.epochs_without_improvement,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            # Loss metrics for this checkpoint
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_policy": train_policy,
            "train_value": train_value,
            "val_policy": val_policy,
            "val_value": val_value,
        }
        if self.scaler is not None:
            state["scaler_state"] = self.scaler.state_dict()

        self.checkpoint_manager.save_best_checkpoint(
            self.checkpoint_name, self.best_count, self.network, state
        )
