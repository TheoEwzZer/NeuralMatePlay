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
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

from .chunk_manager import ChunkManager

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphazero import DualHeadNetwork, get_device, supports_mixed_precision
from alphazero.checkpoint_manager import CheckpointManager
from alphazero.move_encoding import MOVE_ENCODING_SIZE


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

    def __init__(self, device_type: str = "cuda", init_scale: float = 2**10,
                 growth_factor: float = 2.0, backoff_factor: float = 0.5,
                 growth_interval: int = 2000, max_scale: float = 2**16,
                 enabled: bool = True):
        super().__init__(
            device_type,
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled
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
            if hasattr(self, '_scale') and self._scale is not None:
                self._scale.fill_(self._max_scale)
            else:
                # Fallback: reload state with capped scale
                state = self.state_dict()
                state['scale'] = torch.tensor(self._max_scale)
                self.load_state_dict(state)

        self._total_batches += 1

    def reset_on_nan(self) -> None:
        """Reset scale when NaN is detected to recover from bad state."""
        self._nan_count += 1
        # Reset to a safe lower scale (half of init_scale)
        safe_scale = self._init_scale / 2

        # Access internal scale tensor properly
        if hasattr(self, '_scale') and self._scale is not None:
            self._scale.fill_(safe_scale)
        else:
            # Fallback: reload state with reset scale
            state = self.state_dict()
            state['scale'] = torch.tensor(safe_scale)
            self.load_state_dict(state)

        # Reset internal growth tracker if it exists (PyTorch version dependent)
        if hasattr(self, '_growth_tracker') and self._growth_tracker is not None:
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
        prefetch_workers: int = 2,
        gradient_accumulation_steps: int = 1,
        phase_loss_weight: float = 0.1,
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
            prefetch_workers: Number of background threads for chunk prefetching (default 2).
            gradient_accumulation_steps: Accumulate gradients over N steps before optimizer update.
            phase_loss_weight: Weight for phase prediction loss (default 0.1).
        """
        self.network = network
        self.value_loss_weight = value_loss_weight
        self.entropy_coefficient = entropy_coefficient
        self.phase_loss_weight = phase_loss_weight
        self.prefetch_workers = prefetch_workers
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.chunks_dir = chunks_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.patience = patience
        self.output_path = output_path
        self.verbose = verbose

        # Get device and AMP settings
        self.device = get_device()
        self.use_amp = supports_mixed_precision()

        # Use SafeGradScaler with capped scale to prevent float16 overflow
        # With gradient accumulation, gradients accumulate in float16 before unscaling
        # So we need a lower max_scale to prevent overflow during accumulation
        # Formula: max_scale = 2^15 / gradient_accumulation_steps
        safe_max_scale = 2**15 // max(1, gradient_accumulation_steps)
        self.scaler = SafeGradScaler(
            "cuda",
            init_scale=2**8,  # Start lower at 256 (safer with grad accum)
            growth_interval=2000,  # Grow every 2000 successful batches
            max_scale=safe_max_scale,  # Dynamic cap based on grad accum
            enabled=True
        ) if self.use_amp else None

        # Setup optimizer
        self.optimizer = optim.AdamW(
            network.parameters(),
            lr=learning_rate,
            weight_decay=5e-4,  # Increased for regularization
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

        if verbose:
            print(f"Chunks: {len(self.train_chunks)} train, {len(self.val_chunks)} val")

        # Checkpoint manager
        checkpoint_dir = os.path.dirname(output_path) or "models"
        checkpoint_name = os.path.splitext(os.path.basename(output_path))[0]
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, verbose=False)
        self.checkpoint_name = checkpoint_name

        # Random number generator
        self._rng = np.random.default_rng(42)

        # Training state
        self.best_val_loss = float("inf")
        self.best_count = 0
        self.epochs_without_improvement = 0
        self.start_epoch = 0
        self._scheduler_state_to_restore = None

        # Handle resume
        if resume_from is not None:
            self._load_checkpoint(resume_from)

    def _load_checkpoint(self, resume_from: str) -> None:
        """Load checkpoint for resume."""
        # Parse resume_from
        if resume_from.lower() == "latest":
            best_count = None
        else:
            try:
                best_count = int(resume_from)
            except ValueError:
                print(f"Invalid resume value: {resume_from}")
                print("Use 'latest' or a checkpoint number")
                return

        # Load checkpoint
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
            weight_decay=5e-4,  # Increased for regularization
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
                enabled=True
            )

        # Load training state
        state = checkpoint.get("state")
        if state:
            if "optimizer_state" in state:
                self.optimizer.load_state_dict(state["optimizer_state"])
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

        # Enable TensorFloat32 for faster matmul on Ampere+ GPUs (RTX 30xx, 40xx)
        torch.set_float32_matmul_precision("high")

        # Create scheduler - ReduceLROnPlateau for adaptive learning
        # Reduces LR by factor of 0.5 when val loss doesn't improve for 3 epochs
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )

        # Restore scheduler state if resuming
        if hasattr(self, '_scheduler_state_to_restore') and self._scheduler_state_to_restore:
            self.scheduler.load_state_dict(self._scheduler_state_to_restore)
            old_lr = self.optimizer.param_groups[0]['lr']
            # Force the new learning rate from config (override checkpoint)
            if old_lr != self.learning_rate:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                print(f"Restored scheduler state, LR overridden: {old_lr:.6f} → {self.learning_rate:.6f}")
            else:
                print(f"Restored scheduler state (LR: {old_lr:.6f})")
            self._scheduler_state_to_restore = None

        print("\nStarting streaming training...")
        print(f"Device: {self.device}, Mixed precision: {self.use_amp}")
        if self.start_epoch > 0:
            print(f"Resuming from epoch {self.start_epoch + 1}/{epochs}")

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
                print("  Recommendation: Resume from best checkpoint with lower learning rate")
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

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_count += 1
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch)
                print(f"  New best model saved! (#{self.best_count})")
            else:
                self.epochs_without_improvement += 1
                print(
                    f"  No improvement ({self.epochs_without_improvement}/{self.patience})"
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

        # Shuffle chunk order
        train_chunks = self.train_chunks.copy()
        random.shuffle(train_chunks)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_batches = 0
        epoch_start_time = time.time()

        # Use prefetching to load next chunk while GPU processes current chunk
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.prefetch_workers) as executor:
            prefetch_future = None

            for chunk_idx, chunk_path in enumerate(train_chunks):
                # Get current chunk (from prefetch or load directly)
                try:
                    if prefetch_future is not None:
                        chunk = prefetch_future.result()
                    else:
                        chunk = ChunkManager.load_chunk(chunk_path)
                    states = chunk["states"]
                    policy_indices = chunk["policy_indices"]
                    values = chunk["values"]
                    phases = chunk["phases"]
                except MemoryError:
                    # Memory fragmentation - force cleanup and retry
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    chunk = ChunkManager.load_chunk(chunk_path)
                    states = chunk["states"]
                    policy_indices = chunk["policy_indices"]
                    values = chunk["values"]
                    phases = chunk["phases"]

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
                phases = phases[indices]

                # Create mini-batches with gradient accumulation
                n_batches = len(states) // self.batch_size
                accum_steps = self.gradient_accumulation_steps

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

                    # Phases for auxiliary task (0=opening, 1=middlegame, 2=endgame)
                    batch_phases = (
                        torch.from_numpy(phases[start:end])
                        .long()
                        .to(self.device, non_blocking=True)
                    )

                    # Check if this is the last batch in accumulation cycle or chunk
                    is_accumulation_step = (batch_idx + 1) % accum_steps == 0
                    is_last_batch = batch_idx == n_batches - 1

                    # Skip batches with NaN/Inf inputs (corrupted data)
                    if (torch.isnan(batch_states).any() or torch.isinf(batch_states).any() or
                        torch.isnan(batch_values).any() or torch.isinf(batch_values).any()):
                        continue

                    # Clamp values to [-1, 1] to prevent outliers causing large gradients
                    batch_values = torch.clamp(batch_values, min=-1.0, max=1.0)

                    if self.use_amp:
                        with autocast(device_type="cuda"):
                            pred_policies, pred_values, pred_phases = self.network(batch_states)

                            # Clamp logits to prevent softmax overflow (|logit| > 88 causes overflow in float16)
                            pred_policies_clamped = torch.clamp(pred_policies, min=-50, max=50)

                            policy_loss = nn.functional.cross_entropy(
                                pred_policies_clamped, batch_policies, label_smoothing=0.2
                            )
                            value_loss = nn.functional.mse_loss(
                                pred_values, batch_values
                            )

                            # Phase loss (auxiliary task)
                            if pred_phases is not None:
                                # Clamp phase logits to prevent overflow in float16
                                pred_phases_clamped = torch.clamp(pred_phases, min=-50, max=50)
                                phase_loss = nn.functional.cross_entropy(
                                    pred_phases_clamped, batch_phases, label_smoothing=0.1
                                )
                            else:
                                phase_loss = torch.tensor(0.0, device=self.device)

                            # Compute entropy bonus for policy diversity (numerically stable)
                            # Use log_softmax which is more stable than softmax + log
                            log_probs = nn.functional.log_softmax(pred_policies_clamped, dim=-1)
                            probs = torch.exp(log_probs)
                            # Clamp log_probs to prevent -inf * 0 = NaN
                            log_probs_safe = torch.clamp(log_probs, min=-100)
                            entropy = -(probs * log_probs_safe).sum(dim=-1).mean()
                            # Clamp entropy to a safe range
                            entropy = torch.clamp(entropy, min=0.0, max=20.0)

                            # Combined loss with entropy bonus (subtract to maximize entropy)
                            # Scale loss for gradient accumulation
                            loss = (policy_loss + self.value_loss_weight * value_loss + self.phase_loss_weight * phase_loss - self.entropy_coefficient * entropy) / accum_steps

                            # Final safety check - clamp total loss
                            loss = torch.clamp(loss, min=-100, max=100)

                        # Skip backward if loss is NaN/Inf (prevents corrupting gradients)
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"\n  WARNING: Loss is NaN/Inf (scale={self.scaler.get_scale():.0f}), resetting scaler")
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
                                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                        has_bad_grad = True
                                        bad_param_name = name
                                        break

                            if has_bad_grad:
                                print(f"\n  WARNING: NaN/Inf gradient in {bad_param_name} (scale={self.scaler.get_scale():.0f}), resetting scaler")
                                self.optimizer.zero_grad()
                                self.scaler.reset_on_nan()
                                self.scaler.update()
                            else:
                                # Use tighter gradient clipping to prevent explosions
                                nn.utils.clip_grad_norm_(
                                    self.network.parameters(), max_norm=0.5
                                )
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                                self.optimizer.zero_grad()
                    else:
                        pred_policies, pred_values, pred_phases = self.network(batch_states)

                        # Clamp logits to prevent softmax overflow
                        pred_policies_clamped = torch.clamp(pred_policies, min=-50, max=50)

                        policy_loss = nn.functional.cross_entropy(
                            pred_policies_clamped, batch_policies, label_smoothing=0.2
                        )
                        value_loss = nn.functional.mse_loss(pred_values, batch_values)

                        # Phase loss (auxiliary task)
                        if pred_phases is not None:
                            # Clamp phase logits to prevent overflow
                            pred_phases_clamped = torch.clamp(pred_phases, min=-50, max=50)
                            phase_loss = nn.functional.cross_entropy(
                                pred_phases_clamped, batch_phases, label_smoothing=0.1
                            )
                        else:
                            phase_loss = torch.tensor(0.0, device=self.device)

                        # Compute entropy bonus for policy diversity (numerically stable)
                        # Use log_softmax which is more stable than softmax + log
                        log_probs = nn.functional.log_softmax(pred_policies_clamped, dim=-1)
                        probs = torch.exp(log_probs)
                        # Clamp log_probs to prevent -inf * 0 = NaN
                        log_probs_safe = torch.clamp(log_probs, min=-100)
                        entropy = -(probs * log_probs_safe).sum(dim=-1).mean()
                        # Clamp entropy to a safe range
                        entropy = torch.clamp(entropy, min=0.0, max=20.0)

                        # Combined loss with entropy bonus (subtract to maximize entropy)
                        # Scale loss for gradient accumulation
                        loss = (policy_loss + self.value_loss_weight * value_loss + self.phase_loss_weight * phase_loss - self.entropy_coefficient * entropy) / accum_steps

                        # Final safety check - clamp total loss
                        loss = torch.clamp(loss, min=-100, max=100)

                        # Skip backward if loss is NaN/Inf (prevents corrupting gradients)
                        if torch.isnan(loss) or torch.isinf(loss):
                            print("\n  WARNING: Loss is NaN/Inf (no AMP), skipping batch")
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
                                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                        has_bad_grad = True
                                        bad_param_name = name
                                        break

                            if has_bad_grad:
                                print(f"\n  WARNING: NaN/Inf gradient in {bad_param_name}, skipping batch")
                                self.optimizer.zero_grad()
                            else:
                                # Use tighter gradient clipping to prevent explosions
                                nn.utils.clip_grad_norm_(
                                    self.network.parameters(), max_norm=0.5
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
                            print(f"\n  WARNING: Value head may be collapsing (std={value_std:.4f})")

                # Progress update with average time per chunk
                elapsed = time.time() - epoch_start_time
                avg_time_per_chunk = elapsed / (chunk_idx + 1)
                eta_seconds = avg_time_per_chunk * (len(train_chunks) - chunk_idx - 1)
                eta_h = int(eta_seconds // 3600)
                eta_min = int((eta_seconds % 3600) // 60)
                eta_sec = int(eta_seconds % 60)
                eta_str = f"{eta_h}h{eta_min:02d}m{eta_sec:02d}s" if eta_h > 0 else f"{eta_min}m{eta_sec:02d}s"
                pct = (chunk_idx + 1) * 100 // len(train_chunks)
                print(
                    f"\rEpoch {epoch+1}/{total_epochs} train {pct}% (chunk {chunk_idx+1}/{len(train_chunks)}) "
                    f"[{avg_time_per_chunk:.1f}s/chunk, ETA {eta_str}]   ",
                    end="",
                    flush=True,
                )

                # Cleanup chunk data to prevent memory fragmentation
                del chunk, states, policy_indices, values, phases, indices
                if chunk_idx % 5 == 0:  # Frequent lightweight cleanup
                    gc.collect(0)  # Generation 0 only (10x faster than full gc)

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
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.prefetch_workers) as executor:
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
                        # phases not needed for validation (auxiliary task)
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
                        batch_policies.scatter_(1, batch_policy_indices.unsqueeze(1), 1.0)

                        pred_policies, pred_values, _ = self.network(batch_states)
                        policy_loss = nn.functional.cross_entropy(
                            pred_policies, batch_policies, label_smoothing=0.2
                        )
                        value_loss = nn.functional.mse_loss(pred_values, batch_values)

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
                    eta_seconds = avg_time_per_chunk * (len(self.val_chunks) - chunk_idx - 1)
                    eta_h = int(eta_seconds // 3600)
                    eta_min = int((eta_seconds % 3600) // 60)
                    eta_sec = int(eta_seconds % 60)
                    eta_str = f"{eta_h}h{eta_min:02d}m{eta_sec:02d}s" if eta_h > 0 else f"{eta_min}m{eta_sec:02d}s"
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

    def _save_checkpoint(self, epoch: int) -> None:
        """Save best model checkpoint."""
        state = {
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "best_count": self.best_count,
            "epochs_without_improvement": self.epochs_without_improvement,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        if self.scaler is not None:
            state["scaler_state"] = self.scaler.state_dict()

        self.checkpoint_manager.save_best_checkpoint(
            self.checkpoint_name, self.best_count, self.network, state
        )
