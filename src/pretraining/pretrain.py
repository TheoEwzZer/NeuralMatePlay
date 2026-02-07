"""
Pretraining script for supervised learning on master games.

Usage:
    ./neural_mate_pretrain --config config.json
    ./neural_mate_pretrain --pgn data/lichess_elite_2020-08.pgn --epochs 5
    ./neural_mate_pretrain --pgn data/jan.pgn data/feb.pgn data/mar.pgn
    ./neural_mate_pretrain --pgn "data/lichess_*.pgn"
    ./neural_mate_pretrain --resume-pretrained latest
    ./neural_mate_pretrain --resume-pretrained 3 --epochs 10
"""

import glob
import json
import os
import argparse
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphazero import DualHeadNetwork, get_device, supports_mixed_precision
from alphazero.checkpoint_manager import CheckpointManager
from config import Config, NetworkConfig, PretrainingConfig  # type: ignore[import-not-found]
from .dataset import ChessPositionDataset
from .chunk_manager import ChunkManager
from .tactical_weighting import calculate_tactical_weight
from .streaming_trainer import StreamingTrainer


def expand_pgn_paths(paths: list[str]) -> list[str]:
    """
    Expand glob patterns and deduplicate PGN paths.

    Args:
        paths: List of file paths or glob patterns.

    Returns:
        List of expanded, deduplicated file paths.
    """
    expanded: list[str] = []
    for path in paths:
        if "*" in path or "?" in path:
            # Glob pattern - expand it
            matches: list[str] = sorted(glob.glob(path))
            if not matches:
                print(f"Warning: No files match pattern '{path}'")
            expanded.extend(matches)
        else:
            expanded.append(path)
    # Deduplicate while preserving order
    return list(dict.fromkeys(expanded))


def create_dataloaders(
    dataset: ChessPositionDataset,
    batch_size: int = 256,
    validation_split: float = 0.1,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    val_size: int = int(len(dataset) * validation_split)
    train_size: int = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def pretrain(
    cfg: PretrainingConfig,
    network_cfg: NetworkConfig | None = None,
    streaming: bool = False,
    resume_from: str | None = None,
) -> DualHeadNetwork:
    """
    Pretrain network on master games.

    Args:
        cfg: Pretraining configuration.
        network_cfg: Optional network configuration.
        streaming: Use streaming mode (loads chunks incrementally).
        resume_from: Resume from checkpoint ('latest' or checkpoint number).

    Returns:
        Trained DualHeadNetwork.
    """
    device: torch.device = get_device()
    use_amp: bool = supports_mixed_precision()

    print(f"Device: {device}")
    print(f"Mixed precision: {use_amp}")

    # Get all PGN paths
    pgn_paths: list[str] = cfg.get_pgn_paths()

    # Print config
    print("\n[Configuration]")
    print(f"  PGN files: {len(pgn_paths)}")
    for p in pgn_paths:
        print(f"    - {p}")
    print(f"  Output: {cfg.output_path}")
    print(f"  Chunks: {cfg.chunks_dir}")
    print(f"  Chunk size: {cfg.chunk_size}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Max games: {cfg.max_games or 'unlimited'}")
    print(f"  Streaming: {streaming}")

    # Check if we need to create or extend chunks
    metadata: dict[str, Any] | None = ChunkManager.load_metadata(cfg.chunks_dir)
    processed_files: set[str] = (
        set(metadata.get("processed_files", [])) if metadata else set()
    )

    # Check for partially processed file (resume support)
    current_file: str | None = metadata.get("current_file") if metadata else None
    current_file_games: int = metadata.get("current_file_games", 0) if metadata else 0

    files_to_process: list[str] = [p for p in pgn_paths if p not in processed_files]

    if files_to_process:
        from .pgn_processor import PGNProcessor
        from alphazero.spatial_encoding import PositionHistory
        from alphazero.move_encoding import encode_move_from_perspective

        resume_mode: bool = metadata is not None
        if resume_mode:
            print(f"\nExtending chunks with {len(files_to_process)} new file(s)...")
            print(f"  Already processed: {len(processed_files)} file(s)")
        else:
            print(
                f"\nNo chunks found, creating from {len(files_to_process)} PGN file(s)..."
            )

        chunk_manager: ChunkManager = ChunkManager(
            cfg.chunks_dir, chunk_size=cfg.chunk_size, verbose=True, resume=resume_mode
        )
        total_position_count: int = 0
        total_games_count: int = chunk_manager._games_processed

        # Track cumulative totals for display (including pre-resume data)
        initial_positions: int = chunk_manager._total_examples
        # Include games from both completed files AND partial file
        initial_games: int = chunk_manager._games_processed + current_file_games

        for file_idx, pgn_file in enumerate(files_to_process):
            # Check if we're resuming this specific file
            skip_games: int = 0
            file_start_time: float = time.time()
            if current_file and pgn_file == current_file:
                skip_games = current_file_games
                print(
                    f"\n[File {file_idx + 1}/{len(files_to_process)}]"
                    f" Resuming: {pgn_file} (skipping {skip_games} games)"
                )
            else:
                print(
                    f"\n[File {file_idx + 1}/{len(files_to_process)}] Processing: {pgn_file}"
                )

            # Suppress chunk_manager verbose output during progress
            chunk_manager.verbose = False
            last_chunk_count: int = chunk_manager._chunk_count

            # Track current file for crash recovery
            chunk_manager.set_current_file(pgn_file, skip_games)

            processor: PGNProcessor = PGNProcessor(
                pgn_file,
                min_elo=cfg.min_elo,
                # Per-file limit
                max_games=cfg.max_games,
                skip_first_n_moves=cfg.skip_first_n_moves,
                skip_first_n_games=skip_games,
            )

            history: PositionHistory = PositionHistory(3)
            prev_game_idx: int = -1
            file_position_count: int = 0

            for board, move, outcome in processor.process_all():
                if cfg.max_positions and total_position_count >= cfg.max_positions:
                    break

                game_idx: int = processor.games_processed

                if game_idx != prev_game_idx:
                    history.clear()
                    prev_game_idx = game_idx
                    # Update current file progress for crash recovery
                    chunk_manager.set_current_file(pgn_file, skip_games + game_idx)

                history.push(board)
                state = history.encode(from_perspective=True)

                flip: bool = board.turn == False
                move_idx: int | None = encode_move_from_perspective(move, flip)
                if move_idx is None:
                    continue

                value: float = outcome if board.turn else -outcome

                # Calculate tactical weight for this position
                weight: float = calculate_tactical_weight(board, move)

                chunk_manager.add_example(state, move_idx, value, weight=weight)
                file_position_count += 1
                total_position_count += 1

                # Update progress on same line (every new chunk)
                current_chunks: int = chunk_manager._chunk_count
                if current_chunks > last_chunk_count:
                    last_chunk_count = current_chunks
                    file_elapsed: float = time.time() - file_start_time
                    progress: float = processor.progress
                    if progress > 0.001:
                        file_eta_seconds: float = (
                            file_elapsed / progress
                        ) - file_elapsed
                        if file_eta_seconds > 3600:
                            eta_str: str = f"{file_eta_seconds/3600:.1f}h"
                        elif file_eta_seconds > 60:
                            eta_str = f"{file_eta_seconds/60:.0f}m"
                        else:
                            eta_str = f"{file_eta_seconds:.0f}s"
                    else:
                        eta_str = "..."
                    speed: float = (
                        file_position_count / file_elapsed if file_elapsed > 0 else 0
                    )
                    # Show cumulative totals
                    cumul_pos: int = initial_positions + total_position_count
                    cumul_games: int = (
                        initial_games + total_games_count + processor.games_processed
                    )
                    print(
                        f"\r  {cumul_pos:,} pos | {cumul_games:,} games | "
                        f"{current_chunks} chunks | {progress:.1%} | {speed:.0f} pos/s | ETA: {eta_str}    ",
                        end="",
                        flush=True,
                    )

            # Track this file as processed
            processed_files.add(pgn_file)
            total_games_count += processor.games_processed + skip_games
            # File fully processed
            chunk_manager.clear_current_file()
            file_time: float = time.time() - file_start_time
            final_chunks: int = chunk_manager._chunk_count
            if file_time > 60:
                time_str: str = f"{file_time/60:.1f}m"
            else:
                time_str = f"{file_time:.0f}s"
            # Print final summary with cumulative totals
            cumul_pos = initial_positions + total_position_count
            cumul_games = initial_games + total_games_count
            print(
                f"\n  File done: +{file_position_count:,} pos, +{processor.games_processed} games in {time_str}"
            )
            print(
                f"  Total: {cumul_pos:,} positions, {cumul_games:,} games, {final_chunks} chunks"
            )

            # Check global position limit
            if cfg.max_positions and total_position_count >= cfg.max_positions:
                print(f"\nReached max_positions limit ({cfg.max_positions:,})")
                break

        chunk_manager.set_games_processed(total_games_count)
        chunk_manager.set_processed_files(list(processed_files))
        chunk_manager.finalize()
        metadata = ChunkManager.load_metadata(cfg.chunks_dir)
    else:
        print(f"\nAll {len(pgn_paths)} PGN file(s) already processed.")

    if metadata:
        print("\nChunk info:")
        print(f"  Chunks: {metadata['num_chunks']}")
        print(f"  Total positions: {metadata['total_examples']:,}")
        games_processed: int = metadata.get("games_processed", 0)
        print(
            f"  Games processed: {games_processed:,}"
            if games_processed
            else "  Games processed: unknown"
        )

        # Estimate memory for full load (72 planes per position)
        mem_gb: float = metadata["total_examples"] * (72 * 8 * 8 * 4 + 6) / (1024**3)
        print(f"  Est. RAM needed: {mem_gb:.1f} GB")

        # Auto-enable streaming if too much data
        if not streaming and mem_gb > 6:
            print("  Auto-enabling streaming mode (>6GB needed)")
            streaming = True

    # Create network
    if network_cfg:
        network: DualHeadNetwork = DualHeadNetwork(
            num_filters=network_cfg.num_filters,
            num_residual_blocks=network_cfg.num_residual_blocks,
        )
    else:
        network = DualHeadNetwork()

    if streaming:
        # Use streaming trainer
        print("\nUsing streaming trainer (memory efficient)...")
        print(f"  Value loss weight: {cfg.value_loss_weight}")
        print(f"  Entropy coefficient: {cfg.entropy_coefficient}")
        print(f"  Label smoothing: {cfg.label_smoothing}")
        print(f"  Prefetch workers: {cfg.prefetch_workers}")
        print(f"  Gradient accumulation steps: {cfg.gradient_accumulation_steps}")
        trainer: StreamingTrainer = StreamingTrainer(
            network=network,
            chunks_dir=cfg.chunks_dir,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            validation_split=cfg.validation_split,
            patience=cfg.patience,
            output_path=cfg.output_path,
            verbose=True,
            resume_from=resume_from,
            value_loss_weight=cfg.value_loss_weight,
            entropy_coefficient=cfg.entropy_coefficient,
            label_smoothing=cfg.label_smoothing,
            prefetch_workers=cfg.prefetch_workers,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            # Training dynamics (prevent catastrophic forgetting)
            weight_decay=cfg.weight_decay,
            gradient_clip_norm=cfg.gradient_clip_norm,
            lr_decay_factor=cfg.lr_decay_factor,
            lr_decay_patience=cfg.lr_decay_patience,
            min_learning_rate=cfg.min_learning_rate,
            checkpoint_keep_last=cfg.checkpoint_keep_last,
            # Anti-forgetting: Tactical Replay Buffer
            tactical_replay_enabled=cfg.tactical_replay_enabled,
            tactical_replay_ratio=cfg.tactical_replay_ratio,
            tactical_replay_threshold=cfg.tactical_replay_threshold,
            tactical_replay_capacity=cfg.tactical_replay_capacity,
            # Anti-forgetting: Knowledge Distillation
            teacher_enabled=cfg.teacher_enabled,
            teacher_path=cfg.teacher_path,
            teacher_alpha=cfg.teacher_alpha,
            teacher_temperature=cfg.teacher_temperature,
            # Anti-forgetting: EWC
            ewc_enabled=cfg.ewc_enabled,
            ewc_lambda=cfg.ewc_lambda,
            ewc_start_epoch=cfg.ewc_start_epoch,
            ewc_fisher_samples=cfg.ewc_fisher_samples,
            # Testing
            max_chunks=cfg.max_chunks,
        )
        return trainer.train(cfg.epochs)

    # Standard mode: load all into RAM
    print("\nLoading dataset into RAM...")
    dataset: ChessPositionDataset = ChessPositionDataset(
        cfg.pgn_path,
        chunks_dir=cfg.chunks_dir,
        min_elo=cfg.min_elo,
        max_games=cfg.max_games,
        max_positions=cfg.max_positions,
        skip_first_n_moves=cfg.skip_first_n_moves,
        verbose=True,
    )

    print("\nDataset statistics:")
    stats: dict[str, int | float] = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        dataset, cfg.batch_size, cfg.validation_split
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    network.to(device)

    # Optimizer and scheduler
    optimizer: optim.AdamW = optim.AdamW(
        network.parameters(),
        lr=cfg.learning_rate,
        # Increased for regularization
        weight_decay=5e-4,
    )

    # ReduceLROnPlateau - reduces LR when val loss stops improving
    scheduler: optim.lr_scheduler.ReduceLROnPlateau = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )
    )

    scaler: GradScaler | None = GradScaler("cuda") if use_amp else None

    # Checkpoint manager
    checkpoint_dir: str = os.path.dirname(cfg.output_path) or "models"
    checkpoint_name: str = os.path.splitext(os.path.basename(cfg.output_path))[0]
    checkpoint_manager: CheckpointManager = CheckpointManager(
        checkpoint_dir, verbose=True
    )

    # Training state
    best_val_loss: float = float("inf")
    best_count: int = 0
    epochs_without_improvement: int = 0
    start_epoch: int = 0

    # Handle resume
    if resume_from is not None:
        # Parse resume_from
        if resume_from.lower() == "latest":
            best_count_to_load: int | None = None
        else:
            try:
                best_count_to_load = int(resume_from)
            except ValueError:
                print(f"Invalid resume value: {resume_from}")
                print("Use 'latest' or a checkpoint number")
                best_count_to_load = None
                resume_from = None

        if resume_from is not None:
            checkpoint: dict[str, Any] | None = (
                checkpoint_manager.load_best_numbered_checkpoint(
                    checkpoint_name, best_count_to_load
                )
            )

            if checkpoint is not None:
                # Load network weights
                network = DualHeadNetwork.load(checkpoint["network_path"])
                network.to(device)
                print(f"Loaded network from: {checkpoint['network_path']}")

                # Re-create optimizer with loaded network parameters
                optimizer = optim.AdamW(
                    network.parameters(),
                    lr=cfg.learning_rate,
                    # Increased for regularization
                    weight_decay=5e-4,
                )

                # Recreate scaler (don't load old state - causes "No inf checks" error)
                if use_amp:
                    scaler = GradScaler("cuda")

                # Load training state
                state = checkpoint.get("state")
                if state:
                    if "optimizer_state" in state:
                        optimizer.load_state_dict(state["optimizer_state"])
                    if "best_val_loss" in state:
                        best_val_loss = state["best_val_loss"]
                    if "best_count" in state:
                        best_count = state["best_count"]
                    # Note: Don't load scaler state - it causes issues with new optimizer
                    if "epoch" in state:
                        start_epoch = state["epoch"] + 1
                    else:
                        # Old checkpoint without epoch - estimate from best_count
                        start_epoch = best_count
                        print(
                            "(Old checkpoint format - estimating epoch from best_count)"
                        )

                    # Restore scheduler state
                    if "scheduler_state" in state:
                        scheduler.load_state_dict(state["scheduler_state"])
                        print(
                            f"Restored scheduler state (LR: {optimizer.param_groups[0]['lr']:.6f})"
                        )

                    # Restore epochs without improvement
                    if "epochs_without_improvement" in state:
                        epochs_without_improvement = state["epochs_without_improvement"]

                    print(f"Resuming from epoch {start_epoch + 1}")
                    print(f"Best val loss so far: {best_val_loss:.4f}")
                    print(f"Best count: {best_count}")
            else:
                print(f"No checkpoint found to resume from")

    print("\nStarting training...\n")
    for epoch in range(start_epoch, cfg.epochs):
        epoch_start: float = time.time()

        # === Training phase ===
        print(f"Epoch {epoch+1}/{cfg.epochs} loading batches...", flush=True)
        network.train()
        train_policy_loss: float = 0.0
        train_value_loss: float = 0.0
        train_batches: int = 0
        total_batches: int = len(train_loader)

        last_pct: int = -1
        for batch_idx, (states, policies, values) in enumerate(train_loader):
            # Update progress only when percentage changes
            pct: int = (batch_idx + 1) * 100 // total_batches
            if pct != last_pct:
                last_pct = pct
                elapsed: int = int(time.time() - epoch_start)
                print(
                    f"\rEpoch {epoch+1}/{cfg.epochs} ({elapsed}s) train {pct}%   ",
                    end="",
                    flush=True,
                )

            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type="cuda"):
                    pred_policies, pred_values, _ = network(states)
                    policy_loss: torch.Tensor = nn.functional.cross_entropy(
                        pred_policies, policies, label_smoothing=0.1
                    )
                    value_loss: torch.Tensor = nn.functional.mse_loss(
                        pred_values, values
                    )

                    # Compute entropy bonus for policy diversity
                    log_probs: torch.Tensor = nn.functional.log_softmax(
                        pred_policies, dim=-1
                    )
                    probs: torch.Tensor = nn.functional.softmax(pred_policies, dim=-1)
                    entropy: torch.Tensor = -torch.sum(probs * log_probs, dim=-1).mean()

                    loss = (
                        policy_loss
                        + cfg.value_loss_weight * value_loss
                        - cfg.entropy_coefficient * entropy
                    )

                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_policies, pred_values, _ = network(states)
                policy_loss = nn.functional.cross_entropy(
                    pred_policies, policies, label_smoothing=0.1
                )
                value_loss = nn.functional.mse_loss(pred_values, values)

                # Compute entropy bonus for policy diversity
                log_probs = nn.functional.log_softmax(pred_policies, dim=-1)
                probs = nn.functional.softmax(pred_policies, dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1).mean()

                loss = (
                    policy_loss
                    + cfg.value_loss_weight * value_loss
                    - cfg.entropy_coefficient * entropy
                )

                loss.backward()
                optimizer.step()

            train_policy_loss += policy_loss.item()
            train_value_loss += value_loss.item()
            train_batches += 1

        # === Validation phase ===
        network.eval()
        val_policy_loss: float = 0.0
        val_value_loss: float = 0.0
        val_batches: int = 0

        total_val_batches: int = len(val_loader)
        last_pct = -1
        with torch.inference_mode():
            for batch_idx, (states, policies, values) in enumerate(val_loader):
                # Update progress only when percentage changes
                pct = (batch_idx + 1) * 100 // total_val_batches
                if pct != last_pct:
                    last_pct = pct
                    elapsed = int(time.time() - epoch_start)
                    print(
                        f"\rEpoch {epoch+1}/{cfg.epochs} ({elapsed}s) val {pct}%   ",
                        end="",
                        flush=True,
                    )

                states = states.to(device)
                policies = policies.to(device)
                values = values.to(device)

                pred_policies, pred_values, _ = network(states)
                policy_loss = nn.functional.cross_entropy(
                    pred_policies, policies, label_smoothing=0.1
                )
                value_loss = nn.functional.mse_loss(pred_values, values)

                val_policy_loss += policy_loss.item()
                val_value_loss += value_loss.item()
                val_batches += 1

        # Calculate averages
        avg_train_policy: float = train_policy_loss / max(1, train_batches)
        avg_train_value: float = train_value_loss / max(1, train_batches)
        avg_train_total: float = avg_train_policy + avg_train_value
        avg_val_policy: float = val_policy_loss / max(1, val_batches)
        avg_val_value: float = val_value_loss / max(1, val_batches)
        avg_val_total: float = avg_val_policy + avg_val_value

        # Update scheduler with validation loss (ReduceLROnPlateau)
        scheduler.step(avg_val_total)

        epoch_time: float = time.time() - epoch_start

        # Get current LR
        current_lr: float = optimizer.param_groups[0]["lr"]

        # === Print epoch summary ===
        print(f"\rEpoch {epoch+1}/{cfg.epochs} ({epoch_time:.1f}s)")
        print(
            f"  Train - Loss: {avg_train_total:.4f} (P: {avg_train_policy:.4f}, V: {avg_train_value:.4f})"
        )
        print(
            f"  Val   - Loss: {avg_val_total:.4f} (P: {avg_val_policy:.4f}, V: {avg_val_value:.4f})"
        )
        print(f"  LR: {current_lr:.6f}")

        # === Save best model ===
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            best_count += 1
            epochs_without_improvement = 0

            # Save training state for potential resume
            state = {
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_count": best_count,
                "epochs_without_improvement": epochs_without_improvement,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            if scaler is not None:
                state["scaler_state"] = scaler.state_dict()

            # Save checkpoints (manager is silent, we handle messages)
            checkpoint_manager.verbose = False
            checkpoint_manager.save_best_checkpoint(
                checkpoint_name, best_count, network, state
            )

            # Print checkpoint info
            best_path: str = f"{checkpoint_dir}/{checkpoint_name}_best_network.pt"
            print(f"Checkpoint saved: {best_path}")

            is_milestone: bool = best_count == 1 or best_count % 5 == 0
            if is_milestone:
                milestone_path: str = (
                    f"{checkpoint_dir}/{checkpoint_name}_best_{best_count}_network.pt"
                )
                print(f"Checkpoint saved: {milestone_path}")
                print(f"  New best model saved! (milestone #{best_count})")
            else:
                print(f"  New best model saved! (#{best_count})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{cfg.patience})")

        # Blank line between epochs
        print()

        # Early stopping
        if epochs_without_improvement >= cfg.patience:
            print(f"Early stopping: no improvement for {cfg.patience} epochs")
            break

    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Total improvements: {best_count}")

    # Load best model
    best_path = checkpoint_dir + f"/{checkpoint_name}_best_network.pt"
    network = DualHeadNetwork.load(best_path)
    return network


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pretrain chess network on master games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./neural_mate_pretrain --config config.json
  ./neural_mate_pretrain --config config.json --epochs 10 --max-games 5000
  ./neural_mate_pretrain --pgn data/lichess_elite_2020-08.pgn --epochs 5
  ./neural_mate_pretrain --pgn data/jan.pgn data/feb.pgn data/mar.pgn
  ./neural_mate_pretrain --pgn "data/lichess_*.pgn"
  ./neural_mate_pretrain --resume-pretrained latest
  ./neural_mate_pretrain --resume-pretrained 3 --epochs 10
  ./neural_mate_pretrain --generate-config
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/config.json",
        help="Path to JSON config file (default: config/config.json)",
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Print default config and exit",
    )
    parser.add_argument(
        "--pgn",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to PGN file(s). Supports multiple files and glob patterns (e.g., 'data/*.pgn')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for trained network",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=None,
        help="Number of training epochs",
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
        "--min-elo",
        type=int,
        default=None,
        help="Minimum player ELO",
    )
    parser.add_argument(
        "--max-elo",
        type=int,
        default=None,
        help="Maximum player ELO",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum games to process",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Maximum positions to use",
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default=None,
        help="Directory for chunk files",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum chunks to use per epoch (for quick testing)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode (loads all data into RAM)",
    )
    parser.add_argument(
        "--resume-pretrained",
        type=str,
        default=None,
        help="Resume from checkpoint: 'latest' or epoch number",
    )

    args: argparse.Namespace = parser.parse_args()

    # Generate default config
    if args.generate_config:
        from config import generate_default_config  # type: ignore[import-not-found]

        print(generate_default_config())
        return 0

    # Load config
    if args.config and os.path.exists(args.config):
        config: Config = Config.load(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        config = Config.default()
        if args.config and not os.path.exists(args.config):
            print(f"Config file not found: {args.config}, using defaults")

    cfg: PretrainingConfig = config.pretraining

    # Override with CLI arguments
    if args.pgn:
        expanded: list[str] = expand_pgn_paths(args.pgn)
        if len(expanded) == 1:
            cfg.pgn_path = expanded[0]
            cfg.pgn_paths = None
        else:
            cfg.pgn_paths = expanded
            cfg.pgn_path = expanded[0] if expanded else ""
    if args.output:
        cfg.output_path = args.output
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.min_elo:
        cfg.min_elo = args.min_elo
    if args.max_games:
        cfg.max_games = args.max_games
    if args.max_positions:
        cfg.max_positions = args.max_positions
    if args.chunks_dir:
        cfg.chunks_dir = args.chunks_dir
    if args.max_chunks:
        cfg.max_chunks = args.max_chunks

    # Validate required fields
    pgn_paths: list[str] = cfg.get_pgn_paths()
    if not pgn_paths:
        print("Error: No PGN files specified")
        print("Use --pgn to specify PGN file(s) or update config.json")
        return 1

    missing: list[str] = [p for p in pgn_paths if not os.path.exists(p)]
    if missing:
        print("Error: PGN file(s) not found:")
        for p in missing:
            print(f"  - {p}")
        return 1

    try:
        pretrain(
            cfg,
            config.network,
            streaming=not args.no_streaming,
            resume_from=args.resume_pretrained,
        )
        return 0
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
