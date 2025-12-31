"""
Pretraining script for supervised learning on master games.

Usage:
    ./neural_mate_pretrain --config config.json
    ./neural_mate_pretrain --pgn data/lichess_elite_2020-08.pgn --epochs 5
    ./neural_mate_pretrain --resume-pretrained latest
    ./neural_mate_pretrain --resume-pretrained 3 --epochs 10
"""

import json
import os
import argparse
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphazero import DualHeadNetwork, get_device, supports_mixed_precision
from alphazero.checkpoint_manager import CheckpointManager
from config import Config, PretrainingConfig
from .dataset import ChessPositionDataset
from .chunk_manager import ChunkManager
from .streaming_trainer import StreamingTrainer


def create_dataloaders(
    dataset: ChessPositionDataset,
    batch_size: int = 256,
    validation_split: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def pretrain(
    cfg: PretrainingConfig,
    network_cfg=None,
    streaming: bool = False,
    resume_from: Optional[str] = None,
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
    device = get_device()
    use_amp = supports_mixed_precision()

    print(f"Device: {device}")
    print(f"Mixed precision: {use_amp}")

    # Print config
    print("\n[Configuration]")
    print(f"  PGN: {cfg.pgn_path}")
    print(f"  Output: {cfg.output_path}")
    print(f"  Chunks: {cfg.chunks_dir}")
    print(f"  Chunk size: {cfg.chunk_size}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Max games: {cfg.max_games or 'unlimited'}")
    print(f"  Streaming: {streaming}")

    # Check if we need to create or extend chunks
    metadata = ChunkManager.load_metadata(cfg.chunks_dir)
    games_already_processed = metadata.get("games_processed", 0) if metadata else 0
    need_more_games = cfg.max_games and games_already_processed < cfg.max_games

    if metadata is None or need_more_games:
        from .pgn_processor import PGNProcessor
        from alphazero.spatial_encoding import PositionHistory
        from alphazero.move_encoding import encode_move_from_perspective

        if metadata is None:
            print(f"\nNo chunks found, creating from PGN...")
            resume_mode = False
            skip_games = 0
        else:
            print(f"\nExtending chunks: {games_already_processed} → {cfg.max_games} games...")
            resume_mode = True
            skip_games = games_already_processed

        processor = PGNProcessor(
            cfg.pgn_path,
            min_elo=cfg.min_elo,
            max_games=cfg.max_games,
            skip_first_n_moves=cfg.skip_first_n_moves,
        )

        history = PositionHistory(3)  # Default history length
        prev_game_idx = -1

        print(f"Processing up to {cfg.max_games or 'all'} games...")
        if skip_games > 0:
            print(f"Skipping first {skip_games} games (already processed)...")

        chunk_manager = ChunkManager(cfg.chunks_dir, chunk_size=cfg.chunk_size, verbose=True, resume=resume_mode)
        position_count = 0
        games_skipped = 0

        last_skip_print = 0
        for board, move, outcome, phase in processor.process_all():
            if cfg.max_positions and position_count >= cfg.max_positions:
                break

            game_idx = processor.games_processed

            # Skip games that were already processed
            if game_idx <= skip_games:
                if game_idx != prev_game_idx:
                    games_skipped = game_idx
                    prev_game_idx = game_idx
                    # Show progress while skipping (every 5000 games)
                    if game_idx - last_skip_print >= 5000:
                        print(f"  Skipping... {game_idx:,}/{skip_games:,} games", end="\r", flush=True)
                        last_skip_print = game_idx
                continue

            # First position after skipping - print completion message
            if skip_games > 0 and position_count == 0:
                print(f"  Skipped {skip_games:,} games                    ")

            if game_idx != prev_game_idx:
                history.clear()
                prev_game_idx = game_idx

            history.push(board)
            state = history.encode(from_perspective=True)

            flip = board.turn == False
            move_idx = encode_move_from_perspective(move, flip)
            if move_idx is None:
                continue

            value = outcome if board.turn else -outcome
            chunk_manager.add_example(state, move_idx, value, phase)
            position_count += 1

            if position_count % 50000 == 0:
                total_games = processor.games_processed
                new_games = total_games - skip_games
                print(f"  {position_count:,} new positions, {new_games} new games, {processor.progress:.1%}")

        chunk_manager.set_games_processed(processor.games_processed)
        chunk_manager.finalize()
        metadata = ChunkManager.load_metadata(cfg.chunks_dir)

    if metadata:
        print(f"\nChunk info:")
        print(f"  Chunks: {metadata['num_chunks']}")
        print(f"  Total positions: {metadata['total_examples']:,}")
        games_processed = metadata.get('games_processed', 0)
        print(f"  Games processed: {games_processed:,}" if games_processed else "  Games processed: unknown")

        # Estimate memory for full load
        mem_gb = metadata['total_examples'] * (54 * 8 * 8 * 4 + 6) / (1024**3)
        print(f"  Est. RAM needed: {mem_gb:.1f} GB")

        # Auto-enable streaming if too much data
        if not streaming and mem_gb > 12:
            print(f"  Auto-enabling streaming mode (>12GB needed)")
            streaming = True

    # Create network
    if network_cfg:
        network = DualHeadNetwork(
            num_filters=network_cfg.num_filters,
            num_residual_blocks=network_cfg.num_residual_blocks,
            num_input_planes=network_cfg.num_input_planes,
        )
    else:
        network = DualHeadNetwork()

    if streaming:
        # Use streaming trainer
        print("\nUsing streaming trainer (memory efficient)...")
        print(f"  Value loss weight: {cfg.value_loss_weight}")
        print(f"  Entropy coefficient: {cfg.entropy_coefficient}")
        print(f"  Prefetch workers: {cfg.prefetch_workers}")
        print(f"  Gradient accumulation steps: {cfg.gradient_accumulation_steps}")
        trainer = StreamingTrainer(
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
            prefetch_workers=cfg.prefetch_workers,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        )
        return trainer.train(cfg.epochs)

    # Standard mode: load all into RAM
    print("\nLoading dataset into RAM...")
    dataset = ChessPositionDataset(
        cfg.pgn_path,
        chunks_dir=cfg.chunks_dir,
        min_elo=cfg.min_elo,
        max_games=cfg.max_games,
        max_positions=cfg.max_positions,
        skip_first_n_moves=cfg.skip_first_n_moves,
        verbose=True,
    )

    print(f"\nDataset statistics:")
    stats = dataset.get_statistics()
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
    optimizer = optim.AdamW(
        network.parameters(),
        lr=cfg.learning_rate,
        weight_decay=5e-4,  # Increased for regularization
    )

    # ReduceLROnPlateau - reduces LR when val loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    scaler = GradScaler('cuda') if use_amp else None

    # Checkpoint manager
    checkpoint_dir = os.path.dirname(cfg.output_path) or "models"
    checkpoint_name = os.path.splitext(os.path.basename(cfg.output_path))[0]
    checkpoint_manager = CheckpointManager(checkpoint_dir, verbose=True)

    # Training state
    best_val_loss = float("inf")
    best_count = 0
    epochs_without_improvement = 0
    start_epoch = 0

    # Handle resume
    if resume_from is not None:
        # Parse resume_from
        if resume_from.lower() == "latest":
            best_count_to_load = None
        else:
            try:
                best_count_to_load = int(resume_from)
            except ValueError:
                print(f"Invalid resume value: {resume_from}")
                print("Use 'latest' or a checkpoint number")
                best_count_to_load = None
                resume_from = None

        if resume_from is not None:
            checkpoint = checkpoint_manager.load_best_numbered_checkpoint(
                checkpoint_name, best_count_to_load
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
                    weight_decay=5e-4,  # Increased for regularization
                )

                # Recreate scaler (don't load old state - causes "No inf checks" error)
                if use_amp:
                    scaler = GradScaler('cuda')

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
                        print(f"(Old checkpoint format - estimating epoch from best_count)")

                    # Restore scheduler state
                    if "scheduler_state" in state:
                        scheduler.load_state_dict(state["scheduler_state"])
                        print(f"Restored scheduler state (LR: {optimizer.param_groups[0]['lr']:.6f})")

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
        epoch_start = time.time()
        last_print_time = -1

        # === Training phase ===
        print(f"Epoch {epoch+1}/{cfg.epochs} loading batches...", flush=True)
        network.train()
        train_policy_loss = 0.0
        train_value_loss = 0.0
        train_batches = 0
        total_batches = len(train_loader)

        last_pct = -1
        for batch_idx, (states, policies, values) in enumerate(train_loader):
            # Update progress only when percentage changes
            pct = (batch_idx + 1) * 100 // total_batches
            if pct != last_pct:
                last_pct = pct
                elapsed = int(time.time() - epoch_start)
                print(f"\rEpoch {epoch+1}/{cfg.epochs} ({elapsed}s) train {pct}%   ", end="", flush=True)

            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type='cuda'):
                    pred_policies, pred_values, _ = network(states)
                    policy_loss = nn.functional.cross_entropy(
                        pred_policies, policies, label_smoothing=0.2
                    )
                    value_loss = nn.functional.mse_loss(pred_values, values)

                    # Compute entropy bonus for policy diversity
                    log_probs = nn.functional.log_softmax(pred_policies, dim=-1)
                    probs = nn.functional.softmax(pred_policies, dim=-1)
                    entropy = -torch.sum(probs * log_probs, dim=-1).mean()

                    loss = policy_loss + cfg.value_loss_weight * value_loss - cfg.entropy_coefficient * entropy

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_policies, pred_values, _ = network(states)
                policy_loss = nn.functional.cross_entropy(
                    pred_policies, policies, label_smoothing=0.2
                )
                value_loss = nn.functional.mse_loss(pred_values, values)

                # Compute entropy bonus for policy diversity
                log_probs = nn.functional.log_softmax(pred_policies, dim=-1)
                probs = nn.functional.softmax(pred_policies, dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1).mean()

                loss = policy_loss + cfg.value_loss_weight * value_loss - cfg.entropy_coefficient * entropy

                loss.backward()
                optimizer.step()

            train_policy_loss += policy_loss.item()
            train_value_loss += value_loss.item()
            train_batches += 1

        # === Validation phase ===
        network.eval()
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_batches = 0

        total_val_batches = len(val_loader)
        last_pct = -1
        with torch.inference_mode():
            for batch_idx, (states, policies, values) in enumerate(val_loader):
                # Update progress only when percentage changes
                pct = (batch_idx + 1) * 100 // total_val_batches
                if pct != last_pct:
                    last_pct = pct
                    elapsed = int(time.time() - epoch_start)
                    print(f"\rEpoch {epoch+1}/{cfg.epochs} ({elapsed}s) val {pct}%   ", end="", flush=True)

                states = states.to(device)
                policies = policies.to(device)
                values = values.to(device)

                pred_policies, pred_values, _ = network(states)
                policy_loss = nn.functional.cross_entropy(
                    pred_policies, policies, label_smoothing=0.2
                )
                value_loss = nn.functional.mse_loss(pred_values, values)

                val_policy_loss += policy_loss.item()
                val_value_loss += value_loss.item()
                val_batches += 1

        # Calculate averages
        avg_train_policy = train_policy_loss / max(1, train_batches)
        avg_train_value = train_value_loss / max(1, train_batches)
        avg_train_total = avg_train_policy + avg_train_value
        avg_val_policy = val_policy_loss / max(1, val_batches)
        avg_val_value = val_value_loss / max(1, val_batches)
        avg_val_total = avg_val_policy + avg_val_value

        # Update scheduler with validation loss (ReduceLROnPlateau)
        scheduler.step(avg_val_total)

        epoch_time = time.time() - epoch_start

        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']

        # === Print epoch summary ===
        print(f"\rEpoch {epoch+1}/{cfg.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {avg_train_total:.4f} (P: {avg_train_policy:.4f}, V: {avg_train_value:.4f})")
        print(f"  Val   - Loss: {avg_val_total:.4f} (P: {avg_val_policy:.4f}, V: {avg_val_value:.4f})")
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
            best_path = f"{checkpoint_dir}/{checkpoint_name}_best_network.pt"
            print(f"Checkpoint saved: {best_path}")

            is_milestone = (best_count == 1 or best_count % 5 == 0)
            if is_milestone:
                milestone_path = f"{checkpoint_dir}/{checkpoint_name}_best_{best_count}_network.pt"
                print(f"Checkpoint saved: {milestone_path}")
                print(f"  New best model saved! (milestone #{best_count})")
            else:
                print(f"  New best model saved! (#{best_count})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{cfg.patience})")

        print()  # Blank line between epochs

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
  ./neural_mate_pretrain --resume-pretrained latest
  ./neural_mate_pretrain --resume-pretrained 3 --epochs 10
  ./neural_mate_pretrain --generate-config
        """,
    )

    parser.add_argument(
        "--config", "-c",
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
        default=None,
        help="Path to PGN file (overrides config)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for trained network",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size", "-b",
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

    args = parser.parse_args()

    # Generate default config
    if args.generate_config:
        from config import generate_default_config
        print(generate_default_config())
        return 0

    # Load config
    if args.config and os.path.exists(args.config):
        config = Config.load(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        config = Config.default()
        if args.config and not os.path.exists(args.config):
            print(f"Config file not found: {args.config}, using defaults")

    cfg = config.pretraining

    # Override with CLI arguments
    if args.pgn:
        cfg.pgn_path = args.pgn
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

    # Validate required fields
    if not cfg.pgn_path or not os.path.exists(cfg.pgn_path):
        print(f"Error: PGN file not found: {cfg.pgn_path}")
        print("Use --pgn to specify a valid PGN file or update config.json")
        return 1

    try:
        pretrain(cfg, config.network, streaming=not args.no_streaming, resume_from=args.resume_pretrained)
        return 0
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
