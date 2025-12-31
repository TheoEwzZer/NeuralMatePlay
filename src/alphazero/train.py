"""
Self-play training script for AlphaZero.

Usage:
    ./neural_mate_train --config config.json
    ./neural_mate_train --iterations 100 --games 50
    ./neural_mate_train --checkpoint models/pretrained.pt --iterations 100
    ./neural_mate_train --resume-trained latest
    ./neural_mate_train --resume-trained 10 --iterations 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphazero import DualHeadNetwork, AlphaZeroTrainer
from config import Config, TrainingConfig, generate_default_config


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
        "--iterations", "-i",
        type=int,
        default=None,
        help="Number of training iterations",
    )

    parser.add_argument(
        "--games", "-g",
        type=int,
        default=None,
        help="Games per iteration",
    )

    parser.add_argument(
        "--simulations", "-s",
        type=int,
        default=None,
        help="MCTS simulations per move",
    )

    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for checkpoints",
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

    # Print config
    print("\n[Configuration]")
    print(f"  Iterations: {cfg.iterations}")
    print(f"  Games per iteration: {cfg.games_per_iteration}")
    print(f"  MCTS simulations: {cfg.num_simulations}")
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
            num_input_planes=config.network.num_input_planes,
        )
    else:
        print("\nCreating new network")
        network = DualHeadNetwork()

    # Sync history_length with network architecture
    # Formula: num_input_planes = (history_length + 1) * 12 + 12 metadata (includes attack maps)
    network_history_length = (network.num_input_planes - 12) // 12 - 1
    if network_history_length != cfg.history_length:
        print(f"  Adjusting history_length: {cfg.history_length} -> {network_history_length} (from network)")
        cfg.history_length = network_history_length

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

    # Training state for callback
    iteration_stats = {"prev_loss": None, "last_phase": None}

    # Training callback
    def callback(data: dict) -> None:
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
            line = (
                f"  Self-play: {games}/{total} | {examples} ex | "
                f"{avg_moves:.0f} moves | {avg_time:.1f}s/game | W:{w} B:{b} D:{d}"
            )
            # Pad to 100 chars to ensure clean overwrite
            print(f"\r{line:<100}", end="", flush=True)

        elif phase == "training_skip":
            reason = data.get("reason", "")
            print(f"\n  Skipping training: {reason}")

        elif phase == "training_start":
            # Print newline to preserve self-play line
            print()

        elif phase == "training":
            epoch = data.get("epoch", 0)
            epochs = data.get("epochs", 0)
            total_loss = data.get("total_loss", 0)
            policy_loss = data.get("policy_loss", 0)
            value_loss = data.get("value_loss", 0)
            if epochs > 0:
                line = (
                    f"  Epoch {epoch}/{epochs} | Loss: {total_loss:.4f} "
                    f"(policy: {policy_loss:.4f}, value: {value_loss:.4f})"
                )
                print(f"\r{line:<80}", end="", flush=True)

        elif phase == "iteration_complete":
            stats = data.get("stats", {})
            training_stats = stats.get("training_stats", {})
            final_loss = training_stats.get("avg_total_loss", 0)
            final_policy = training_stats.get("avg_policy_loss", 0)
            final_value = training_stats.get("avg_value_loss", 0)
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
                print(
                    f"\n  Final: {final_loss:.4f}{trend} "
                    f"(policy: {final_policy:.4f}, value: {final_value:.4f})"
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
            print(f"\r  Arena: vs {opponent}...                              ", end="", flush=True)

        elif phase == "arena_complete":
            elo = data.get("elo", 1500)
            best_elo = data.get("best_elo", elo)
            is_new_best = data.get("is_new_best", False)
            veto = data.get("veto", False)
            veto_reason = data.get("veto_reason")
            vs_random = data.get("vs_random", {})
            vs_best = data.get("vs_best")
            vs_old = data.get("vs_old")

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
                print(f"  vs Best #{best_iter}: {w}W-{l}L-{d}D ({sc:.0f}% score) {t:.1f}s/game")

            # vs Old
            if vs_old:
                w = vs_old.get("wins", 0)
                l = vs_old.get("losses", 0)
                d = vs_old.get("draws", 0)
                sc = vs_old.get("score", 0.5) * 100
                old_iter = vs_old.get("from_iteration", "?")
                t = vs_old.get("avg_time", 0)
                print(f"  vs Old #{old_iter}: {w}W-{l}L-{d}D ({sc:.0f}% score) {t:.1f}s/game")

            # ELO
            print(f"  Elo: {elo:.0f} (best: {best_elo:.0f})")

            # Promotion status
            if is_new_best and not veto:
                print("  NEW BEST MODEL!")
            elif veto:
                print("  VETO: Catastrophic forgetting detected!")
                print(f"  {veto_reason}")

            print("  ======================================")

    # Run training
    remaining = cfg.iterations - start_iteration
    print(f"\nStarting training for {remaining} iterations (total: {cfg.iterations})")
    if start_iteration > 0:
        print(f"Resuming from iteration {start_iteration}")
    print()

    try:
        for i in range(start_iteration, cfg.iterations):
            print(f"=== Iteration {i + 1}/{cfg.iterations} ===")
            trainer.train_iteration(callback)
            print()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return 1

    print("Training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
