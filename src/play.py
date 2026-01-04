"""
Main entry point for NeuralMate2 chess engine.

Unified CLI supporting play, match, training, and pretraining modes.

Usage:
    neural_mate_play [OPTIONS]

Options:
    --network, -n PATH    Load network file(s) (.pt). Use two for match mode.
    --simulations, -s N   Number of MCTS simulations per move (default: 200)
    --device, -d DEVICE   Device to use: 'cuda', 'cpu', or 'auto' (default: auto)
    --games, -g N         Number of games for match mode (default: 10)
    --train, -t           Start in training mode
    --pretrain            Supervised pretraining on master games
    --gui                 Visual mode (training or match with UI)
    --config, -c PATH     Training configuration file (JSON)
    --iterations, -i N    Number of training iterations (default: 10)
    --epochs, -e N        Number of epochs for pretraining (default: 10)
    --output, -o PATH     Output path for trained network
    --resume, -r NAME     Resume training from checkpoint ('latest' or 'iteration_N')
    --help, -h            Show this help message

Examples:
    # Play against untrained AI
    neural_mate_play

    # Play against trained AI
    neural_mate_play --network alphazero.pt

    # Match: two networks play against each other
    neural_mate_play --network model1.pt model2.pt --games 20

    # Match against random player (baseline test)
    neural_mate_play --network model.pt random --games 20

    # Match against pure MCTS (no neural network)
    neural_mate_play --network model.pt mcts --games 10 --simulations 800

    # Visual match: watch games in UI
    neural_mate_play --network model1.pt model2.pt --games 10 --gui

    # Increase AI strength (more thinking time)
    neural_mate_play --network alphazero.pt --simulations 800

    # Verbose mode: see MCTS search tree after each AI move
    neural_mate_play --network alphazero.pt --verbose

    # Force CPU mode
    neural_mate_play --network alphazero.pt --device cpu

    # Train a new network
    neural_mate_play --train --iterations 100 --output trained.pt

    # Train with custom config
    neural_mate_play --train --config configs/alphazero.json

    # Resume training from last checkpoint
    neural_mate_play --train --resume latest --iterations 100

    # Visual training (see games in real-time)
    neural_mate_play --train --gui --config configs/alphazero.json --iterations 50

    # Supervised pretraining on downloaded data
    neural_mate_play --pretrain --epochs 10

    # Resume self-play training from pretrained network
    neural_mate_play --train --resume pretrained --iterations 100
"""

import argparse
import os
import sys

# Add src directory to path for imports
_src_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_src_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def set_device(device: str) -> None:
    """Set the device to use for PyTorch."""
    if device and device != "auto":
        os.environ["NEURALMATE_DEVICE"] = device


def detect_mode(args) -> str:
    """Detect which mode to run based on arguments."""
    if args.pretrain:
        return "pretrain"
    if args.train:
        return "train"
    if args.network and len(args.network) >= 2:
        return "match"
    return "play"


def resolve_resume(resume_name: str, checkpoint_dir: str = "checkpoints") -> str:
    """
    Resolve resume checkpoint name to a file path.

    Args:
        resume_name: 'latest', 'pretrained', or 'iteration_N'
        checkpoint_dir: Directory containing checkpoints.

    Returns:
        Path to network file.
    """
    from alphazero.checkpoint_manager import CheckpointManager

    if resume_name == "pretrained":
        # Look for pretrained best network
        pretrained_path = os.path.join(checkpoint_dir, "pretrained_best_network.pt")
        if os.path.exists(pretrained_path):
            return pretrained_path
        # Fallback to other pretrained names
        for name in ["pretrained.pt", "pretrained_network.pt"]:
            path = os.path.join(checkpoint_dir, name)
            if os.path.exists(path):
                return path
        print(f"Warning: No pretrained network found in {checkpoint_dir}")
        return None

    cm = CheckpointManager(checkpoint_dir, verbose=False)

    if resume_name == "latest":
        latest = cm.get_latest_checkpoint()
        if latest:
            return os.path.join(checkpoint_dir, f"{latest}_network.pt")
        return None

    # Assume it's an iteration name like "iteration_10"
    path = os.path.join(checkpoint_dir, f"{resume_name}_network.pt")
    if os.path.exists(path):
        return path

    print(f"Warning: Checkpoint not found: {path}")
    return None


def run_gui(
    network_path: str = None,
    num_simulations: int = 200,
    verbose: bool = False,
) -> int:
    """Launch the graphical user interface for playing."""
    try:
        from ui.app import ChessGameApp

        app = ChessGameApp(
            network_path=network_path,
            num_simulations=num_simulations,
            verbose=verbose,
        )
        app.run()
        return 0

    except ImportError as e:
        print(f"Error importing UI module: {e}")
        print("Make sure tkinter is installed.")
        return 1


def run_cli(
    network_path: str = None,
    num_simulations: int = 200,
    batch_size: int = 16,
) -> int:
    """Run in command-line mode."""
    import chess

    from alphazero import DualHeadNetwork, MCTS, get_device

    print("NeuralMate2 Chess Engine - CLI Mode")
    print("=" * 40)
    print(f"Device: {get_device()}")

    # Load or create network
    if network_path and os.path.exists(network_path):
        print(f"Loading network from {network_path}")
        network = DualHeadNetwork.load(network_path)
    else:
        print("Creating new untrained network")
        network = DualHeadNetwork()

    # Create MCTS with batched inference
    mcts = MCTS(network, num_simulations=num_simulations, batch_size=batch_size)
    mcts.temperature = 0.1  # Deterministic play

    # Game loop
    board = chess.Board()

    print("\nCommands:")
    print("  Enter move in UCI format (e.g., e2e4)")
    print("  'quit' to exit")
    print("  'new' for new game")
    print("  'undo' to undo last move")
    print()

    while True:
        print(f"\n{board}")
        print()

        if board.is_game_over():
            print(f"Game over: {board.result()}")
            print("Type 'new' for a new game or 'quit' to exit")

        if board.turn == chess.WHITE:
            # Human plays white
            while True:
                user_input = input("Your move: ").strip().lower()

                if user_input == "quit":
                    print("Goodbye!")
                    return 0
                elif user_input == "new":
                    board = chess.Board()
                    mcts.clear_cache()
                    break
                elif user_input == "undo":
                    if len(board.move_stack) >= 2:
                        board.pop()
                        board.pop()
                        print("Undid last two moves")
                    break

                try:
                    move = chess.Move.from_uci(user_input)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move. Try again.")
                except ValueError:
                    print("Invalid move format. Use UCI notation (e.g., e2e4)")

        else:
            # Engine plays black
            if not board.is_game_over():
                print("Thinking...")
                move = mcts.get_best_move(board)
                if move:
                    print(f"Engine plays: {board.san(move)}")
                    board.push(move)
                else:
                    print("Engine has no move (game over?)")


def run_match(
    networks: list,
    num_games: int,
    num_simulations: int,
    gui: bool = False,
) -> int:
    """
    Run a match between two networks or network vs random.

    Args:
        networks: List of two network paths (or "random").
        num_games: Number of games to play.
        num_simulations: MCTS simulations per move.
        gui: Whether to show visual match.

    Returns:
        Exit code.
    """
    if len(networks) < 2:
        print("Error: Match mode requires two networks")
        return 1

    network1_path = networks[0]
    network2_path = networks[1]

    if gui:
        # Visual match mode
        try:
            from ui.match_app import MatchApp

            app = MatchApp(
                network1_path=network1_path,
                network2_path=network2_path,
                num_games=num_games,
                num_simulations=num_simulations,
            )
            app.run()
            return 0
        except ImportError as e:
            print(f"Error importing match UI: {e}")
            return 1
    else:
        # CLI match mode
        import chess
        from alphazero import DualHeadNetwork, get_device
        from alphazero.arena import Arena, NetworkPlayer, RandomPlayer, PureMCTSPlayer
        from src.chess_encoding.board_utils import get_raw_material_diff

        device = get_device()
        print(f"NeuralMate2 Match Mode")
        print("=" * 60)
        print(f"Device: {device}")

        # Helper to get history_length from network
        def get_history_length(network):
            planes = network.num_input_planes
            if planes == 18:
                return 0
            return (planes - 6) // 12 - 1

        # Load player 1
        history_length = 0
        if network1_path.lower() == "random":
            player1 = RandomPlayer(name="Random")
            name1 = "Random"
            printf("Player 1: Random")
        elif network1_path.lower() == "mcts":
            player1 = PureMCTSPlayer(num_simulations=num_simulations, name="PureMCTS")
            name1 = "PureMCTS"
            print(f"Player 1: Pure MCTS ({num_simulations} simulations)")
        else:
            print(f"Loading player 1: {network1_path}")
            net1 = DualHeadNetwork.load(network1_path, device=device)
            history_length = get_history_length(net1)
            name1 = os.path.splitext(os.path.basename(network1_path))[0]
            player1 = NetworkPlayer(
                net1,
                num_simulations=num_simulations,
                name=name1,
                history_length=history_length,
            )

        # Load player 2
        if network2_path.lower() == "random":
            player2 = RandomPlayer(name="Random")
            name2 = "Random"
            print(f"Player 2: Random")
        elif network2_path.lower() == "mcts":
            player2 = PureMCTSPlayer(num_simulations=num_simulations, name="PureMCTS")
            name2 = "PureMCTS"
            print(f"Player 2: Pure MCTS ({num_simulations} simulations)")
        else:
            print(f"Loading player 2: {network2_path}")
            net2 = DualHeadNetwork.load(network2_path, device=device)
            hl2 = get_history_length(net2)
            history_length = max(history_length, hl2)
            name2 = os.path.splitext(os.path.basename(network2_path))[0]
            player2 = NetworkPlayer(
                net2,
                num_simulations=num_simulations,
                name=name2,
                history_length=hl2,
            )

        print(f"\nMatch: {name1} vs {name2}")
        print(f"Games: {num_games}")
        print(f"Simulations: {num_simulations}")
        print("=" * 60)
        print()

        arena = Arena(
            num_games=num_games,
            num_simulations=num_simulations,
            max_moves=200,
            history_length=history_length,
        )

        results = {"player1": 0, "player2": 0, "draws": 0}

        for game_num in range(num_games):
            # Alternate colors
            if game_num % 2 == 0:
                white, black = player1, player2
                white_name, black_name = name1, name2
            else:
                white, black = player2, player1
                white_name, black_name = name2, name1

            # Play game
            board = chess.Board()
            white.reset()
            black.reset()
            move_count = 0

            while not board.is_game_over() and move_count < 200:
                current = white if board.turn == chess.WHITE else black
                move = current.select_move(board)
                if move is None:
                    break
                board.push(move)
                move_count += 1

            # Determine result
            if board.is_checkmate():
                winner = "black" if board.turn == chess.WHITE else "white"
                termination = "checkmate"
            elif board.is_stalemate():
                winner = "draw"
                termination = "stalemate"
            elif move_count >= 200:
                winner = "draw"
                termination = "max_moves"
            else:
                winner = "draw"
                termination = "other"

            # Update results
            if winner == "white":
                if white_name == name1:
                    results["player1"] += 1
                else:
                    results["player2"] += 1
                winner_str = white_name
            elif winner == "black":
                if black_name == name1:
                    results["player1"] += 1
                else:
                    results["player2"] += 1
                winner_str = black_name
            else:
                results["draws"] += 1
                winner_str = "Draw"

            # Calculate material advantage for draws
            material_info = ""
            if winner_str == "Draw":
                diff = get_raw_material_diff(board)
                if diff > 0:
                    material_info = f" [W+{diff}]"
                elif diff < 0:
                    material_info = f" [B+{abs(diff)}]"
                else:
                    material_info = " [=]"

            print(
                f"Game {game_num + 1}/{num_games}: "
                f"{white_name} (W) vs {black_name} (B) | "
                f"{move_count} moves | {termination} | {winner_str}{material_info}"
            )

        # Print final results
        print("\n" + "=" * 60)
        print("MATCH RESULTS")
        print("=" * 60)
        print(f"{name1}: {results['player1']} wins")
        print(f"{name2}: {results['player2']} wins")
        print(f"Draws: {results['draws']}")
        print("=" * 60)

        return 0


def run_train(
    config_path: str = None,
    iterations: int = 10,
    output_path: str = None,
    resume: str = None,
    gui: bool = False,
) -> int:
    """
    Run self-play training.

    Args:
        config_path: Path to config file.
        iterations: Number of training iterations.
        output_path: Output path for trained network.
        resume: Checkpoint to resume from.
        gui: Whether to show visual training.

    Returns:
        Exit code.
    """
    if gui:
        # Visual training mode
        try:
            from ui.training_app import TrainingApp

            app = TrainingApp(
                config_path=config_path,
                resume_checkpoint=resume,
                iterations=iterations,
                output_path=output_path or "alphazero_trained.pt",
            )
            app.run()
            return 0
        except ImportError as e:
            print(f"Error importing training UI: {e}")
            return 1
    else:
        # CLI training mode - reuse train.py logic
        from alphazero import DualHeadNetwork, AlphaZeroTrainer
        from config import Config

        # Load config
        if config_path:
            config = Config.load(config_path)
        else:
            config = Config.default()

        cfg = config.training
        cfg.iterations = iterations
        if output_path:
            cfg.checkpoint_path = os.path.dirname(output_path) or "checkpoints"

        print("\n[Configuration]")
        print(f"  Iterations: {cfg.iterations}")
        print(f"  Games per iteration: {cfg.games_per_iteration}")
        print(f"  MCTS simulations: {cfg.num_simulations}")
        print(f"  Batch size: {cfg.batch_size}")
        print(f"  Learning rate: {cfg.learning_rate}")
        print(f"  Checkpoint path: {cfg.checkpoint_path}")

        # Create or load network
        if resume:
            checkpoint_path = resolve_resume(resume, cfg.checkpoint_path)
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"\nLoading network from {checkpoint_path}")
                network = DualHeadNetwork.load(checkpoint_path)
            else:
                print(f"\nCheckpoint not found, creating new network")
                network = DualHeadNetwork()
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

        # Create trainer
        trainer = AlphaZeroTrainer(network, cfg)

        # Training callback
        def callback(data: dict) -> None:
            phase = data.get("phase", "")

            if phase == "self_play":
                games = data.get("games_played", 0)
                total = data.get("total_games", 0)
                print(f"\r  Self-play: {games}/{total} games", end="", flush=True)

            elif phase == "training":
                epoch = data.get("epoch", 0)
                epochs = data.get("epochs", 0)
                loss = data.get("total_loss", 0)
                print(
                    f"\r  Training: epoch {epoch}/{epochs}, loss={loss:.4f}",
                    end="",
                    flush=True,
                )

            elif phase == "iteration_complete":
                iteration = data.get("iteration", 0)
                stats = data.get("stats", {})
                print(f"\n  Iteration {iteration} complete")
                print(f"    Buffer: {stats.get('buffer_size', 0)} positions")
                print(f"    LR: {stats.get('learning_rate', 0):.6f}")

            elif phase == "arena_complete":
                elo = data.get("elo", 1500)
                print(f"  Arena ELO: {elo:.0f}")

        # Run training
        print(f"\nStarting training for {cfg.iterations} iterations")
        print()

        try:
            for i in range(cfg.iterations):
                print(f"=== Iteration {i + 1}/{cfg.iterations} ===")
                trainer.train_iteration(callback)
                print()
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            return 1

        # Save final network if output specified
        if output_path:
            print(f"Saving network to {output_path}")
            network.save(output_path)

        print("Training complete!")
        return 0


def run_pretrain(
    config_path: str = None,
    epochs: int = 10,
    output_path: str = None,
) -> int:
    """
    Run supervised pretraining on master games.

    Args:
        config_path: Path to config file.
        epochs: Number of training epochs.
        output_path: Output path for trained network.

    Returns:
        Exit code.
    """
    from config import Config
    from pretraining.pretrain import pretrain

    # Load config
    if config_path:
        config = Config.load(config_path)
    else:
        config = Config.default()

    cfg = config.pretraining
    cfg.epochs = epochs
    if output_path:
        cfg.output_path = output_path

    # Validate PGN exists
    if not cfg.pgn_path or not os.path.exists(cfg.pgn_path):
        print(f"Error: PGN file not found: {cfg.pgn_path}")
        print("Use --config to specify a config with a valid pgn_path")
        print("Or run: neural_mate_pretrain --pgn your_games.pgn")
        return 1

    try:
        pretrain(cfg, config.network, streaming=True)
        return 0
    except KeyboardInterrupt:
        print("\n\nPretraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NeuralMate2 Chess Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Play against untrained AI
  neural_mate_play

  # Play against trained AI
  neural_mate_play --network alphazero.pt

  # Match: two networks play against each other
  neural_mate_play --network model1.pt model2.pt --games 20

  # Match against random player (baseline test)
  neural_mate_play --network model.pt random --games 20

  # Match against pure MCTS (no neural network)
  neural_mate_play --network model.pt mcts --games 10 --simulations 800

  # Visual match: watch games in UI
  neural_mate_play --network model1.pt model2.pt --games 10 --gui

  # Train a new network
  neural_mate_play --train --iterations 100 --output trained.pt

  # Visual training (see games in real-time)
  neural_mate_play --train --gui --iterations 50

  # Resume self-play training from pretrained network
  neural_mate_play --train --resume pretrained --iterations 100

  # Supervised pretraining
  neural_mate_play --pretrain --epochs 10
        """,
    )

    parser.add_argument(
        "--network",
        "-n",
        type=str,
        nargs="*",
        default=None,
        help="Path to network file(s) (.pt). Use two for match mode. Special: 'random' or 'mcts'.",
    )

    parser.add_argument(
        "--simulations",
        "-s",
        type=int,
        default=200,
        help="Number of MCTS simulations per move (default: 200)",
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device to use: 'cuda', 'cpu', or 'auto' (default: auto)",
    )

    parser.add_argument(
        "--games",
        "-g",
        type=int,
        default=10,
        help="Number of games for match mode (default: 10)",
    )

    parser.add_argument(
        "--train",
        "-t",
        action="store_true",
        help="Start in training mode (self-play)",
    )

    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Start supervised pretraining on master games",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Visual mode (training or match with UI)",
    )

    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in command-line mode instead of GUI",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Training configuration file (JSON)",
    )

    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=10,
        help="Number of training iterations (default: 10)",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=10,
        help="Number of epochs for pretraining (default: 10)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for trained network",
    )

    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        default=None,
        help="Resume training from checkpoint ('latest', 'pretrained', or 'iteration_N')",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print MCTS search tree after each AI move (for debugging/analysis)",
    )

    args = parser.parse_args()

    # Set device
    set_device(args.device)

    # Detect mode
    mode = detect_mode(args)

    if mode == "pretrain":
        return run_pretrain(args.config, args.epochs, args.output)

    elif mode == "train":
        return run_train(
            config_path=args.config,
            iterations=args.iterations,
            output_path=args.output,
            resume=args.resume,
            gui=args.gui,
        )

    elif mode == "match":
        return run_match(
            networks=args.network,
            num_games=args.games,
            num_simulations=args.simulations,
            gui=args.gui,
        )

    else:  # play mode
        network_path = args.network[0] if args.network else None
        if args.cli:
            return run_cli(network_path, args.simulations)
        else:
            return run_gui(network_path, args.simulations, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
