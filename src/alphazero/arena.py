"""
Arena for evaluating chess networks.

Provides players (NetworkPlayer, RandomPlayer) and match infrastructure
for comparing network strength and estimating ELO.
"""

import math
import random
import time
from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field

import chess

from .network import DualHeadNetwork
from .mcts import MCTS
from .spatial_encoding import DEFAULT_HISTORY_LENGTH


@dataclass
class MatchResult:
    """Result of a single match against one opponent."""
    opponent: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    score: float = 0.0  # (wins + 0.5*draws) / total
    avg_time: float = 0.0
    from_iteration: Optional[int] = None  # For vs_best and vs_old

    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws


@dataclass
class ArenaStats:
    """Complete arena evaluation results."""
    vs_random: Optional[MatchResult] = None
    vs_mcts: Optional[MatchResult] = None
    vs_best: Optional[MatchResult] = None
    vs_old: Optional[MatchResult] = None
    vs_pretrained: Optional[MatchResult] = None  # vs initial pretrained model

    elo: float = 1500.0
    best_elo: float = 1500.0

    # Promotion decision
    is_new_best: bool = False
    veto: bool = False
    veto_reason: Optional[str] = None

    # Thresholds used
    random_threshold: float = 0.95
    mcts_threshold: float = 0.70  # Should beat pure MCTS 70%+
    promotion_threshold: float = 0.55
    veto_threshold: float = 0.35


class Player(ABC):
    """Abstract base class for chess players."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Player name for display."""
        pass

    @abstractmethod
    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Select a move for the given position.

        Args:
            board: Current board state.

        Returns:
            Selected move, or None if no legal moves.
        """
        pass

    def reset(self) -> None:
        """Reset player state for a new game."""
        pass


class NetworkPlayer(Player):
    """
    Chess player using MCTS with neural network guidance.
    """

    def __init__(
        self,
        network: DualHeadNetwork,
        num_simulations: int = 200,
        name: str = "Network",
        history_length: int = DEFAULT_HISTORY_LENGTH,
        temperature: float = 0.1,
        batch_size: int = 16,
    ):
        """
        Initialize network player.

        Args:
            network: Neural network for evaluation.
            num_simulations: Number of MCTS simulations per move.
            name: Display name.
            history_length: Position history length.
            temperature: Move selection temperature (0 = deterministic).
            batch_size: Batch size for MCTS GPU inference.
        """
        self._name = name
        self._network = network
        self._num_simulations = num_simulations
        self._mcts = MCTS(
            network,
            num_simulations=num_simulations,
            history_length=history_length,
            batch_size=batch_size,
        )
        self._mcts.temperature = temperature

    @property
    def name(self) -> str:
        return self._name

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Select move using MCTS."""
        return self._mcts.get_best_move(board, add_noise=False)

    def reset(self) -> None:
        """Clear MCTS cache for new game."""
        self._mcts.clear_cache()


class RandomPlayer(Player):
    """
    Chess player that selects random legal moves.

    Useful as a baseline for testing.
    """

    def __init__(self, name: str = "Random"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Select a random legal move."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        return random.choice(legal_moves)


class PureMCTSPlayer(Player):
    """
    Chess player using pure MCTS without neural network.

    Uses random rollouts for evaluation and uniform priors.
    This is the classic MCTS algorithm before AlphaZero.
    """

    def __init__(
        self,
        num_simulations: int = 800,
        max_rollout_depth: int = 50,
        c_puct: float = 1.4,
        name: str = "PureMCTS",
        verbose: bool = False,
    ):
        """
        Initialize pure MCTS player.

        Args:
            num_simulations: Number of MCTS simulations per move.
            max_rollout_depth: Maximum moves in random rollout.
            c_puct: Exploration constant for UCB.
            name: Display name.
            verbose: If True, print search tree after each move.
        """
        self._name = name
        self._num_simulations = num_simulations
        self._max_rollout_depth = max_rollout_depth
        self._c_puct = c_puct
        self._verbose = verbose

        # Store last search results for verbose output
        self._last_visit_counts: Dict[chess.Move, int] = {}
        self._last_total_values: Dict[chess.Move, float] = {}
        self._last_board: Optional[chess.Board] = None

    @property
    def name(self) -> str:
        return self._name

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Select move using pure MCTS with random rollouts."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Initialize visit counts and values for each move
        visit_counts = {move: 0 for move in legal_moves}
        total_values = {move: 0.0 for move in legal_moves}

        # Run simulations
        for _ in range(self._num_simulations):
            # Selection: UCB formula
            total_visits = sum(visit_counts.values()) + 1
            best_move = None
            best_ucb = float("-inf")

            for move in legal_moves:
                if visit_counts[move] == 0:
                    # Prioritize unvisited moves
                    ucb = float("inf")
                else:
                    q = total_values[move] / visit_counts[move]
                    u = self._c_puct * math.sqrt(math.log(total_visits) / visit_counts[move])
                    ucb = q + u

                if ucb > best_ucb:
                    best_ucb = ucb
                    best_move = move

            # Expansion + Rollout
            test_board = board.copy()
            test_board.push(best_move)
            value = self._rollout(test_board)

            # Value is from opponent's perspective, negate it
            value = -value

            # Backpropagation
            visit_counts[best_move] += 1
            total_values[best_move] += value

        # Store for verbose output
        self._last_visit_counts = visit_counts
        self._last_total_values = total_values
        self._last_board = board.copy()

        # Print search tree if verbose
        if self._verbose:
            self.print_search_tree(board)

        # Select move with most visits (use Q-value as tiebreaker)
        def move_score(m):
            visits = visit_counts[m]
            q_value = total_values[m] / max(visits, 1)
            return (visits, q_value)  # Tuple: visits first, then Q-value

        return max(legal_moves, key=move_score)

    def print_search_tree(self, board: chess.Board, top_n: int = 5) -> None:
        """Print the search results."""
        print(self.get_search_tree_string(board, top_n))

    def get_search_tree_string(self, board: chess.Board, top_n: int = 5) -> str:
        """Get a string representation of the search results."""
        lines = []
        lines.append("=" * 60)
        lines.append("Pure MCTS Search (no neural network)")
        lines.append("=" * 60)

        visit_counts = self._last_visit_counts
        total_values = self._last_total_values

        if not visit_counts:
            lines.append("No search data available.")
            lines.append("=" * 60)
            return "\n".join(lines)

        total_visits = sum(visit_counts.values())
        lines.append(f"Total simulations: {total_visits}")
        lines.append(f"Rollout depth: {self._max_rollout_depth}")
        lines.append("")

        # Sort by visit count
        sorted_moves = sorted(
            visit_counts.keys(),
            key=lambda m: visit_counts[m],
            reverse=True,
        )

        # Show top moves
        lines.append(f"Top {min(top_n, len(sorted_moves))} moves:")
        lines.append("-" * 60)
        lines.append(f"{'Move':<8} {'Visits':>8} {'%':>7} {'Q-value':>9} {'Eval':>8}")
        lines.append("-" * 60)

        for move in sorted_moves[:top_n]:
            visits = visit_counts[move]
            pct = (visits / max(total_visits, 1)) * 100
            q_value = total_values[move] / max(visits, 1)

            # Convert Q-value to a more readable eval
            # Q is in [-1, 1], positive = good for current player
            eval_str = f"{q_value:+.3f}"

            lines.append(
                f"{board.san(move):<8} {visits:>8} {pct:>6.1f}% "
                f"{q_value:>+8.3f} {eval_str:>8}"
            )

        lines.append("")
        lines.append("Note: Pure MCTS uses random rollouts (no neural network)")
        lines.append("      Q-value is based on random game simulations")
        lines.append("=" * 60)
        return "\n".join(lines)

    def _rollout(self, board: chess.Board) -> float:
        """
        Random rollout to estimate position value.

        Returns:
            Value from perspective of side to move: 1.0 win, -1.0 loss, 0.0 draw.
        """
        test_board = board.copy()
        depth = 0

        while not test_board.is_game_over() and depth < self._max_rollout_depth:
            moves = list(test_board.legal_moves)
            if not moves:
                break
            test_board.push(random.choice(moves))
            depth += 1

        # Evaluate final position
        if test_board.is_checkmate():
            # Current side is checkmated = loss
            return -1.0
        elif test_board.is_game_over():
            # Draw
            return 0.0
        else:
            # Use material evaluation for incomplete rollouts
            return self._material_eval(test_board)

    def _material_eval(self, board: chess.Board) -> float:
        """
        Simple material-based evaluation.

        Returns value in [-1, 1] from perspective of side to move.
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

        white_material = 0
        black_material = 0

        for piece_type, value in piece_values.items():
            white_material += len(board.pieces(piece_type, chess.WHITE)) * value
            black_material += len(board.pieces(piece_type, chess.BLACK)) * value

        # Normalize to [-1, 1] range (39 = max material difference)
        diff = (white_material - black_material) / 39.0

        # Return from perspective of side to move
        if board.turn == chess.WHITE:
            return max(-1.0, min(1.0, diff))
        else:
            return max(-1.0, min(1.0, -diff))


@dataclass
class GameResult:
    """Result of a single game."""

    winner: Optional[chess.Color]  # WHITE, BLACK, or None for draw
    termination: str  # "checkmate", "stalemate", "fifty_moves", etc.
    moves: int
    white_player: str
    black_player: str
    pgn: str = ""


class Arena:
    """
    Arena for playing matches between chess players.

    Supports network vs network, network vs random, and tracks ELO.
    """

    def __init__(
        self,
        num_games: int = 20,
        num_simulations: int = 100,
        max_moves: int = 200,
        history_length: int = DEFAULT_HISTORY_LENGTH,
    ):
        """
        Initialize arena.

        Args:
            num_games: Number of games per match.
            num_simulations: MCTS simulations for network players.
            max_moves: Maximum moves per game before draw.
            history_length: Position history length.
        """
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.max_moves = max_moves
        self.history_length = history_length

        # ELO tracking
        self._elo = 1500.0
        self._games_played = 0

    def get_elo(self) -> float:
        """Get current ELO estimate."""
        return self._elo

    def play_match(
        self,
        player1: Player,
        player2: Player,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """
        Play a match between two players.

        Args:
            player1: First player.
            player2: Second player.
            callback: Optional callback for game updates.

        Returns:
            Match results dictionary.
        """
        results = {
            "player1": player1.name,
            "player2": player2.name,
            "player1_wins": 0,
            "player2_wins": 0,
            "draws": 0,
            "games": [],
            "total_time": 0.0,
            "avg_time": 0.0,
        }

        total_time = 0.0

        for game_idx in range(self.num_games):
            # Alternate colors
            if game_idx % 2 == 0:
                white, black = player1, player2
            else:
                white, black = player2, player1

            # Play game
            game_start = time.time()
            result = self.play_game(white, black, callback)
            game_time = time.time() - game_start
            total_time += game_time

            results["games"].append(result)

            # Update counts
            if result.winner == chess.WHITE:
                if white == player1:
                    results["player1_wins"] += 1
                else:
                    results["player2_wins"] += 1
            elif result.winner == chess.BLACK:
                if black == player1:
                    results["player1_wins"] += 1
                else:
                    results["player2_wins"] += 1
            else:
                results["draws"] += 1

            if callback:
                callback(
                    {
                        "phase": "game_complete",
                        "game": game_idx + 1,
                        "total_games": self.num_games,
                        "result": result,
                    }
                )

        # Update ELO based on results
        self._update_elo(results)

        # Add timing stats
        results["total_time"] = total_time
        results["avg_time"] = total_time / max(1, self.num_games)

        return results

    def play_game(
        self,
        white: Player,
        black: Player,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> GameResult:
        """
        Play a single game.

        Args:
            white: White player.
            black: Black player.
            callback: Optional callback for move updates.

        Returns:
            GameResult with game outcome.
        """
        board = chess.Board()
        white.reset()
        black.reset()

        move_count = 0
        moves_san = []

        while not board.is_game_over() and move_count < self.max_moves:
            # Get current player
            player = white if board.turn == chess.WHITE else black

            # Get move
            move = player.select_move(board)

            if move is None:
                break

            # Record move
            moves_san.append(board.san(move))

            # Make move
            board.push(move)
            move_count += 1

            if callback:
                callback(
                    {
                        "phase": "move",
                        "fen": board.fen(),
                        "move": move.uci(),
                        "san": moves_san[-1],
                        "move_number": move_count,
                    }
                )

            # Check for adjudication
            adjudicated, winner, reason = self._check_adjudication(board, move_count)
            if adjudicated:
                return GameResult(
                    winner=winner,
                    termination=reason,
                    moves=move_count,
                    white_player=white.name,
                    black_player=black.name,
                    pgn=self._moves_to_pgn(moves_san, white.name, black.name),
                )

        # Determine result
        if board.is_checkmate():
            winner = chess.BLACK if board.turn == chess.WHITE else chess.WHITE
            termination = "checkmate"
        elif board.is_stalemate():
            winner = None
            termination = "stalemate"
        elif board.is_insufficient_material():
            winner = None
            termination = "insufficient_material"
        elif board.can_claim_fifty_moves():
            winner = None
            termination = "fifty_moves"
        elif board.can_claim_threefold_repetition():
            winner = None
            termination = "threefold_repetition"
        elif move_count >= self.max_moves:
            winner = None
            termination = "max_moves"
        else:
            winner = None
            termination = "unknown"

        return GameResult(
            winner=winner,
            termination=termination,
            moves=move_count,
            white_player=white.name,
            black_player=black.name,
            pgn=self._moves_to_pgn(moves_san, white.name, black.name),
        )

    def _check_adjudication(
        self,
        board: chess.Board,
        move_count: int,
    ) -> tuple[bool, Optional[chess.Color], str]:
        """
        Check if game should be adjudicated.

        Args:
            board: Current board state.
            move_count: Number of moves played.

        Returns:
            Tuple of (adjudicated, winner, reason).
        """
        # Material-based adjudication
        if move_count >= 40:  # Only after move 40
            material_diff = self._get_material_diff(board)

            # Decisive material advantage for 10+ moves
            if abs(material_diff) >= 15:
                if material_diff > 0:
                    return True, chess.WHITE, "material"
                else:
                    return True, chess.BLACK, "material"

        return False, None, ""

    def _get_material_diff(self, board: chess.Board) -> int:
        """Calculate material difference (positive = white advantage)."""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

        diff = 0
        for piece_type, value in piece_values.items():
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            diff += (white_count - black_count) * value

        return diff

    def _update_elo(self, results: dict) -> None:
        """Update ELO based on match results."""
        total = results["player1_wins"] + results["player2_wins"] + results["draws"]
        if total == 0:
            return

        # Calculate score (wins = 1, draws = 0.5)
        score = (results["player1_wins"] + 0.5 * results["draws"]) / total

        # Update ELO (K-factor = 32)
        expected = 0.5  # Assume equal strength initially
        self._elo += 32 * (score - expected) * total / self.num_games
        self._games_played += total

    def _moves_to_pgn(
        self,
        moves: list[str],
        white: str,
        black: str,
    ) -> str:
        """Convert move list to PGN string."""
        pgn_lines = [
            f'[White "{white}"]',
            f'[Black "{black}"]',
            "",
        ]

        # Format moves
        move_text = []
        for i, san in enumerate(moves):
            if i % 2 == 0:
                move_text.append(f"{i // 2 + 1}. {san}")
            else:
                move_text[-1] += f" {san}"

        pgn_lines.append(" ".join(move_text))
        return "\n".join(pgn_lines)

    def get_best_elo(self) -> float:
        """Get best ELO achieved."""
        return getattr(self, "_best_elo", self._elo)

    def run_full_evaluation(
        self,
        current_network: DualHeadNetwork,
        best_network: Optional[DualHeadNetwork] = None,
        old_network: Optional[DualHeadNetwork] = None,
        pretrained_network: Optional[DualHeadNetwork] = None,
        best_iteration: Optional[int] = None,
        old_iteration: Optional[int] = None,
        callback: Optional[Callable[[dict], None]] = None,
        promotion_threshold: float = 0.55,
        veto_threshold: float = 0.35,
    ) -> ArenaStats:
        """
        Run complete arena evaluation against all opponents.

        Args:
            current_network: Network being evaluated.
            best_network: Current best network (optional).
            old_network: Old checkpoint network (optional).
            pretrained_network: Initial pretrained network for progress tracking (optional).
            best_iteration: Iteration of best network.
            old_iteration: Iteration of old network.
            callback: Progress callback.
            promotion_threshold: Score needed to beat best (default 0.55).
            veto_threshold: Score needed against old to avoid veto (default 0.35).

        Returns:
            ArenaStats with all results.
        """
        stats = ArenaStats(
            promotion_threshold=promotion_threshold,
            veto_threshold=veto_threshold,
        )

        # Create current player
        current_player = NetworkPlayer(
            current_network,
            num_simulations=self.num_simulations,
            name="Current",
            history_length=self.history_length,
        )

        # 1. vs Random - Sanity check
        if callback:
            callback({"phase": "arena_match", "opponent": "Random", "status": "starting"})

        random_player = RandomPlayer(name="Random")
        random_results = self.play_match(current_player, random_player, callback)

        stats.vs_random = MatchResult(
            opponent="Random",
            wins=random_results["player1_wins"],
            losses=random_results["player2_wins"],
            draws=random_results["draws"],
            score=(random_results["player1_wins"] + 0.5 * random_results["draws"]) / max(1, self.num_games),
            avg_time=random_results.get("avg_time", 0.0),
        )

        # 2. vs Pure MCTS - Intermediate strength check
        if callback:
            callback({"phase": "arena_match", "opponent": "PureMCTS", "status": "starting"})

        mcts_player = PureMCTSPlayer(
            num_simulations=self.num_simulations,
            name="PureMCTS",
        )
        mcts_results = self.play_match(current_player, mcts_player, callback)

        stats.vs_mcts = MatchResult(
            opponent="PureMCTS",
            wins=mcts_results["player1_wins"],
            losses=mcts_results["player2_wins"],
            draws=mcts_results["draws"],
            score=(mcts_results["player1_wins"] + 0.5 * mcts_results["draws"]) / max(1, self.num_games),
            avg_time=mcts_results.get("avg_time", 0.0),
        )

        # 3. vs Best - Promotion check
        if best_network is not None:
            if callback:
                callback({"phase": "arena_match", "opponent": f"Best #{best_iteration}", "status": "starting"})

            best_player = NetworkPlayer(
                best_network,
                num_simulations=self.num_simulations,
                name=f"Best #{best_iteration}",
                history_length=self.history_length,
            )
            best_results = self.play_match(current_player, best_player, callback)

            stats.vs_best = MatchResult(
                opponent=f"Best #{best_iteration}",
                wins=best_results["player1_wins"],
                losses=best_results["player2_wins"],
                draws=best_results["draws"],
                score=(best_results["player1_wins"] + 0.5 * best_results["draws"]) / max(1, self.num_games),
                avg_time=best_results.get("avg_time", 0.0),
                from_iteration=best_iteration,
            )

            # Check for promotion
            if stats.vs_best.score >= promotion_threshold:
                stats.is_new_best = True

        else:
            # No best model yet - first iteration is automatically best
            stats.is_new_best = True

        # 4. vs Old - Catastrophic forgetting check
        if old_network is not None:
            if callback:
                callback({"phase": "arena_match", "opponent": f"Old #{old_iteration}", "status": "starting"})

            old_player = NetworkPlayer(
                old_network,
                num_simulations=self.num_simulations,
                name=f"Old #{old_iteration}",
                history_length=self.history_length,
            )
            old_results = self.play_match(current_player, old_player, callback)

            stats.vs_old = MatchResult(
                opponent=f"Old #{old_iteration}",
                wins=old_results["player1_wins"],
                losses=old_results["player2_wins"],
                draws=old_results["draws"],
                score=(old_results["player1_wins"] + 0.5 * old_results["draws"]) / max(1, self.num_games),
                avg_time=old_results.get("avg_time", 0.0),
                from_iteration=old_iteration,
            )

            # Check for veto (catastrophic forgetting)
            if stats.vs_old.score < veto_threshold:
                stats.veto = True
                stats.veto_reason = f"Lost to old model #{old_iteration} ({stats.vs_old.score*100:.0f}% < {veto_threshold*100:.0f}%)"
                stats.is_new_best = False  # Block promotion

        # 5. vs Pretrained - Progress tracking against initial model
        if pretrained_network is not None:
            if callback:
                callback({"phase": "arena_match", "opponent": "Pretrained", "status": "starting"})

            pretrained_player = NetworkPlayer(
                pretrained_network,
                num_simulations=self.num_simulations,
                name="Pretrained",
                history_length=self.history_length,
            )
            pretrained_results = self.play_match(current_player, pretrained_player, callback)

            stats.vs_pretrained = MatchResult(
                opponent="Pretrained",
                wins=pretrained_results["player1_wins"],
                losses=pretrained_results["player2_wins"],
                draws=pretrained_results["draws"],
                score=(pretrained_results["player1_wins"] + 0.5 * pretrained_results["draws"]) / max(1, self.num_games),
                avg_time=pretrained_results.get("avg_time", 0.0),
            )

        # Update ELO based on vs_best results (or vs_random if no best)
        if stats.vs_best is not None:
            self._update_elo_from_match(stats.vs_best)
        else:
            self._update_elo_from_match(stats.vs_random)

        stats.elo = self._elo
        stats.best_elo = self.get_best_elo()

        # Update best ELO if this is new best
        if stats.is_new_best and not stats.veto:
            self._best_elo = max(getattr(self, "_best_elo", 1500.0), self._elo)
            stats.best_elo = self._best_elo

        return stats

    def _update_elo_from_match(self, result: MatchResult) -> None:
        """Update ELO based on match result."""
        total = result.total_games
        if total == 0:
            return

        score = result.score
        expected = 0.5
        self._elo += 32 * (score - expected) * total / self.num_games
        self._games_played += total


def compare_networks(
    network1_path: str,
    network2_path: str,
    num_games: int = 20,
    num_simulations: int = 100,
) -> dict:
    """
    Compare two networks by playing a match.

    Args:
        network1_path: Path to first network.
        network2_path: Path to second network.
        num_games: Number of games to play.
        num_simulations: MCTS simulations per move.

    Returns:
        Match results dictionary.
    """
    from .network import DualHeadNetwork

    net1 = DualHeadNetwork.load(network1_path)
    net2 = DualHeadNetwork.load(network2_path)

    player1 = NetworkPlayer(net1, num_simulations, name="Network1")
    player2 = NetworkPlayer(net2, num_simulations, name="Network2")

    arena = Arena(num_games, num_simulations)
    return arena.play_match(player1, player2)
