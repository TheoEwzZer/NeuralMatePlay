"""
Monte Carlo Tree Search with neural network guidance.

Features:
- PUCT selection formula (Polynomial Upper Confidence Trees)
- Virtual losses for parallel search
- Transposition table for position caching
- Dirichlet noise at root for exploration
- Temperature-based move selection
- Batched neural network inference
"""

import math
from typing import Optional
from dataclasses import dataclass, field

import chess
import numpy as np

from .network import DualHeadNetwork
from .move_encoding import (
    encode_move_from_perspective,
    decode_move_from_perspective,
    MOVE_ENCODING_SIZE,
)
from .spatial_encoding import (
    encode_board_with_history,
    PositionHistory,
    DEFAULT_HISTORY_LENGTH,
)


@dataclass
class MCTSNode:
    """
    Node in the MCTS tree.

    Stores visit counts, value estimates, and child nodes.
    """

    prior: float = 0.0  # P(s, a) - prior probability from policy network
    visit_count: int = 0  # N(s, a) - number of visits
    total_value: float = 0.0  # W(s, a) - total value from all visits
    virtual_loss: int = 0  # Virtual loss for parallel search
    children: dict = field(default_factory=dict)  # move -> MCTSNode
    is_expanded: bool = False

    @property
    def q_value(self) -> float:
        """Q(s, a) = W(s, a) / N(s, a) - mean value."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def effective_visits(self) -> int:
        """Visits including virtual losses."""
        return self.visit_count + self.virtual_loss


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.

    Uses PUCT formula for selection:
    U(s, a) = c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
    select = argmax(Q(s, a) + U(s, a))
    """

    def __init__(
        self,
        network: DualHeadNetwork,
        num_simulations: int = 800,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        batch_size: int = 8,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        fpu_reduction: float = 0.25,  # First Play Urgency reduction
        temperature: float = 0.0,  # 0 = deterministic, >0 = sample from policy
    ):
        """
        Initialize MCTS.

        Args:
            network: Neural network for policy and value prediction.
            num_simulations: Number of simulations per search.
            c_puct: Exploration constant.
            dirichlet_alpha: Dirichlet noise alpha (0.3 for chess).
            dirichlet_epsilon: Weight of Dirichlet noise at root.
            batch_size: Batch size for neural network inference.
            history_length: Number of past positions to track.
            fpu_reduction: FPU reduction for unexplored moves.
            temperature: Move selection temperature (0 = best move, >0 = sample).
        """
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        self.history_length = history_length
        self.fpu_reduction = fpu_reduction
        self.temperature = temperature

        # Random number generator
        self._rng = np.random.default_rng(42)

        # Transposition table (position hash -> node)
        self._transposition_table: dict[str, MCTSNode] = {}

        # Evaluation cache (position hash -> (policy, value))
        # Avoids re-evaluating same positions reached via different paths
        self._eval_cache: dict[str, tuple[np.ndarray, float]] = {}
        self._max_eval_cache_size = 50000  # ~400MB max (8KB per entry)

        # Position history tracker
        self._history = PositionHistory(history_length)

    def search(
        self,
        board: chess.Board,
        add_noise: bool = False,
    ) -> np.ndarray:
        """
        Run MCTS search and return visit count distribution.

        Args:
            board: Current board position.
            add_noise: Whether to add Dirichlet noise at root.

        Returns:
            Policy array of shape (MOVE_ENCODING_SIZE,) with visit proportions.
        """
        # Initialize history with current position
        self._history.reset(board)

        # Get or create root node
        root = self._get_or_create_node(board)

        # Expand root if needed
        if not root.is_expanded:
            self._expand_node(root, board)

        # Add Dirichlet noise at root for exploration
        if add_noise:
            self._add_dirichlet_noise(root, board)

        # Run simulations with batched neural network evaluation
        self._run_batched_simulations(board, root, self.num_simulations)

        # Build policy from visit counts
        policy = np.zeros(MOVE_ENCODING_SIZE, dtype=np.float32)
        flip = board.turn == chess.BLACK

        total_visits = sum(child.visit_count for child in root.children.values())

        # Check for winning moves (Q >= 0.9999 from our perspective = Q <= -0.9999 stored)
        # Prioritize guaranteed wins (like checkmate) even if they have fewer visits
        winning_moves = [
            (move, child)
            for move, child in root.children.items()
            if child.visit_count > 0 and child.q_value <= -0.9999
        ]

        if winning_moves:
            # Found winning move(s): select the one with best Q-value (most negative)
            best_move, _ = min(winning_moves, key=lambda x: x[1].q_value)
            idx = encode_move_from_perspective(best_move, flip)
            if idx is not None:
                policy[idx] = 1.0
        elif total_visits == 0:
            # No simulations run: use prior probabilities from neural network
            for move, child in root.children.items():
                idx = encode_move_from_perspective(move, flip)
                if idx is not None:
                    policy[idx] = child.prior
        else:
            for move, child in root.children.items():
                if child.visit_count > 0:
                    idx = encode_move_from_perspective(move, flip)
                    if idx is not None:
                        if self.temperature > 0:
                            policy[idx] = child.visit_count / total_visits
                        else:
                            # Temperature 0: deterministic selection by visits, then Q-value
                            max_visits = max(c.visit_count for c in root.children.values())
                            best_q = min(
                                c.q_value for c in root.children.values()
                                if c.visit_count == max_visits
                            )
                            policy[idx] = (
                                1.0
                                if child.visit_count == max_visits and child.q_value == best_q
                                else 0.0
                            )

        # Apply temperature
        if self.temperature > 0 and self.temperature != 1.0:
            # Raise to power of 1/temperature
            policy = np.power(policy + 1e-10, 1.0 / self.temperature)
            policy /= policy.sum() + 1e-10

        return policy

    def get_best_move(
        self,
        board: chess.Board,
        add_noise: bool = False,
    ) -> Optional[chess.Move]:
        """
        Run MCTS and return the best move.

        Args:
            board: Current board position.
            add_noise: Whether to add exploration noise.

        Returns:
            Best move according to MCTS, or None if no legal moves.
        """
        if not list(board.legal_moves):
            return None

        policy = self.search(board, add_noise)

        # Sample or select best move based on temperature
        if self.temperature > 0:
            # Sample from policy
            policy = policy / (policy.sum() + 1e-10)
            try:
                idx = self._rng.choice(len(policy), p=policy)
            except ValueError:
                # Fallback to argmax if sampling fails
                idx = np.argmax(policy)
        else:
            idx = np.argmax(policy)

        # Decode move
        flip = board.turn == chess.BLACK
        move = decode_move_from_perspective(idx, board, flip)

        # Fallback to most visited child if decode fails
        if move is None:
            root = self._get_or_create_node(board)
            if root.children:
                # Check for winning moves first
                winning = [
                    (m, c) for m, c in root.children.items()
                    if c.visit_count > 0 and c.q_value <= -0.9999
                ]
                if winning:
                    move = min(winning, key=lambda x: x[1].q_value)[0]
                else:
                    move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]

        return move

    def _simulate(self, board: chess.Board, node: MCTSNode) -> float:
        """
        Run one MCTS simulation.

        Args:
            board: Current board state (will be modified).
            node: Current tree node.

        Returns:
            Value of the terminal/leaf state.
        """
        path = [(None, node)]  # (move, node) pairs

        # Selection: traverse tree to leaf
        while node.is_expanded and node.children:
            # Check for terminal state
            if board.is_game_over():
                break

            # Select best child using PUCT
            move, child = self._select_child(node, board)

            # Safety check: verify move is legal (transposition table can cause issues)
            if move is None or move not in board.legal_moves:
                break

            # Apply virtual loss
            child.virtual_loss += 1

            # Make move
            board.push(move)
            path.append((move, child))
            node = child

        # Get value
        if board.is_game_over():
            # Terminal state
            result = board.result()
            if result == "1-0":
                value = 1.0
            elif result == "0-1":
                value = -1.0
            else:
                value = 0.0
            # Adjust for perspective
            if board.turn == chess.BLACK:
                value = -value
        else:
            # Expand if not expanded
            if not node.is_expanded:
                self._expand_node(node, board)

            # Use value already computed during expansion
            # _expand_node stores value in node.total_value with visit_count=1
            # So q_value = total_value / visit_count = value
            value = node.q_value if node.visit_count > 0 else 0.0

        # Backpropagation
        for i, (move, path_node) in enumerate(reversed(path)):
            # Remove virtual loss
            if move is not None:
                path_node.virtual_loss = max(0, path_node.virtual_loss - 1)

            # Update statistics (value is from current player's perspective at leaf)
            # Need to flip for each level
            if i % 2 == 0:
                path_node.visit_count += 1
                path_node.total_value += value
            else:
                path_node.visit_count += 1
                path_node.total_value -= value

        return value

    # =========================================================================
    # Batched MCTS methods (for GPU efficiency)
    # =========================================================================

    def _collect_leaf(
        self,
        board: chess.Board,
        root: MCTSNode,
    ) -> tuple[list[tuple[chess.Move | None, MCTSNode]], chess.Board, bool]:
        """
        Traverse tree to a leaf node, applying virtual losses.

        Returns:
            Tuple of (path, leaf_board, is_terminal):
            - path: list of (move, node) pairs from root to leaf
            - leaf_board: board state at the leaf
            - is_terminal: True if leaf is a terminal game state
        """
        node = root
        path = [(None, node)]

        # Selection: traverse tree to leaf
        while node.is_expanded and node.children:
            # Check for terminal state
            if board.is_game_over():
                return path, board, True

            # Select best child using PUCT
            move, child = self._select_child(node, board)

            # Safety check: verify move is legal (transposition table can cause issues)
            if move is None or move not in board.legal_moves:
                # Node has stale children from transposition, treat as leaf
                break

            # Apply virtual loss (prevents other traversals from selecting same path)
            child.virtual_loss += 1

            # Make move
            board.push(move)
            path.append((move, child))
            node = child

        # Check if terminal at leaf
        is_terminal = board.is_game_over()

        return path, board, is_terminal

    def _backpropagate(
        self,
        path: list[tuple[chess.Move | None, MCTSNode]],
        value: float,
    ) -> None:
        """
        Backpropagate value through the path, removing virtual losses.

        Args:
            path: List of (move, node) pairs from root to leaf.
            value: Value from the leaf node's perspective.
        """
        for i, (move, path_node) in enumerate(reversed(path)):
            # Remove virtual loss
            if move is not None:
                path_node.virtual_loss = max(0, path_node.virtual_loss - 1)

            # Update statistics (flip value for each level)
            if i % 2 == 0:
                path_node.visit_count += 1
                path_node.total_value += value
            else:
                path_node.visit_count += 1
                path_node.total_value -= value

    def _expand_node_with_eval(
        self,
        node: MCTSNode,
        board: chess.Board,
        policy: np.ndarray,
        value: float,
    ) -> None:
        """
        Expand node using pre-computed policy and value from batch evaluation.

        Args:
            node: Node to expand.
            board: Board state at this node.
            policy: Policy array from neural network.
            value: Value from neural network.
        """
        if node.is_expanded:
            return

        # Store initial value
        node.visit_count = 1
        node.total_value = value

        # Create children for legal moves
        flip = board.turn == chess.BLACK
        legal_moves = list(board.legal_moves)

        # Mask and renormalize policy
        policy_sum = 0.0
        move_priors = []

        for move in legal_moves:
            idx = encode_move_from_perspective(move, flip)
            if idx is not None and idx < len(policy):
                prior = policy[idx]
            else:
                prior = 1e-6
            move_priors.append((move, prior))
            policy_sum += prior

        # Normalize
        for move, prior in move_priors:
            normalized_prior = prior / (policy_sum + 1e-10)
            node.children[move] = MCTSNode(prior=normalized_prior)

        node.is_expanded = True

    def _run_batched_simulations(
        self,
        board: chess.Board,
        root: MCTSNode,
        num_simulations: int,
    ) -> None:
        """
        Run simulations with batched neural network evaluation.

        Uses virtual loss to enable parallel tree traversals, then batches
        neural network calls for GPU efficiency.

        Args:
            board: Root board position.
            root: Root node.
            num_simulations: Number of simulations to run.
        """
        simulations_done = 0

        while simulations_done < num_simulations:
            # Determine batch size for this iteration
            remaining = num_simulations - simulations_done
            current_batch_size = min(self.batch_size, remaining)

            # Collect leaves (apply virtual loss during traversal)
            pending_leaves = []  # List of (path, board, node, is_terminal)

            for _ in range(current_batch_size):
                board_copy = board.copy()
                path, leaf_board, is_terminal = self._collect_leaf(board_copy, root)
                leaf_node = path[-1][1]
                pending_leaves.append((path, leaf_board, leaf_node, is_terminal))

            # Separate terminal, already expanded, and new leaves
            terminal_leaves = []
            expanded_leaves = []
            new_leaves = []

            for path, leaf_board, leaf_node, is_terminal in pending_leaves:
                if is_terminal:
                    terminal_leaves.append((path, leaf_board, leaf_node))
                elif leaf_node.is_expanded:
                    # Already expanded (race condition with virtual loss)
                    expanded_leaves.append((path, leaf_board, leaf_node))
                else:
                    new_leaves.append((path, leaf_board, leaf_node))

            # Handle terminal leaves
            for path, leaf_board, leaf_node in terminal_leaves:
                result = leaf_board.result()
                if result == "1-0":
                    value = 1.0
                elif result == "0-1":
                    value = -1.0
                else:
                    value = 0.0
                # Adjust for perspective
                if leaf_board.turn == chess.BLACK:
                    value = -value

                self._backpropagate(path, value)
                simulations_done += 1

            # Handle already expanded leaves (use stored value)
            for path, leaf_board, leaf_node in expanded_leaves:
                value = leaf_node.q_value
                self._backpropagate(path, value)
                simulations_done += 1

            # Batch evaluate new leaves (with cache lookup)
            if new_leaves:
                # Separate cached and uncached positions
                cached_evals = []  # (index, policy, value)
                to_evaluate = []  # (index, state, leaf_board)

                for i, (path, leaf_board, leaf_node) in enumerate(new_leaves):
                    cached = self._get_cached_eval(leaf_board)
                    if cached is not None:
                        cached_evals.append((i, cached[0], cached[1]))
                    else:
                        self._history.clear()
                        self._history.push(leaf_board)
                        state = self._history.encode(from_perspective=True)
                        to_evaluate.append((i, state, leaf_board))

                # Batch evaluate uncached positions
                eval_results = {}  # index -> (policy, value)

                if to_evaluate:
                    states_array = np.array(
                        [s for _, s, _ in to_evaluate], dtype=np.float32
                    )
                    policies, values = self.network.predict_batch(states_array)

                    for j, (i, _, leaf_board) in enumerate(to_evaluate):
                        policy = policies[j]
                        value = float(values[j])
                        eval_results[i] = (policy, value)
                        # Cache the result
                        self._cache_eval(leaf_board, policy, value)

                # Add cached results
                for i, policy, value in cached_evals:
                    eval_results[i] = (policy, value)

                # Expand nodes and backpropagate
                for i, (path, leaf_board, leaf_node) in enumerate(new_leaves):
                    policy, value = eval_results[i]

                    # Expand the node with pre-computed values
                    self._expand_node_with_eval(leaf_node, leaf_board, policy, value)

                    # Backpropagate
                    self._backpropagate(path, value)
                    simulations_done += 1

    def _select_child(
        self, node: MCTSNode, board: chess.Board
    ) -> tuple[chess.Move, MCTSNode]:
        """
        Select child node using PUCT formula.

        Args:
            node: Parent node.
            board: Current board state.

        Returns:
            Tuple of (selected_move, selected_child).
        """
        sqrt_total = math.sqrt(max(1, node.visit_count))

        best_score = float("-inf")
        best_move = None
        best_child = None

        # FPU (First Play Urgency)
        fpu_value = node.q_value - self.fpu_reduction

        for move, child in node.children.items():
            if child.effective_visits == 0:
                q = fpu_value
            else:
                # Negate Q-value: child stores value from child's perspective,
                # but parent needs value from parent's perspective
                q = -child.q_value

            # PUCT formula
            u = self.c_puct * child.prior * sqrt_total / (1 + child.effective_visits)
            score = q + u

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _expand_node(self, node: MCTSNode, board: chess.Board) -> None:
        """
        Expand node by adding children for all legal moves.

        Args:
            node: Node to expand.
            board: Current board state.
        """
        if node.is_expanded:
            return

        # Check cache first
        cached = self._get_cached_eval(board)
        if cached is not None:
            policy, value = cached
        else:
            # Get policy from neural network
            self._history.clear()
            self._history.push(board)
            state = self._history.encode(from_perspective=True)
            policy, value = self.network.predict_single(state)
            # Cache the result
            self._cache_eval(board, policy, value)

        # Store initial value
        node.visit_count = 1
        node.total_value = value

        # Create children for legal moves
        flip = board.turn == chess.BLACK
        legal_moves = list(board.legal_moves)

        # Mask and renormalize policy
        policy_sum = 0.0
        move_priors = []

        for move in legal_moves:
            idx = encode_move_from_perspective(move, flip)
            if idx is not None and idx < len(policy):
                prior = policy[idx]
            else:
                prior = 1e-6
            move_priors.append((move, prior))
            policy_sum += prior

        # Normalize
        for move, prior in move_priors:
            normalized_prior = prior / (policy_sum + 1e-10)
            node.children[move] = MCTSNode(prior=normalized_prior)

        node.is_expanded = True

    def _add_dirichlet_noise(self, node: MCTSNode, board: chess.Board) -> None:
        """
        Add Dirichlet noise to root node priors for exploration.

        Args:
            node: Root node.
            board: Current board state.
        """
        if not node.children:
            return

        # Generate Dirichlet noise
        num_moves = len(node.children)
        noise = self._rng.dirichlet([self.dirichlet_alpha] * num_moves)

        # Apply noise to priors
        for i, (move, child) in enumerate(node.children.items()):
            child.prior = (
                1 - self.dirichlet_epsilon
            ) * child.prior + self.dirichlet_epsilon * noise[i]

    def _get_or_create_node(self, board: chess.Board) -> MCTSNode:
        """
        Get node from transposition table or create new one.

        Args:
            board: Board position.

        Returns:
            MCTSNode for this position.
        """
        # Use board FEN + turn as key (different turns = different nodes)
        # This prevents transposition bugs where same position with different
        # side to move would share children (causing illegal moves)
        turn = "w" if board.turn == chess.WHITE else "b"
        key = f"{board.board_fen()} {turn}"

        if key not in self._transposition_table:
            self._transposition_table[key] = MCTSNode()

        return self._transposition_table[key]

    def clear_cache(self, reset_rng: bool = False) -> None:
        """
        Clear transposition table and evaluation cache.

        Args:
            reset_rng: If True, also reset the random generator for deterministic replay.
        """
        self._transposition_table.clear()
        self._eval_cache.clear()
        if reset_rng:
            self._rng = np.random.default_rng(42)

    def _get_cached_eval(
        self,
        board: chess.Board,
    ) -> tuple[np.ndarray, float] | None:
        """
        Get cached evaluation for a position.

        Args:
            board: Board position.

        Returns:
            Tuple of (policy, value) if cached, None otherwise.
        """
        # Use FEN with side-to-move for unique key
        key = board.board_fen() + ("w" if board.turn == chess.WHITE else "b")
        return self._eval_cache.get(key)

    def _cache_eval(
        self,
        board: chess.Board,
        policy: np.ndarray,
        value: float,
    ) -> None:
        """
        Cache evaluation for a position.

        Args:
            board: Board position.
            policy: Policy array.
            value: Value estimate.
        """
        # Clear cache if too large to avoid memory issues
        if len(self._eval_cache) >= self._max_eval_cache_size:
            self._eval_cache.clear()

        key = board.board_fen() + ("w" if board.turn == chess.WHITE else "b")
        self._eval_cache[key] = (policy.copy(), value)

    def get_pv(self, board: chess.Board, depth: int = 10) -> list[chess.Move]:
        """
        Get principal variation (most visited path).

        Args:
            board: Starting position.
            depth: Maximum depth.

        Returns:
            List of moves in the PV.
        """
        pv = []
        current_board = board.copy()

        for _ in range(depth):
            node = self._get_or_create_node(current_board)

            if not node.children:
                break

            # Check for winning moves first (Q <= -0.9999 = guaranteed win)
            winning = [
                (m, c) for m, c in node.children.items()
                if c.visit_count > 0 and c.q_value <= -0.9999
            ]
            if winning:
                best_move = min(winning, key=lambda x: x[1].q_value)[0]
            else:
                # Get most visited move (use Q-value as tiebreaker: lower = better for us)
                best_move = max(
                    node.children.items(), key=lambda x: (x[1].visit_count, -x[1].q_value)
                )[0]
            pv.append(best_move)

            current_board.push(best_move)

        return pv

    def get_visit_counts(self, board: chess.Board) -> dict[chess.Move, int]:
        """
        Get visit counts for all moves at root.

        Args:
            board: Current position.

        Returns:
            Dictionary mapping moves to visit counts.
        """
        node = self._get_or_create_node(board)
        return {move: child.visit_count for move, child in node.children.items()}

    def get_root_statistics(self, board: chess.Board) -> list[dict]:
        """
        Get detailed statistics for all moves at root.

        Used by the GUI MCTS panel to display search results.

        Args:
            board: Current position.

        Returns:
            List of dicts with keys: move, visits, q_value, prior
            Sorted by visits descending.
        """
        node = self._get_or_create_node(board)
        stats = []

        for move, child in node.children.items():
            stats.append({
                "move": move,
                "visits": child.visit_count,
                # Negate Q-value: stored Q is from opponent's perspective,
                # we want value from current player's perspective
                "q_value": -child.q_value if child.visit_count > 0 else 0.0,
                "prior": child.prior,
                "total_value": child.total_value,
            })

        # Sort by visits descending
        stats.sort(key=lambda x: x["visits"], reverse=True)
        return stats

    def get_root_value(self, board: chess.Board) -> float:
        """
        Get the value estimate at the root node.

        Args:
            board: Current position.

        Returns:
            Value from current player's perspective (-1 to +1).
        """
        node = self._get_or_create_node(board)
        if node.visit_count > 0:
            return node.q_value
        return 0.0

    def get_search_tree_string(
        self,
        board: chess.Board,
        top_n: int = 5,
        max_depth: int = 10,
    ) -> str:
        """
        Get a string representation of the search tree for verbose output.

        Args:
            board: Current position.
            top_n: Number of top moves to show at each level.
            max_depth: Maximum depth to display.

        Returns:
            Formatted string showing the search tree.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("MCTS Search Tree")
        lines.append("=" * 60)

        root = self._get_or_create_node(board)
        total_visits = sum(child.visit_count for child in root.children.values())

        lines.append(f"Total simulations: {total_visits}")
        lines.append(f"Root value: {root.q_value:+.3f}")
        lines.append("")

        # Sort children by visit count (use Q-value as tiebreaker: lower = better for us)
        sorted_children = sorted(
            root.children.items(),
            key=lambda x: (x[1].visit_count, -x[1].q_value if x[1].visit_count > 0 else x[1].prior),
            reverse=True,
        )

        # Show top moves
        lines.append(f"Top {min(top_n, len(sorted_children))} moves:")
        lines.append("-" * 60)
        lines.append(f"{'Move':<8} {'Visits':>8} {'%':>7} {'Q-value':>9} {'Prior':>7}")
        lines.append("-" * 60)

        for move, child in sorted_children[:top_n]:
            pct = (child.visit_count / max(total_visits, 1)) * 100
            # Negate Q-value: stored Q is from opponent's perspective,
            # but we want to show value for the player making this move
            display_q = -child.q_value if child.visit_count > 0 else 0.0
            lines.append(
                f"{board.san(move):<8} {child.visit_count:>8} {pct:>6.1f}% "
                f"{display_q:>+8.3f} {child.prior:>6.1%}"
            )

        lines.append("")

        # Show principal variation
        pv = self.get_pv(board, depth=10)
        if pv:
            pv_san = []
            temp_board = board.copy()
            for move in pv:
                pv_san.append(temp_board.san(move))
                temp_board.push(move)
            lines.append(f"Principal Variation: {' '.join(pv_san)}")
            lines.append("")

        # Show tree structure for top moves
        lines.append("Search Tree (depth limited):")
        lines.append("-" * 60)

        for move, child in sorted_children[: min(3, len(sorted_children))]:
            self._format_tree_node(
                lines, board, move, child, depth=0, max_depth=max_depth, top_n=2
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def get_search_tree_data(
        self,
        board: chess.Board,
        top_n: int = 3,
        max_depth: int = 3,
    ) -> list[dict]:
        """
        Get the search tree as a nested data structure for GUI display.

        Args:
            board: Current position.
            top_n: Number of top moves to show at each level.
            max_depth: Maximum depth to return.

        Returns:
            List of dicts, each with keys: san, visits, q_value, prior, children
        """
        root = self._get_or_create_node(board)

        # Sort children by visit count
        sorted_children = sorted(
            root.children.items(),
            key=lambda x: x[1].visit_count,
            reverse=True,
        )

        result = []
        for move, child in sorted_children[:top_n]:
            if child.visit_count > 0:
                node_data = self._build_tree_node_data(
                    board, move, child, depth=0, max_depth=max_depth, top_n=top_n
                )
                result.append(node_data)

        return result

    def _build_tree_node_data(
        self,
        board: chess.Board,
        move: chess.Move,
        node: MCTSNode,
        depth: int,
        max_depth: int,
        top_n: int,
    ) -> dict:
        """Build a tree node data dict recursively."""
        san_move = board.san(move)
        # Negate Q-value: stored Q is from opponent's perspective
        display_q = -node.q_value if node.visit_count > 0 else 0.0

        node_data = {
            "san": san_move,
            "visits": node.visit_count,
            "q_value": display_q,
            "prior": node.prior,
            "children": [],
        }

        if depth < max_depth and node.children:
            # Apply move to get next board state
            next_board = board.copy()
            next_board.push(move)

            # Sort and add top children
            sorted_children = sorted(
                node.children.items(),
                key=lambda x: x[1].visit_count,
                reverse=True,
            )

            for child_move, child_node in sorted_children[:top_n]:
                if child_node.visit_count > 0:
                    child_data = self._build_tree_node_data(
                        next_board, child_move, child_node,
                        depth + 1, max_depth, top_n
                    )
                    node_data["children"].append(child_data)

        return node_data

    def _format_tree_node(
        self,
        lines: list[str],
        board: chess.Board,
        move: chess.Move,
        node: MCTSNode,
        depth: int,
        max_depth: int,
        top_n: int,
    ) -> None:
        """
        Recursively format a tree node for display.

        Args:
            lines: List to append formatted lines to.
            board: Board state before this move.
            move: Move leading to this node.
            node: The node to format.
            depth: Current depth in the tree.
            max_depth: Maximum depth to display.
            top_n: Number of top children to show.
        """
        indent = "  " * depth
        prefix = "├─" if depth > 0 else ""

        san_move = board.san(move)
        # Negate Q-value: stored Q is from opponent's perspective,
        # but we want to show value for the player making this move
        display_q = -node.q_value if node.visit_count > 0 else 0.0
        lines.append(
            f"{indent}{prefix}{san_move} "
            f"(N={node.visit_count}, Q={display_q:+.3f}, P={node.prior:.1%})"
        )

        if depth >= max_depth or not node.children:
            return

        # Apply move to get next board state
        next_board = board.copy()
        next_board.push(move)

        # Sort and show top children
        sorted_children = sorted(
            node.children.items(),
            key=lambda x: x[1].visit_count,
            reverse=True,
        )

        for child_move, child_node in sorted_children[:top_n]:
            if child_node.visit_count > 0:
                self._format_tree_node(
                    lines,
                    next_board,
                    child_move,
                    child_node,
                    depth + 1,
                    max_depth,
                    top_n,
                )

    def print_search_tree(
        self,
        board: chess.Board,
        top_n: int = 10,
        max_depth: int = 10,
    ) -> None:
        """
        Print the search tree to stdout.

        Args:
            board: Current position.
            top_n: Number of top moves to show at each level.
            max_depth: Maximum depth to display.
        """
        print(self.get_search_tree_string(board, top_n, max_depth))
