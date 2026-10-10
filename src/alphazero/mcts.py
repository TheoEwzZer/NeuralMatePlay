"""
Monte Carlo Tree Search with neural network guidance.

Features:
- PUCT selection formula (Polynomial Upper Confidence Trees)
- Virtual losses for parallel search
- Transposition table for position caching
- Dirichlet noise at root for exploration
- Temperature-based move selection
- Batched neural network inference
- Draw detection inside the tree (repetition, 50-move rule, insufficient material)
"""

import math
from collections import Counter
from typing import Any, Hashable, Iterator, Optional
from dataclasses import dataclass, field

import chess
import numpy as np
from numpy.random import Generator

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
class MCTSNode(object):
    """
    Node in the MCTS tree.

    Stores visit counts, value estimates, and child nodes.
    """

    # P(s, a) - prior probability from policy network
    prior: float = 0.0
    # N(s, a) - number of visits
    visit_count: int = 0
    # W(s, a) - total value from all visits
    total_value: float = 0.0
    # Virtual loss for parallel search
    virtual_loss: int = 0
    # move -> MCTSNode
    children: dict[chess.Move, "MCTSNode"] = field(default_factory=dict)
    is_expanded: bool = False
    # WDL probabilities: [P(win), P(draw), P(loss)] - summed for averaging
    total_wdl: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    @property
    def q_value(self) -> float:
        """Q(s, a) = W(s, a) / N(s, a) - mean value"""
        if self.visit_count == 0:
            return 0.0

        return self.total_value / self.visit_count

    @property
    def wdl(self) -> np.ndarray:
        """Average WDL probabilities [P(win), P(draw), P(loss)]."""
        if self.visit_count == 0:
            return np.array([0.33, 0.34, 0.33], dtype=np.float32)
        return self.total_wdl / self.visit_count

    @property
    def effective_visits(self) -> int:
        """Visits including virtual losses."""
        return self.visit_count + self.virtual_loss


class MCTS(object):
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
        # First Play Urgency reduction
        fpu_reduction: float = 0.25,
        # 0 = deterministic, >0 = sample from policy
        temperature: float = 0.0,
        # WDL-aware parameters
        # Dynamic draw penalty (scales with W-L)
        contempt: float = 0.0,
        # Exploration bonus for sharp positions
        uncertainty_weight: float = 0.0,
        # Adaptive FPU when draw is found
        draw_sibling_fpu: bool = False,
        # Softmax temperature applied to the network priors (1.0 = unchanged)
        policy_temperature: float = 1.0,
    ) -> None:
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
            contempt: Dynamic draw penalty. Only active when winning (W > L) to avoid
                draws. When losing, contempt=0 to play for complications. Use ~0.5.
            uncertainty_weight: Bonus for exploring sharp positions (high W and L).
            draw_sibling_fpu: If True, don't reduce FPU when a draw move is found.
            policy_temperature: Temperature applied to the priors before search.
                Values > 1 flatten an overconfident policy so that the search
                still considers alternative moves.
        """
        self.network: DualHeadNetwork = network
        self.num_simulations: int = num_simulations
        self.c_puct: float = c_puct
        self.dirichlet_alpha: float = dirichlet_alpha
        self.dirichlet_epsilon: float = dirichlet_epsilon
        self.batch_size: int = batch_size
        self.history_length: int = history_length
        self.fpu_reduction: float = fpu_reduction
        self.temperature: float = temperature
        # WDL-aware settings
        self.contempt: float = contempt
        self.uncertainty_weight: float = uncertainty_weight
        self.draw_sibling_fpu: bool = draw_sibling_fpu
        self.policy_temperature: float = policy_temperature

        # Random number generator
        self._rng: Generator = np.random.default_rng(42)

        # Root nodes (position key -> node), used for tree reuse between moves
        self._transposition_table: dict[Hashable, MCTSNode] = {}

        # Evaluation cache (network input key -> (policy, value, wdl))
        # Avoids re-evaluating same positions reached via different paths
        self._eval_cache: dict[Hashable, tuple[np.ndarray, float, np.ndarray]] = {}
        # ~400MB max (8KB per entry)
        self._max_eval_cache_size = 50000

        # Occurrences of each position in the game before the root
        # (filled at the start of each search, used for repetition draws)
        self._game_position_counts: Counter[Hashable] = Counter()

        # Position history tracker
        self._history = PositionHistory(history_length)

    @staticmethod
    def _position_key(board: chess.Board) -> Hashable:
        """
        Key identifying a position for repetition purposes.

        Includes piece placement, side to move, castling rights and the
        en passant square (same notion as python-chess repetition checks).
        """
        return board._transposition_key()

    def _eval_key(self, history: list[chess.Board]) -> Hashable:
        """
        Key identifying the full network input for the evaluation cache.

        The encoding depends on the history positions, the move counters and
        the repetition count, so all of them are part of the key.
        """
        current: chess.Board = history[0]
        if current.is_repetition(3):
            repetition = 2
        elif current.is_repetition(2):
            repetition = 1
        else:
            repetition = 0
        return (
            tuple(self._position_key(b) for b in history[: self.history_length + 1]),
            current.halfmove_clock,
            min(current.fullmove_number, 200),
            repetition,
        )

    def _prepare_root_history(self, board: chess.Board) -> None:
        """Count game positions before the root (for repetition detection)."""
        self._game_position_counts = Counter()
        replay: chess.Board = board.root()
        for move in board.move_stack:
            self._game_position_counts[self._position_key(replay)] += 1
            replay.push(move)

    @staticmethod
    def _terminal_value(board: chess.Board) -> float | None:
        """
        Value of a finished game from the side to move, or None if not finished.

        Only checks rules that depend on the position itself (mate, stalemate,
        insufficient material). Repetitions and the 50-move rule are handled
        during tree traversal.
        """
        if not any(board.generate_legal_moves()):
            return -1.0 if board.is_check() else 0.0
        if board.is_insufficient_material():
            return 0.0
        return None

    def search(
        self,
        board: chess.Board,
        add_noise: bool = False,
        history_boards: list[chess.Board] | None = None,
    ) -> np.ndarray:
        """
        Run MCTS search and return visit count distribution.

        Args:
            board: Current board position.
            add_noise: Whether to add Dirichlet noise at root.
            history_boards: Optional list of previous board positions [T-1, T-2, ...].
                           Used for proper history encoding during search.

        Returns:
            Policy array of shape (MOVE_ENCODING_SIZE,) with visit proportions.
        """
        # Store initial history for use during simulations
        # history_boards should be [T-1, T-2, ...] (most recent first, excluding current)
        self._initial_history: list[chess.Board] = (
            history_boards if history_boards is not None else []
        )
        self._prepare_root_history(board)

        # Get or create root node
        root: MCTSNode = self._get_or_create_node(board)

        # Expand root if needed
        if not root.is_expanded:
            self._expand_node(root, board)

        # Add Dirichlet noise at root for exploration
        if add_noise:
            self._add_dirichlet_noise(root, board)

        # Run simulations with batched neural network evaluation
        self._run_batched_simulations(board, root, self.num_simulations)

        # Build policy from visit counts
        policy: np.ndarray = np.zeros(MOVE_ENCODING_SIZE, dtype=np.float32)
        flip: bool = board.turn == chess.BLACK

        total_visits: int = sum(child.visit_count for child in root.children.values())

        # Check for REAL winning moves: must be checkmate (not just high Q-value)
        # Q-values > 1.0 are numerical errors, not real wins
        winning_moves = []
        for move, child in root.children.items():
            if child.visit_count > 0:
                test_board: chess.Board = board.copy()
                test_board.push(move)
                if test_board.is_checkmate():
                    winning_moves.append((move, child))

        if winning_moves:
            # Found checkmate(s): select the one with most visits (most explored)
            best_move, _ = max(winning_moves, key=lambda x: x[1].visit_count)
            idx: int | None = encode_move_from_perspective(best_move, flip)
            if idx is not None:
                policy[idx] = 1.0
        elif total_visits == 0:
            # No simulations run: use prior probabilities from neural network
            for move, child in root.children.items():
                idx = encode_move_from_perspective(move, flip)
                if idx is not None:
                    policy[idx] = child.prior
        else:
            # Repetition draws are scored as draws inside the tree, so the
            # visit counts already reflect them: no extra filtering here.
            if self.temperature > 0:
                # Temperature > 0: use visit counts for stochastic selection
                for move, child in root.children.items():
                    if child.visit_count > 0:
                        idx = encode_move_from_perspective(move, flip)
                        if idx is not None:
                            policy[idx] = child.visit_count
                policy_sum = policy.sum()
                if policy_sum > 0:
                    policy /= policy_sum
            else:
                # Temperature 0: deterministic selection by visits
                sorted_children: list[tuple[chess.Move, MCTSNode]] = sorted(
                    [(m, c) for m, c in root.children.items() if c.visit_count > 0],
                    key=lambda x: (x[1].visit_count, -x[1].q_value),
                    reverse=True,
                )

                best_move = None
                # (move, mate_distance) - mates for us
                winning_mates = []
                # (move, mate_distance, visits) - mates for opponent
                losing_mates = []

                # Analyze each move
                for move, child in sorted_children:
                    test_board = board.copy()
                    test_board.push(move)

                    # Mate for us? (to play immediately)
                    if test_board.is_checkmate():
                        winning_mates.append((move, 1))
                        continue

                    # Only if the move has been explored enough (avoids false positives)
                    if child.visit_count >= 5:
                        # Deeper mate for us? (via MCTS tree)
                        our_mate: int | None = self._find_forced_mate(
                            child,
                            test_board,
                            is_our_turn=False,
                            our_moves=0,
                            max_our_moves=5,
                        )
                        if our_mate is not None:
                            winning_mates.append((move, our_mate))
                            continue

                        # Mate for the opponent? (to avoid)
                        opp_mate: int | None = self._find_forced_mate(
                            child,
                            test_board,
                            is_our_turn=True,
                            our_moves=0,
                            max_our_moves=5,
                        )
                        if opp_mate is not None:
                            losing_mates.append((move, opp_mate, child.visit_count))

                # Priority 1: Play the fastest mate for us
                if winning_mates:
                    winning_mates.sort(key=lambda x: x[1])
                    best_move = winning_mates[0][0]
                else:
                    # Priority 2: Most visited move that doesn't lead to opponent mate
                    dominated_moves = {m for m, _, _ in losing_mates}

                    for move, child in sorted_children:
                        if move not in dominated_moves:
                            best_move = move
                            break

                    # Priority 3: All moves lose → longest mate distance, then most visited
                    if best_move is None and losing_mates:
                        # Sort by: mate_distance DESC (further = better), then visits DESC
                        losing_mates.sort(key=lambda x: (-x[1], -x[2]))
                        best_move = losing_mates[0][0]

                    # Fallback: take the most visited
                    if best_move is None and sorted_children:
                        best_move = sorted_children[0][0]

                if best_move:
                    idx = encode_move_from_perspective(best_move, flip)
                    if idx is not None:
                        policy[idx] = 1.0

        # Apply temperature
        if self.temperature > 0 and not np.isclose(
            self.temperature, 1.0, rtol=1e-09, atol=1e-09
        ):
            # Raise to power of 1/temperature
            policy = np.power(policy + 1e-10, 1.0 / self.temperature)
            policy /= policy.sum() + 1e-10

        return policy

    def get_best_move(
        self,
        board: chess.Board,
        add_noise: bool = False,
        history_boards: list[chess.Board] | None = None,
    ) -> Optional[chess.Move]:
        """
        Run MCTS and return the best move.

        Args:
            board: Current board position.
            add_noise: Whether to add exploration noise.
            history_boards: Optional list of previous board positions [T-1, T-2, ...].

        Returns:
            Best move according to MCTS, or None if no legal moves.
        """
        if not list(board.legal_moves):
            return None

        policy: np.ndarray = self.search(board, add_noise, history_boards)

        # Sample or select best move based on temperature
        if self.temperature > 0:
            # Sample from policy
            policy = policy / (policy.sum() + 1e-10)
            try:
                idx: int | np.intp = self._rng.choice(len(policy), p=policy)
            except ValueError:
                # Fallback to argmax if sampling fails
                idx = np.argmax(policy)
        else:
            idx = np.argmax(policy)

        # Decode move
        flip: bool = board.turn == chess.BLACK
        move: chess.Move | None = decode_move_from_perspective(idx, board, flip)

        # Fallback to most visited child if decode fails
        if move is None:
            root: MCTSNode = self._get_or_create_node(board)
            if root.children:
                # Check for checkmate first
                for m, c in root.children.items():
                    if c.visit_count > 0:
                        test_board: chess.Board = board.copy()
                        test_board.push(m)
                        if test_board.is_checkmate():
                            move = m
                            break
                # Otherwise take most visited
                if move is None:
                    move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]

        return move

    # =========================================================================
    # Batched MCTS methods (for GPU efficiency)
    # =========================================================================

    def _collect_leaf(
        self,
        board: chess.Board,
        root: MCTSNode,
        root_board: chess.Board,
    ) -> tuple[
        list[tuple[chess.Move | None, MCTSNode]],
        chess.Board,
        float | None,
        list[chess.Board],
    ]:
        """
        Traverse tree to a leaf node, applying virtual losses.

        Args:
            board: Board copy to modify during traversal.
            root: Root node of the tree.
            root_board: Original root board (for history building).

        Returns:
            Tuple of (path, leaf_board, terminal_value, history_boards):
            - path: list of (move, node) pairs from root to leaf
            - leaf_board: board state at the leaf
            - terminal_value: value from the leaf's side to move if the game
              is over at the leaf (mate, stalemate, draw by rule), else None
            - history_boards: list of boards [current, T-1, T-2, ...] for encoding
        """
        node: MCTSNode = root
        path: list[tuple[chess.Move | None, MCTSNode]] = [(None, node)]

        # Positions seen since the root (repeating one of them is a draw)
        path_keys: set[Hashable] = {self._position_key(board)}

        # Selection: traverse tree to leaf
        while node.is_expanded and node.children:
            # Select best child using PUCT
            move, child = self._select_child(node)

            # Apply virtual loss (prevents other traversals from selecting same path)
            child.virtual_loss += 1

            # Make move
            board.push(move)
            path.append((move, child))
            node = child

            # Draw by repetition: position already seen since the root, or
            # twice in the game before the root (third occurrence)
            key: Hashable = self._position_key(board)
            if key in path_keys or self._game_position_counts[key] >= 2:
                return path, board, 0.0, self._leaf_history(board, len(path) - 1)
            path_keys.add(key)

            # Draw by the 50-move rule (unless this move delivers mate)
            if board.halfmove_clock >= 100 and not board.is_checkmate():
                return path, board, 0.0, self._leaf_history(board, len(path) - 1)

        if node.is_expanded:
            # Expanded node without children: no legal moves
            terminal_value: float | None = -1.0 if board.is_check() else 0.0
        else:
            terminal_value = self._terminal_value(board)

        history_boards: list[chess.Board] = (
            []
            if terminal_value is not None
            else self._leaf_history(board, len(path) - 1)
        )
        return path, board, terminal_value, history_boards

    def _leaf_history(self, board: chess.Board, depth: int) -> list[chess.Board]:
        """
        Build the encoding history [current, T-1, T-2, ...] for a leaf.

        Args:
            board: Board at the leaf (with move stack).
            depth: Number of moves played from the root to reach the leaf.

        Returns:
            Up to history_length + 1 boards, most recent first.
        """
        history: list[chess.Board] = [board.copy()]
        steps: int = min(depth, self.history_length)
        if steps:
            tmp: chess.Board = board.copy(stack=steps)
            for _ in range(steps):
                tmp.pop()
                history.append(tmp.copy(stack=False))

        missing: int = self.history_length + 1 - len(history)
        if missing > 0 and self._initial_history:
            # Positions before the root, most recent first
            recent: list[chess.Board] = list(
                reversed(self._initial_history[-self.history_length :])
            )
            history += recent[:missing]
        return history

    @staticmethod
    def _backpropagate(
        path: list[tuple[chess.Move | None, MCTSNode]],
        value: float,
        wdl: np.ndarray | None = None,
    ) -> None:
        """
        Backpropagate value and WDL through the path, removing virtual losses.

        Args:
            path: List of (move, node) pairs from root to leaf.
            value: Value from the leaf node's perspective.
            wdl: WDL probabilities [W, D, L] from leaf's perspective.
        """
        # Default WDL based on value if not provided
        if wdl is None:
            if value > 0.5:
                wdl = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif value < -0.5:
                wdl = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                wdl = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        flipped_wdl: np.ndarray = wdl[::-1]

        for i, (move, path_node) in enumerate(reversed(path)):
            # Remove virtual loss
            if move is not None:
                path_node.virtual_loss = max(0, path_node.virtual_loss - 1)

            # Update statistics (flip value and WDL for each level)
            path_node.visit_count += 1
            if i % 2 == 0:
                path_node.total_value += value
                path_node.total_wdl += wdl
            else:
                path_node.total_value -= value
                # Flip WDL: opponent's win is our loss
                path_node.total_wdl += flipped_wdl

    @staticmethod
    def _revert_virtual_loss(path: list[tuple[chess.Move | None, MCTSNode]]) -> None:
        """Remove the virtual losses of a traversal that is not evaluated."""
        for move, path_node in path:
            if move is not None:
                path_node.virtual_loss = max(0, path_node.virtual_loss - 1)

    def _create_children(
        self,
        node: MCTSNode,
        board: chess.Board,
        policy: np.ndarray,
    ) -> None:
        """
        Create children for all legal moves with masked, normalized priors.

        Args:
            node: Node to expand.
            board: Board state at this node.
            policy: Policy array from neural network.
        """
        flip: bool = board.turn == chess.BLACK
        legal_moves: list[chess.Move] = list(board.legal_moves)

        priors: np.ndarray = np.empty(len(legal_moves), dtype=np.float64)
        for i, move in enumerate(legal_moves):
            idx: int | None = encode_move_from_perspective(move, flip)
            if idx is not None and idx < len(policy):
                priors[i] = policy[idx]
            else:
                priors[i] = 1e-6

        # Optional policy softmax temperature: p^(1/T), renormalized
        if self.policy_temperature != 1.0 and len(priors) > 0:
            priors = np.power(np.maximum(priors, 1e-12), 1.0 / self.policy_temperature)

        priors /= priors.sum() + 1e-10

        for move, prior in zip(legal_moves, priors):
            node.children[move] = MCTSNode(prior=float(prior))

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
        neural network calls for GPU efficiency. When a traversal reaches a
        leaf that is already waiting for evaluation in the current batch
        (collision), the batch is closed early instead of evaluating the same
        position twice.

        Args:
            board: Root board position.
            root: Root node.
            num_simulations: Number of simulations to run.
        """
        simulations_done = 0

        while simulations_done < num_simulations:
            # Determine batch size for this iteration
            remaining: int = num_simulations - simulations_done
            current_batch_size: int = min(self.batch_size, remaining)

            # (path, leaf_board, leaf_node, history)
            new_leaves: list[
                tuple[
                    list[tuple[chess.Move | None, MCTSNode]],
                    chess.Board,
                    MCTSNode,
                    list[chess.Board],
                ]
            ] = []
            pending_ids: set[int] = set()

            for _ in range(current_batch_size):
                board_copy: chess.Board = board.copy()
                path, leaf_board, terminal_value, history = self._collect_leaf(
                    board_copy, root, board
                )
                leaf_node: MCTSNode = path[-1][1]

                if terminal_value is not None:
                    # Game over at the leaf: exact value, no network call
                    if terminal_value > 0:
                        wdl = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    elif terminal_value < 0:
                        wdl = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    else:
                        wdl = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    self._backpropagate(path, terminal_value, wdl)
                    simulations_done += 1
                    continue

                if id(leaf_node) in pending_ids:
                    # Collision: this leaf is already in the batch
                    self._revert_virtual_loss(path)
                    break

                pending_ids.add(id(leaf_node))
                new_leaves.append((path, leaf_board, leaf_node, history))

            if not new_leaves:
                continue

            # Batch evaluate new leaves (with cache lookup)
            # index -> (policy, value, wdl)
            eval_results: dict[int, tuple[np.ndarray, float, np.ndarray]] = {}
            # (index, state, cache_key)
            to_evaluate: list[tuple[int, np.ndarray, Hashable]] = []

            for i, (path, leaf_board, leaf_node, history) in enumerate(new_leaves):
                # Pad with duplicates if history is shorter than needed
                if len(history) < self.history_length + 1:
                    pad_board: chess.Board = history[-1] if history else leaf_board
                    history = history + [pad_board] * (
                        self.history_length + 1 - len(history)
                    )
                cache_key: Hashable = self._eval_key(history)
                cached = self._eval_cache.get(cache_key)
                if cached is not None:
                    eval_results[i] = cached
                else:
                    state: np.ndarray = encode_board_with_history(
                        history, from_perspective=True
                    )
                    to_evaluate.append((i, state, cache_key))

            if to_evaluate:
                states_array: np.ndarray = np.array(
                    [s for _, s, _ in to_evaluate], dtype=np.float32
                )
                policies, values, wdl_batch = self.network.predict_batch_with_wdl(
                    states_array
                )

                for j, (i, _, cache_key) in enumerate(to_evaluate):
                    result = (policies[j], float(values[j]), wdl_batch[j])
                    eval_results[i] = result
                    self._cache_eval(cache_key, *result)

            # Expand nodes and backpropagate
            for i, (path, leaf_board, leaf_node, _history) in enumerate(new_leaves):
                policy, value, wdl = eval_results[i]
                if not leaf_node.is_expanded:
                    self._create_children(leaf_node, leaf_board, policy)
                self._backpropagate(path, value, wdl)
                simulations_done += 1

    def _select_child(self, node: MCTSNode) -> tuple[chess.Move, MCTSNode]:
        """
        Select child node using PUCT formula with WDL enhancements.

        Children with pending virtual losses are scored as if those pending
        visits were losses, so parallel traversals spread over different moves.

        Args:
            node: Parent node.

        Returns:
            Tuple of (selected_move, selected_child).
        """
        if not node.children:
            raise ValueError("Cannot select child from a node with no children")

        sqrt_total: float = math.sqrt(max(1, node.visit_count))

        # Initialize with first child to avoid None
        iterator: Iterator[tuple[chess.Move, MCTSNode]] = iter(node.children.items())
        best_move, best_child = next(iterator)
        best_score = float("-inf")

        # Draw-Sibling-FPU
        draw_found = False
        if self.draw_sibling_fpu:
            for child in node.children.values():
                if (
                    child.visit_count > 0
                    and child.total_wdl[1] > 0.9 * child.visit_count
                ):
                    draw_found = True
                    break

        if draw_found:
            fpu_value: float = node.q_value
        else:
            fpu_value = node.q_value - self.fpu_reduction

        # Dynamic contempt only depends on the parent node
        dynamic_contempt = 0.0
        if self.contempt != 0.0 and node.visit_count > 0:
            node_wdl: np.ndarray = node.wdl
            dynamic_contempt = self.contempt * max(
                float(node_wdl[0] - node_wdl[2]), 0.0
            )

        c_puct: float = self.c_puct
        uncertainty_weight: float = self.uncertainty_weight

        for move, child in node.children.items():
            visits: int = child.visit_count
            virtual_loss: int = child.virtual_loss
            effective_visits: int = visits + virtual_loss

            if effective_visits == 0:
                q: float = fpu_value
            else:
                # Parent's view of the child, pending visits counted as losses
                q = (-child.total_value - virtual_loss) / effective_visits

                if dynamic_contempt > 0.0 and visits > 0:
                    draw_prob = child.total_wdl[1] / visits
                    q -= dynamic_contempt * draw_prob

            u: float = c_puct * child.prior * sqrt_total / (1 + effective_visits)

            if uncertainty_weight > 0.0 and visits > 0:
                uncertainty: float = (
                    math.sqrt(max(float(child.total_wdl[0] * child.total_wdl[2]), 0.0))
                    / visits
                )
                u *= 1.0 + uncertainty_weight * uncertainty

            score: float = q + u

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _expand_node(
        self,
        node: MCTSNode,
        board: chess.Board,
    ) -> None:
        """
        Expand the root node: evaluate it and add children for all legal moves.

        The root's own evaluation counts as its first visit.

        Args:
            node: Node to expand.
            board: Current board state.
        """
        if node.is_expanded:
            return

        # Build history from initial_history: [current, T-1, T-2, ...]
        # Take the LAST history_length positions and reverse to get [T-1, T-2, T-3] order
        if self._initial_history:
            recent_history: list[chess.Board] = list(
                reversed(self._initial_history[-self.history_length :])
            )
        else:
            recent_history = []
        history_boards: list[chess.Board] = [board] + recent_history

        # Pad if needed
        if len(history_boards) < self.history_length + 1:
            pad_board: chess.Board = history_boards[-1]
            history_boards = history_boards + [pad_board] * (
                self.history_length + 1 - len(history_boards)
            )

        cache_key: Hashable = self._eval_key(history_boards)
        cached = self._eval_cache.get(cache_key)
        if cached is not None:
            policy, value, wdl = cached
        else:
            state: np.ndarray = encode_board_with_history(
                history_boards, from_perspective=True
            )
            policy, value, wdl = self.network.predict_single_with_wdl(state)
            self._cache_eval(cache_key, policy, value, wdl)

        # Store initial value and WDL
        node.visit_count = 1
        node.total_value = value
        node.total_wdl = wdl.copy()

        self._create_children(node, board, policy)

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
        num_moves: int = len(node.children)
        noise: np.ndarray = self._rng.dirichlet([self.dirichlet_alpha] * num_moves)

        # Apply noise to priors
        for i, (_, child) in enumerate(node.children.items()):
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
        # Key includes side to move, castling rights and en passant square,
        # so a stored node always has children matching the legal moves
        key: Hashable = self._position_key(board)

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

    def advance_root(self, new_board: chess.Board) -> int:
        """
        Advance the search tree to a new position after a move is played.

        This enables tree reuse: instead of starting from scratch, we keep
        the subtree rooted at the new position (if it was explored before).

        Args:
            new_board: The board position after the move was played.

        Returns:
            Number of simulations preserved (visit count of new root),
            or 0 if position wasn't in the tree.
        """
        new_root: MCTSNode | None = None
        preserved_visits = 0

        if new_board.move_stack:
            last_move: chess.Move = new_board.peek()
            new_board.pop()
            old_root: MCTSNode | None = self._transposition_table.get(
                self._position_key(new_board)
            )
            new_board.push(last_move)

            if old_root and last_move in old_root.children:
                new_root = old_root.children[last_move]
                preserved_visits = new_root.visit_count

        # Clear transposition table and set new root
        self._transposition_table.clear()

        if new_root:
            self._transposition_table[self._position_key(new_board)] = new_root

        # Clear eval cache (old positions no longer needed)
        self._eval_cache.clear()

        return preserved_visits

    def _cache_eval(
        self,
        key: Hashable,
        policy: np.ndarray,
        value: float,
        wdl: np.ndarray,
    ) -> None:
        """
        Cache evaluation for a network input.

        Args:
            key: Key from _eval_key.
            policy: Policy array.
            value: Value estimate.
            wdl: WDL probabilities [P(win), P(draw), P(loss)].
        """
        # Clear cache if too large to avoid memory issues
        if len(self._eval_cache) >= self._max_eval_cache_size:
            self._eval_cache.clear()

        self._eval_cache[key] = (policy.copy(), value, wdl.copy())

    def get_pv(self, board: chess.Board, depth: int = 50) -> list[chess.Move]:
        """
        Get principal variation (most visited path).

        Args:
            board: Starting position.
            depth: Maximum depth.

        Returns:
            List of moves in the PV.
        """
        pv = []
        current_board: chess.Board = board.copy()
        node: MCTSNode = self._get_or_create_node(current_board)

        for _ in range(depth):
            if not node.children:
                break

            # Only consider children with visits
            visited_children = [
                (m, c) for m, c in node.children.items() if c.visit_count > 0
            ]
            if not visited_children:
                break

            # Check for checkmate first
            checkmate_move = None
            for m, c in visited_children:
                test_board: chess.Board = current_board.copy()
                test_board.push(m)
                if test_board.is_checkmate():
                    checkmate_move = (m, c)
                    break

            if checkmate_move:
                best_move, best_child = checkmate_move
            else:
                # Get most visited move (use Q-value as tiebreaker: lower = better for us)
                best_move, best_child = max(
                    visited_children,
                    key=lambda x: (x[1].visit_count, -x[1].q_value),
                )
            pv.append(best_move)
            current_board.push(best_move)

            # Follow the child node directly (not via transposition table)
            node = best_child

        return pv

    def get_mate_in(self, board: chess.Board) -> int | None:
        """
        Check if there's a forced mate in the search tree.

        Uses minimax: we play optimally to mate fastest, opponent defends optimally.

        Args:
            board: Current position.

        Returns:
            Mate in X (number of our moves), or None if no forced mate found.
        """
        root: MCTSNode = self._get_or_create_node(board)
        if not root.children:
            return None

        return self._find_forced_mate(root, board.copy(), is_our_turn=True, our_moves=0)

    def _find_forced_mate(
        self,
        node: "MCTSNode",
        board: chess.Board,
        is_our_turn: bool,
        our_moves: int,
        max_our_moves: int = 15,
    ) -> int | None:
        """
        Minimax search for forced mate in the MCTS tree.

        Args:
            node: Current MCTS node.
            board: Current board state.
            is_our_turn: True if it's our turn to move.
            our_moves: Number of our moves played so far.
            max_our_moves: Maximum depth to search.

        Returns:
            Mate in X (number of our moves), or None if no forced mate.
        """
        if our_moves > max_our_moves:
            return None

        if not node.children:
            return None

        visited: list[tuple[chess.Move, MCTSNode]] = [
            (m, c) for m, c in node.children.items() if c.visit_count > 0
        ]
        if not visited:
            return None

        if is_our_turn:
            # Our turn: find the fastest mate among all our options
            best_mate: int | None = None
            for move, child in visited:
                temp_board = board.copy()
                temp_board.push(move)

                if temp_board.is_checkmate():
                    # Mate in 1!
                    return our_moves + 1

                child_mate: int | None = self._find_forced_mate(
                    child, temp_board, False, our_moves + 1, max_our_moves
                )
                if child_mate is not None:
                    if best_mate is None or child_mate < best_mate:
                        best_mate = child_mate

            return best_mate
        else:
            # Opponent's turn: ALL responses must lead to mate (forced).
            # An unexplored reply may be a defence, so every legal reply
            # must have been searched before concluding.
            if len(visited) < len(node.children):
                return None

            # Return the worst case (longest mate) among opponent's options
            worst_mate: int | None = None
            for move, child in visited:
                temp_board: chess.Board = board.copy()
                temp_board.push(move)

                if temp_board.is_checkmate():
                    # We got mated - not a winning line
                    return None

                child_mate = self._find_forced_mate(
                    child, temp_board, True, our_moves, max_our_moves
                )
                if child_mate is None:
                    # Opponent has escape - no forced mate
                    return None

                if worst_mate is None or child_mate > worst_mate:
                    worst_mate = child_mate

            return worst_mate

    def get_visit_counts(self, board: chess.Board) -> dict[chess.Move, int]:
        """
        Get visit counts for all moves at root.

        Args:
            board: Current position.

        Returns:
            Dictionary mapping moves to visit counts.
        """
        node: MCTSNode = self._get_or_create_node(board)
        return {move: child.visit_count for move, child in node.children.items()}

    def get_root_statistics(self, board: chess.Board) -> list[dict[str, Any]]:
        """
        Get detailed statistics for all moves at root.

        Used by the GUI MCTS panel to display search results.

        Args:
            board: Current position.

        Returns:
            List of dicts with keys: move, visits, q_value, prior, wdl
            Sorted by visits descending.
        """
        node: MCTSNode = self._get_or_create_node(board)
        stats = []

        for move, child in node.children.items():
            # Get WDL from child's perspective and flip for current player
            # Child stores [P(win), P(draw), P(loss)] from opponent's view
            # We flip to [P(loss), P(draw), P(win)] = current player's view
            child_wdl: np.ndarray = child.wdl
            # Flip: current player's win = opponent's loss
            flipped_wdl: np.ndarray = np.array(
                [child_wdl[2], child_wdl[1], child_wdl[0]], dtype=np.float32
            )

            # Check for forced mates after this move
            our_mate_in = None
            opponent_mate_in = None

            test_board: chess.Board = board.copy()
            test_board.push(move)

            # Check if this move leads to draw by repetition
            leads_to_draw_repetition: bool = test_board.is_repetition(3)

            # Direct check: is this move checkmate? (mate in 1)
            if test_board.is_checkmate():
                our_mate_in = 1
            elif child.visit_count > 0:
                # Check deeper mates via MCTS tree
                # Check if WE have a forced mate (good for us)
                our_mate_in: int | None = self._find_forced_mate(
                    child, test_board, is_our_turn=False, our_moves=0, max_our_moves=5
                )
                # Check if OPPONENT has a forced mate (bad for us)
                if our_mate_in is None:
                    opponent_mate_in: int | None = self._find_forced_mate(
                        child,
                        test_board,
                        is_our_turn=True,
                        our_moves=0,
                        max_our_moves=5,
                    )

            stats.append(
                {
                    "move": move,
                    "visits": child.visit_count,
                    # Negate Q-value: stored Q is from opponent's perspective,
                    # we want value from current player's perspective
                    "q_value": -child.q_value if child.visit_count > 0 else 0.0,
                    "prior": child.prior,
                    "total_value": child.total_value,
                    # [P(win), P(draw), P(loss)] for current player
                    "wdl": flipped_wdl,
                    # Forced mate FOR US (winning)
                    "our_mate_in": our_mate_in,
                    # Forced mate for opponent (losing)
                    "opponent_mate_in": opponent_mate_in,
                    # Draw by repetition
                    "leads_to_draw_repetition": leads_to_draw_repetition,
                }
            )

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
        node: MCTSNode = self._get_or_create_node(board)
        if node.visit_count > 0:
            return node.q_value
        return 0.0

    def get_tree_depth(self, board: chess.Board) -> int:
        """
        Get the actual maximum depth of the search tree.

        Args:
            board: Current position.

        Returns:
            Maximum depth reached in the tree (0 if no children).
        """
        node: MCTSNode = self._get_or_create_node(board)
        return self._compute_tree_depth(node)

    def _compute_tree_depth(self, node: MCTSNode) -> int:
        """Recursively compute max depth from a node."""
        if not node.children:
            return 0
        # -1 means no visited children yet
        max_child_depth = -1
        for child in node.children.values():
            if child.visit_count > 0:
                child_depth: int = self._compute_tree_depth(child)
                max_child_depth: int = max(max_child_depth, child_depth)
        # Only count this level if at least one child was visited
        if max_child_depth == -1:
            return 0
        return 1 + max_child_depth

    def get_max_branching_factor(self, board: chess.Board) -> int:
        """
        Get the maximum branching factor (max children) in the tree.

        Args:
            board: Current position.

        Returns:
            Maximum number of children at any node with visits > 0.
        """
        node: MCTSNode = self._get_or_create_node(board)
        return self._compute_max_branching(node)

    def _compute_max_branching(self, node: MCTSNode) -> int:
        """Recursively compute max branching factor."""
        if not node.children:
            return 0
        # Count children with visits
        visited_children: list[MCTSNode] = [
            c for c in node.children.values() if c.visit_count > 0
        ]
        current_branching: int = len(visited_children)
        max_branching: int = current_branching
        for child in visited_children:
            child_max: int = self._compute_max_branching(child)
            max_branching = max(max_branching, child_max)
        return max_branching

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

        root: MCTSNode = self._get_or_create_node(board)
        total_visits: int = sum(child.visit_count for child in root.children.values())

        lines.append(f"Total simulations: {total_visits}")
        lines.append(f"Root value: {root.q_value:+.3f}")
        lines.append("")

        # Sort children by visit count (use Q-value as tiebreaker: lower = better for us)
        sorted_children: list[tuple[chess.Move, MCTSNode]] = sorted(
            root.children.items(),
            key=lambda x: (
                x[1].visit_count,
                -x[1].q_value if x[1].visit_count > 0 else x[1].prior,
            ),
            reverse=True,
        )

        # Show top moves
        lines.append(f"Top {min(top_n, len(sorted_children))} moves:")
        lines.append("-" * 60)
        lines.append(f"{'Move':<8} {'Visits':>8} {'%':>7} {'Q-value':>9} {'Prior':>7}")
        lines.append("-" * 60)

        for move, child in sorted_children[:top_n]:
            pct: float = (child.visit_count / max(total_visits, 1)) * 100
            # Negate Q-value: stored Q is from opponent's perspective,
            # but we want to show value for the player making this move
            display_q: float = -child.q_value if child.visit_count > 0 else 0.0
            lines.append(
                f"{board.san(move):<8} {child.visit_count:>8} {pct:>6.1f}% "
                f"{display_q:>+8.3f} {child.prior:>6.1%}"
            )

        lines.append("")

        # Show principal variation
        pv: list[chess.Move] = self.get_pv(board, depth=10)
        if pv:
            pv_san: list[str] = []
            temp_board: chess.Board = board.copy()
            for move in pv:
                pv_san.append(temp_board.san(move))
                temp_board.push(move)
            lines.append(f"Principal Variation: {' '.join(pv_san)}")
            lines.append("")

        # Show tree structure for top moves
        actual_depth: int = self._compute_tree_depth(root)
        actual_branching: int = self._compute_max_branching(root)
        mate_in: int | None = self.get_mate_in(board)
        if mate_in == 1:
            mate_str = " - Mate"
        elif mate_in:
            mate_str: str = f" - Mate in {mate_in - 1}"
        else:
            mate_str = ""
        lines.append(
            f"Search Tree (depth: {actual_depth}, top: {actual_branching}){mate_str}:"
        )
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
    ) -> list[dict[str, Any]]:
        """
        Get the search tree as a nested data structure for GUI display.

        Args:
            board: Current position.
            top_n: Number of top moves to show at each level.
            max_depth: Maximum depth to return.

        Returns:
            List of dicts, each with keys: san, visits, q_value, prior, children
        """
        root: MCTSNode = self._get_or_create_node(board)

        # Sort children by visit count
        sorted_children: list[tuple[chess.Move, MCTSNode]] = sorted(
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
    ) -> dict[str, Any]:
        """Build a tree node data dict recursively."""
        san_move: str = board.san(move)
        # Negate Q-value: stored Q is from opponent's perspective
        display_q: float = -node.q_value if node.visit_count > 0 else 0.0

        # Check for forced mates at this node (only at root level)
        our_mate_in = None
        opponent_mate_in = None
        if depth == 0:
            next_board: chess.Board = board.copy()
            next_board.push(move)

            # Direct check: is this move checkmate?
            if next_board.is_checkmate():
                our_mate_in = 1
            elif node.visit_count > 0:
                # Check deeper mates via MCTS tree
                our_mate_in: int | None = self._find_forced_mate(
                    node, next_board, is_our_turn=False, our_moves=0, max_our_moves=5
                )
                if our_mate_in is None:
                    opponent_mate_in: int | None = self._find_forced_mate(
                        node, next_board, is_our_turn=True, our_moves=0, max_our_moves=5
                    )

        node_data = {
            "san": san_move,
            "visits": node.visit_count,
            "q_value": display_q,
            "prior": node.prior,
            "our_mate_in": our_mate_in,
            "opponent_mate_in": opponent_mate_in,
            "children": [],
        }

        if depth < max_depth and node.children:
            # Apply move to get next board state
            next_board = board.copy()
            next_board.push(move)

            # Sort and add top children
            sorted_children: list[tuple[chess.Move, MCTSNode]] = sorted(
                node.children.items(),
                key=lambda x: x[1].visit_count,
                reverse=True,
            )

            for child_move, child_node in sorted_children[:top_n]:
                if child_node.visit_count > 0:
                    child_data: dict[str, Any] = self._build_tree_node_data(
                        next_board, child_move, child_node, depth + 1, max_depth, top_n
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
        indent: str = "  " * depth
        prefix: str = "├─" if depth > 0 else ""

        san_move: str = board.san(move)
        # Negate Q-value: stored Q is from opponent's perspective,
        # but we want to show value for the player making this move
        display_q: float = -node.q_value if node.visit_count > 0 else 0.0
        lines.append(
            f"{indent}{prefix}{san_move} "
            f"(N={node.visit_count}, Q={display_q:+.3f}, P={node.prior:.1%})"
        )

        if depth >= max_depth or not node.children:
            return

        # Apply move to get next board state
        next_board: chess.Board = board.copy()
        next_board.push(move)

        # Sort and show top children
        sorted_children: list[tuple[chess.Move, MCTSNode]] = sorted(
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
