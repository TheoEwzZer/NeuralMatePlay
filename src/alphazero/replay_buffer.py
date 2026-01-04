"""
Experience replay buffer for AlphaZero training.

Features:
- Circular buffer with configurable max size
- Priority sampling favoring recent examples
- Efficient batch sampling
"""

import random
from typing import Optional
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Circular buffer for storing and sampling training examples.

    Stores (state, policy, value) tuples from self-play games.
    """

    def __init__(
        self,
        max_size: int = 500000,
        recent_weight: float = 0.8,
    ):
        """
        Initialize replay buffer.

        Args:
            max_size: Maximum number of examples to store.
            recent_weight: Weight for sampling from recent examples (0-1).
                           0.8 means 80% of samples come from the recent 25%.
        """
        self.max_size = max_size
        self.recent_weight = recent_weight

        self._states: deque = deque(maxlen=max_size)
        self._policies: deque = deque(maxlen=max_size)
        self._values: deque = deque(maxlen=max_size)

    def add(
        self,
        state: np.ndarray,
        policy: np.ndarray,
        value: float,
    ) -> None:
        """
        Add a single training example.

        Args:
            state: Board encoding of shape (planes, 8, 8).
            policy: Policy target of shape (policy_size,).
            value: Value target in [-1, +1].
        """
        self._states.append(state)
        self._policies.append(policy)
        self._values.append(value)

    def add_game(
        self,
        states: list[np.ndarray],
        policies: list[np.ndarray],
        outcome: float,
    ) -> None:
        """
        Add all positions from a game.

        Args:
            states: List of board encodings.
            policies: List of MCTS policy targets.
            outcome: Game outcome (+1 white win, -1 black win, 0 draw).
        """
        for i, (state, policy) in enumerate(zip(states, policies)):
            # Value from the perspective of the player to move
            # Alternates based on whose turn it was
            if i % 2 == 0:
                value = outcome  # White's perspective
            else:
                value = -outcome  # Black's perspective

            self.add(state, policy, value)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of examples with priority weighting.

        Recent examples are sampled more frequently based on recent_weight.

        Args:
            batch_size: Number of examples to sample.

        Returns:
            Tuple of (states, policies, values) as numpy arrays.
        """
        n = len(self)
        if n == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_size = min(batch_size, n)

        # Split between recent and old samples
        recent_count = int(batch_size * self.recent_weight)
        old_count = batch_size - recent_count

        # Recent samples from last 25% of buffer
        recent_start = max(0, n - n // 4)
        recent_indices = random.sample(
            range(recent_start, n), min(recent_count, n - recent_start)
        )

        # Old samples from first 75% of buffer
        if recent_start > 0 and old_count > 0:
            old_indices = random.sample(
                range(0, recent_start), min(old_count, recent_start)
            )
        else:
            old_indices = []

        # Combine indices
        indices = recent_indices + old_indices

        # Fill remaining with random samples if needed
        while len(indices) < batch_size:
            indices.append(random.randrange(n))

        # Gather samples
        states = np.array([self._states[i] for i in indices])
        policies = np.array([self._policies[i] for i in indices])
        values = np.array([self._values[i] for i in indices])

        return states, policies, values

    def sample_uniform(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample uniformly without priority weighting.

        Args:
            batch_size: Number of examples to sample.

        Returns:
            Tuple of (states, policies, values) as numpy arrays.
        """
        n = len(self)
        if n == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_size = min(batch_size, n)
        indices = random.sample(range(n), batch_size)

        states = np.array([self._states[i] for i in indices])
        policies = np.array([self._policies[i] for i in indices])
        values = np.array([self._values[i] for i in indices])

        return states, policies, values

    def get_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all examples in the buffer.

        Returns:
            Tuple of (states, policies, values) as numpy arrays.
        """
        if len(self) == 0:
            raise ValueError("Buffer is empty")

        states = np.array(list(self._states))
        policies = np.array(list(self._policies))
        values = np.array(list(self._values))

        return states, policies, values

    def clear(self) -> None:
        """Clear all examples from buffer."""
        self._states.clear()
        self._policies.clear()
        self._values.clear()

    def purge_recent(self, ratio: float) -> int:
        """
        Remove most recent entries from buffer.

        Used to purge potentially low-quality data after veto detection.

        Args:
            ratio: Fraction of buffer to remove (0.25 = remove 25% of entries).

        Returns:
            Number of entries removed.
        """
        if ratio <= 0 or len(self) == 0:
            return 0

        purge_count = int(len(self) * ratio)
        purge_count = min(purge_count, len(self))

        for _ in range(purge_count):
            self._states.pop()
            self._policies.pop()
            self._values.pop()

        return purge_count

    def __len__(self) -> int:
        """Return number of examples in buffer."""
        return len(self._states)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self) == 0

    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self) >= self.max_size

    def save(self, path: str) -> None:
        """
        Save buffer to file.

        Args:
            path: Path to save file (.npz format).
        """
        states, policies, values = self.get_all()
        np.savez_compressed(
            path,
            states=states,
            policies=policies,
            values=values,
        )

    def load(self, path: str) -> None:
        """
        Load buffer from file.

        Args:
            path: Path to saved file.
        """
        data = np.load(path)

        self.clear()
        for state, policy, value in zip(
            data["states"], data["policies"], data["values"]
        ):
            self.add(state, policy, float(value))

    def get_statistics(self) -> dict:
        """
        Get buffer statistics.

        Returns:
            Dictionary with buffer statistics.
        """
        if len(self) == 0:
            return {
                "size": 0,
                "capacity": self.max_size,
                "fill_ratio": 0.0,
            }

        values = np.array(list(self._values))

        return {
            "size": len(self),
            "capacity": self.max_size,
            "fill_ratio": len(self) / self.max_size,
            "value_mean": float(np.mean(values)),
            "value_std": float(np.std(values)),
            "value_min": float(np.min(values)),
            "value_max": float(np.max(values)),
        }
