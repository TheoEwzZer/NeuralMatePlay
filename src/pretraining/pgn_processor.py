"""
Chunked PGN processor for large game databases.

Handles memory-efficient processing of large PGN files
by reading games in chunks.
"""

import os
import re
from typing import Generator, Optional, Tuple
from io import StringIO

import chess
import chess.pgn


class PGNProcessor:
    """
    Memory-efficient PGN file processor.

    Reads PGN files in chunks to handle large databases
    without loading everything into memory.
    """

    def __init__(
        self,
        pgn_path: str,
        min_elo: int = 2200,
        max_games: Optional[int] = None,
        chunk_size: int = 10000,
        skip_first_n_moves: int = 8,
    ):
        """
        Initialize PGN processor.

        Args:
            pgn_path: Path to PGN file.
            min_elo: Minimum ELO for both players.
            max_games: Maximum number of games to process (None = all).
            chunk_size: Number of games to process per chunk.
            skip_first_n_moves: Skip opening moves (too theoretical).
        """
        self.pgn_path = pgn_path
        self.min_elo = min_elo
        self.max_games = max_games
        self.chunk_size = chunk_size
        self.skip_first_n_moves = skip_first_n_moves

        # File stats
        self._file_size = os.path.getsize(pgn_path) if os.path.exists(pgn_path) else 0
        self._bytes_read = 0
        self._games_processed = 0
        self._positions_extracted = 0

    @property
    def progress(self) -> float:
        """Return processing progress 0-1."""
        # If max_games is set, use games progress
        if self.max_games is not None and self.max_games > 0:
            return min(1.0, self._games_processed / self.max_games)
        # Otherwise use file progress
        if self._file_size == 0:
            return 0.0
        return min(1.0, self._bytes_read / self._file_size)

    @property
    def games_processed(self) -> int:
        """Return number of games processed."""
        return self._games_processed

    @property
    def positions_extracted(self) -> int:
        """Return number of positions extracted."""
        return self._positions_extracted

    def process_all(
        self,
    ) -> Generator[Tuple[chess.Board, chess.Move, float], None, None]:
        """
        Process entire PGN file.

        Yields:
            Tuple of (board, move, outcome) for each position.
            outcome: 1.0 for white win, -1.0 for black win, 0.0 for draw.
        """
        with open(self.pgn_path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                # Check max_games limit
                if (
                    self.max_games is not None
                    and self._games_processed >= self.max_games
                ):
                    break

                game = chess.pgn.read_game(f)
                if game is None:
                    break

                # Update progress
                self._bytes_read = f.tell()

                # Check ELO filter
                if not self._passes_elo_filter(game):
                    continue

                # Get outcome
                outcome = self._get_outcome(game)
                if outcome is None:
                    continue

                self._games_processed += 1

                # Extract positions
                board = game.board()
                move_num = 0

                for move in game.mainline_moves():
                    move_num += 1

                    # Skip opening moves
                    if move_num <= self.skip_first_n_moves:
                        board.push(move)
                        continue

                    # Yield position
                    yield board.copy(), move, outcome
                    self._positions_extracted += 1

                    board.push(move)

    def process_chunk(
        self,
        start_game: int = 0,
    ) -> Generator[Tuple[chess.Board, chess.Move, float], None, None]:
        """
        Process a chunk of games.

        Args:
            start_game: Game index to start from.

        Yields:
            Tuple of (board, move, outcome) for each position.
        """
        games_in_chunk = 0
        current_game = 0

        with open(self.pgn_path, "r", encoding="utf-8", errors="ignore") as f:
            while games_in_chunk < self.chunk_size:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                self._bytes_read = f.tell()
                current_game += 1

                # Skip to start position
                if current_game <= start_game:
                    continue

                # Check ELO filter
                if not self._passes_elo_filter(game):
                    continue

                # Get outcome
                outcome = self._get_outcome(game)
                if outcome is None:
                    continue

                games_in_chunk += 1
                self._games_processed += 1

                # Extract positions
                board = game.board()
                move_num = 0

                for move in game.mainline_moves():
                    move_num += 1

                    if move_num <= self.skip_first_n_moves:
                        board.push(move)
                        continue

                    yield board.copy(), move, outcome
                    self._positions_extracted += 1

                    board.push(move)

    def _passes_elo_filter(self, game: chess.pgn.Game) -> bool:
        """Check if game passes ELO filter."""
        try:
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))

            # Check minimum ELO
            if white_elo < self.min_elo or black_elo < self.min_elo:
                return False

            return True
        except (ValueError, TypeError):
            return False

    def _get_outcome(self, game: chess.pgn.Game) -> Optional[float]:
        """Get game outcome as float. Filters out time forfeits and abandoned games."""
        # Filter out games that didn't end normally (timeout, abandoned, etc.)
        # These have results that don't reflect the actual position quality
        termination = game.headers.get("Termination", "Normal")
        if termination and termination.lower() not in ["normal", ""]:
            return None  # Skip time forfeit, abandoned, rules infraction

        result = game.headers.get("Result", "*")
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        elif result == "1/2-1/2":
            return 0.0
        else:
            return None

    def count_games(self) -> int:
        """Count total games in PGN (can be slow for large files)."""
        count = 0
        with open(self.pgn_path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                count += 1
        return count

    def estimate_games(self) -> int:
        """Estimate number of games based on file size."""
        # Rough estimate: ~5KB per game on average
        return self._file_size // 5000

    def reset(self) -> None:
        """Reset processor state."""
        self._bytes_read = 0
        self._games_processed = 0
        self._positions_extracted = 0


def parse_pgn_string(pgn_string: str) -> Optional[chess.pgn.Game]:
    """Parse a PGN string into a game object."""
    return chess.pgn.read_game(StringIO(pgn_string))
