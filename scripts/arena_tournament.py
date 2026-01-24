#!/usr/bin/env python3
"""
Arena Tournament - Round-robin tournament between checkpoints.

Plays all checkpoints against each other N times to determine rankings.

Usage:
    python arena_tournament.py [checkpoint_dir] [options]

Options:
    -t, --type TYPE       Checkpoint type: train or pretrain (default: pretrain)
    -n, --games N         Number of games per match (default: 10)
    -s, --simulations N   MCTS simulations per move (default: 100)
    -m, --max-checkpoints Max number of checkpoints to include (default: all)
    --quick               Quick mode: 4 games, 50 simulations

Examples:
    python arena_tournament.py models -t pretrain         # Tournament of pretrain epochs
    python arena_tournament.py models -t pretrain -n 20   # 20 games per match
    python arena_tournament.py checkpoints -t train       # Tournament of RL iterations
    python arena_tournament.py models --quick             # Quick test mode
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from alphazero import DualHeadNetwork
from alphazero.arena import Arena, NetworkPlayer
from diagnostics.core.network_loader import get_available_checkpoints
from diagnostics.core.colors import Colors, header, subheader


@dataclass
class PlayerStats:
    """Statistics for a single player in the tournament."""
    name: str
    checkpoint_num: int
    path: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    points: float = 0.0  # 1 for win, 0.5 for draw
    games_played: int = 0

    # Head-to-head results: opponent_num -> (wins, losses, draws)
    head_to_head: dict = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    @property
    def score_rate(self) -> float:
        """Points per game (1 for win, 0.5 for draw)."""
        if self.games_played == 0:
            return 0.0
        return self.points / self.games_played


def load_checkpoints(checkpoint_dir: str, checkpoint_type: str, max_checkpoints: Optional[int] = None):
    """Load all available checkpoints."""
    checkpoints = get_available_checkpoints(checkpoint_dir, checkpoint_type)

    if not checkpoints:
        print(f"{Colors.RED}No {checkpoint_type} checkpoints found in {checkpoint_dir}!{Colors.ENDC}")
        return []

    if max_checkpoints and len(checkpoints) > max_checkpoints:
        # Keep evenly distributed checkpoints including first and last
        step = (len(checkpoints) - 1) / (max_checkpoints - 1) if max_checkpoints > 1 else 1
        indices = [int(i * step) for i in range(max_checkpoints)]
        checkpoints = [checkpoints[i] for i in indices]

    return checkpoints


def run_tournament(
    checkpoint_dir: str,
    checkpoint_type: str = "pretrain",
    num_games: int = 10,
    num_simulations: int = 100,
    max_checkpoints: Optional[int] = None,
):
    """Run a round-robin tournament between all checkpoints."""

    print(header("ARENA TOURNAMENT"))
    print(f"  Directory:    {checkpoint_dir}")
    print(f"  Type:         {checkpoint_type}")
    print(f"  Games/match:  {num_games}")
    print(f"  Simulations:  {num_simulations}")
    print()

    # Load checkpoints
    checkpoints = load_checkpoints(checkpoint_dir, checkpoint_type, max_checkpoints)

    if len(checkpoints) < 2:
        print(f"{Colors.RED}Need at least 2 checkpoints for a tournament!{Colors.ENDC}")
        return

    print(f"Found {len(checkpoints)} checkpoints:")
    for num, path in checkpoints:
        label = "epoch" if checkpoint_type == "pretrain" else "iteration"
        print(f"  - {label} {num}: {os.path.basename(path)}")
    print()

    # Calculate total matches
    n = len(checkpoints)
    total_matches = n * (n - 1) // 2
    total_games = total_matches * num_games

    print(f"Total matches: {total_matches}")
    print(f"Total games:   {total_games}")
    print()

    # Initialize player stats
    label = "Epoch" if checkpoint_type == "pretrain" else "Iter"
    players = {}
    for num, path in checkpoints:
        players[num] = PlayerStats(
            name=f"{label} {num}",
            checkpoint_num=num,
            path=path,
        )

    # Load networks
    print(subheader("Loading Networks"))
    networks = {}
    for num, path in checkpoints:
        print(f"  Loading {label} {num}...", end=" ", flush=True)
        try:
            networks[num] = DualHeadNetwork.load(path)
            print(f"{Colors.GREEN}OK{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}FAILED: {e}{Colors.ENDC}")
            return
    print()

    # Create arena
    arena = Arena(
        num_games=num_games,
        num_simulations=num_simulations,
        max_moves=200,
    )

    # Run round-robin
    print(subheader("Running Matches"))
    match_count = 0
    start_time = time.time()

    checkpoint_nums = sorted(players.keys())

    for i, num1 in enumerate(checkpoint_nums):
        for num2 in checkpoint_nums[i + 1:]:
            match_count += 1

            print(f"\n[{match_count}/{total_matches}] {players[num1].name} vs {players[num2].name}")

            # Create players
            player1 = NetworkPlayer(
                networks[num1],
                num_simulations=num_simulations,
                name=players[num1].name,
            )
            player2 = NetworkPlayer(
                networks[num2],
                num_simulations=num_simulations,
                name=players[num2].name,
            )

            # Play match
            match_start = time.time()
            results = arena.play_match(player1, player2)
            match_time = time.time() - match_start

            p1_wins = results["player1_wins"]
            p2_wins = results["player2_wins"]
            draws = results["draws"]

            # Update stats
            players[num1].wins += p1_wins
            players[num1].losses += p2_wins
            players[num1].draws += draws
            players[num1].points += p1_wins + 0.5 * draws
            players[num1].games_played += num_games
            players[num1].head_to_head[num2] = (p1_wins, p2_wins, draws)

            players[num2].wins += p2_wins
            players[num2].losses += p1_wins
            players[num2].draws += draws
            players[num2].points += p2_wins + 0.5 * draws
            players[num2].games_played += num_games
            players[num2].head_to_head[num1] = (p2_wins, p1_wins, draws)

            # Print result
            if p1_wins > p2_wins:
                result_str = f"{Colors.GREEN}{p1_wins}-{p2_wins}{Colors.ENDC}"
            elif p2_wins > p1_wins:
                result_str = f"{Colors.RED}{p1_wins}-{p2_wins}{Colors.ENDC}"
            else:
                result_str = f"{Colors.YELLOW}{p1_wins}-{p2_wins}{Colors.ENDC}"

            if draws > 0:
                result_str += f" ({draws} draws)"

            print(f"  Result: {result_str} ({match_time:.1f}s)")

    total_time = time.time() - start_time

    # Print results
    print()
    print(header("TOURNAMENT RESULTS"))
    print(f"Total time: {total_time / 60:.1f} minutes")
    print()

    # Ranking table
    print(subheader("Final Rankings"))

    # Sort by points (desc), then win rate (desc), then checkpoint number (desc)
    ranked = sorted(
        players.values(),
        key=lambda p: (p.points, p.win_rate, p.checkpoint_num),
        reverse=True,
    )

    print()
    print(f"{'Rank':<6}{'Player':<12}{'Points':<10}{'W':<6}{'D':<6}{'L':<6}{'Score':<10}{'WinRate':<10}")
    print("-" * 76)

    for rank, player in enumerate(ranked, 1):
        score = f"{player.score_rate*100:.1f}%"
        win_rate = f"{player.win_rate*100:.1f}%"

        # Color based on rank
        if rank == 1:
            color = Colors.GREEN
        elif rank == len(ranked):
            color = Colors.RED
        else:
            color = ""
        end_color = Colors.ENDC if color else ""

        print(f"{color}{rank:<6}{player.name:<12}{player.points:<10.1f}{player.wins:<6}{player.draws:<6}{player.losses:<6}{score:<10}{win_rate:<10}{end_color}")

    print()

    # Head-to-head matrix
    print(subheader("Head-to-Head Matrix"))
    print("(Rows = player, Columns = opponent, Values = wins-losses)")
    print()

    # Header row
    header_row = f"{'':>10}"
    for num in checkpoint_nums:
        header_row += f"{label[:1]}{num:>4} "
    print(header_row)
    print("-" * (10 + 5 * len(checkpoint_nums)))

    # Data rows
    for num1 in checkpoint_nums:
        row = f"{label[:1]}{num1:>4}     "
        for num2 in checkpoint_nums:
            if num1 == num2:
                row += "  -  "
            elif num2 in players[num1].head_to_head:
                w, l, d = players[num1].head_to_head[num2]
                if w > l:
                    row += f"{Colors.GREEN}{w}-{l}{Colors.ENDC}  "
                elif l > w:
                    row += f"{Colors.RED}{w}-{l}{Colors.ENDC}  "
                else:
                    row += f"{Colors.YELLOW}{w}-{l}{Colors.ENDC}  "
            else:
                row += "  ?  "
        print(row)

    print()

    # Winner announcement
    winner = ranked[0]
    print(f"{Colors.GREEN}=== WINNER: {winner.name} ==={Colors.ENDC}")
    print(f"    Points: {winner.points:.1f} | Win Rate: {winner.win_rate*100:.1f}%")
    print()

    # Detailed stats
    print(subheader("Detailed Statistics"))

    # Best against specific opponents
    for player in ranked:
        best_opponent = None
        best_score = -1
        worst_opponent = None
        worst_score = float('inf')

        for opp_num, (w, l, d) in player.head_to_head.items():
            score = w + 0.5 * d
            if score > best_score:
                best_score = score
                best_opponent = opp_num
            if score < worst_score:
                worst_score = score
                worst_opponent = opp_num

        if best_opponent is not None:
            w, l, d = player.head_to_head[best_opponent]
            print(f"{player.name}:")
            print(f"  Best vs:  {label} {best_opponent} ({w}-{l}, {d}D)")
            if worst_opponent != best_opponent:
                w, l, d = player.head_to_head[worst_opponent]
                print(f"  Worst vs: {label} {worst_opponent} ({w}-{l}, {d}D)")

def main():
    parser = argparse.ArgumentParser(
        description="Run round-robin tournament between checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "checkpoint_dir",
        nargs="?",
        default="models",
        help="Directory containing checkpoints (default: models)",
    )

    parser.add_argument(
        "-t", "--type",
        choices=["train", "pretrain"],
        default="pretrain",
        help="Checkpoint type (default: pretrain)",
    )

    parser.add_argument(
        "-n", "--games",
        type=int,
        default=10,
        help="Number of games per match (default: 10)",
    )

    parser.add_argument(
        "-s", "--simulations",
        type=int,
        default=100,
        help="MCTS simulations per move (default: 100)",
    )

    parser.add_argument(
        "-m", "--max-checkpoints",
        type=int,
        default=None,
        help="Maximum number of checkpoints to include (default: all)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 4 games, 50 simulations",
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.games = 4
        args.simulations = 50

    run_tournament(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_type=args.type,
        num_games=args.games,
        num_simulations=args.simulations,
        max_checkpoints=args.max_checkpoints,
    )


if __name__ == "__main__":
    main()
