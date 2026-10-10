#!/usr/bin/env python3
"""
MCTS Match - Compare two versions of the MCTS code with the same network.

The baseline MCTS is loaded from a git revision (default: main), the candidate
is the current working tree. Both engines use the same network, the same number
of simulations and the same search parameters. Each opening is played twice
with colors swapped, so the result only reflects the search code.

Usage:
    python scripts/mcts_match.py [options]

Options:
    -n, --network PATH     Network file (default: models/pretrained_best_network.pt)
    -s, --simulations N    MCTS simulations per move (default: 200)
    -o, --openings N       Number of openings, each played twice (default: all)
    -w, --workers N        Parallel worker processes (default: 4)
    --set NAME=VALUE       Extra MCTS option for the new engine only
    --baseline-ref REF     Git revision of the baseline MCTS (default: main)
    --pgn PATH             Save games to a PGN file

Examples:
    python scripts/mcts_match.py
    python scripts/mcts_match.py -s 400 -o 20 -w 6 --pgn match.pgn
"""

import argparse
import importlib.util
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from typing import Any

import chess
import chess.pgn

project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Balanced, varied openings (SAN). Each one is played with both colors.
OPENINGS: list[str] = [
    "e4 e5 Nf3 Nc6 Bb5 a6",  # Ruy Lopez
    "e4 e5 Nf3 Nc6 Bc4 Bc5",  # Italian
    "e4 e5 Nf3 Nf6 Nxe5 d6",  # Petroff
    "e4 e5 Nf3 Nc6 d4 exd4",  # Scotch
    "e4 c5 Nf3 d6 d4 cxd4",  # Sicilian Open
    "e4 c5 Nf3 Nc6 d4 cxd4",  # Sicilian Classical
    "e4 c5 Nc3 Nc6 g3 g6",  # Closed Sicilian
    "e4 c5 c3 Nf6 e5 Nd5",  # Alapin
    "e4 e6 d4 d5 Nc3 Bb4",  # French Winawer
    "e4 e6 d4 d5 e5 c5",  # French Advance
    "e4 c6 d4 d5 e5 Bf5",  # Caro-Kann Advance
    "e4 c6 d4 d5 Nc3 dxe4",  # Caro-Kann Classical
    "e4 d5 exd5 Qxd5 Nc3 Qa5",  # Scandinavian
    "e4 Nf6 e5 Nd5 d4 d6",  # Alekhine
    "e4 d6 d4 Nf6 Nc3 g6",  # Pirc
    "e4 g6 d4 Bg7 Nc3 d6",  # Modern
    "d4 d5 c4 e6 Nc3 Nf6",  # QGD
    "d4 d5 c4 c6 Nf3 Nf6",  # Slav
    "d4 d5 c4 dxc4 Nf3 Nf6",  # QGA
    "d4 Nf6 c4 g6 Nc3 Bg7",  # King's Indian
    "d4 Nf6 c4 e6 Nc3 Bb4",  # Nimzo-Indian
    "d4 Nf6 c4 e6 Nf3 b6",  # Queen's Indian
    "d4 Nf6 c4 g6 Nc3 d5",  # Grünfeld
    "d4 Nf6 c4 c5 d5 b5",  # Benko
    "d4 f5 g3 Nf6 Bg2 g6",  # Dutch Leningrad
    "d4 d5 Nf3 Nf6 Bf4 e6",  # London
    "c4 e5 Nc3 Nf6 Nf3 Nc6",  # English Four Knights
    "c4 c5 Nc3 Nc6 g3 g6",  # Symmetrical English
    "Nf3 d5 g3 Nf6 Bg2 c6",  # Réti
    "Nf3 Nf6 c4 b6 g3 Bb7",  # Réti / Queen's Indian setup
    "e4 e5 f4 exf4 Nf3 g5",  # King's Gambit
    "e4 e5 Nc3 Nf6 f4 d5",  # Vienna
    "d4 e6 c4 Bb4+ Bd2 Qe7",  # Bogo-Indian
    "e4 c5 Nf3 e6 d4 cxd4",  # Sicilian Taimanov/Kan
    "d4 Nf6 Bg5 e6 e4 h6",  # Trompowsky
    "b3 e5 Bb2 Nc6 e3 d5",  # Larsen
    "e4 e5 Nf3 Nc6 Nc3 Nf6",  # Four Knights
    "d4 c5 d5 e5 e4 d6",  # Old Benoni
    "g3 d5 Bg2 e5 d3 Nf6",  # King's Fianchetto
    "e4 c5 Nf3 g6 d4 cxd4",  # Accelerated Dragon
]


def load_baseline_mcts_class(ref: str) -> type:
    """Load the MCTS class from a git revision as src.alphazero.mcts_baseline."""
    source: str = subprocess.check_output(
        ["git", "show", f"{ref}:src/alphazero/mcts.py"],
        cwd=project_root,
        text=True,
        encoding="utf-8",
    )
    module_name = "src.alphazero.mcts_baseline"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "src.alphazero"
    sys.modules[module_name] = module
    exec(compile(source, f"{ref}:src/alphazero/mcts.py", "exec"), module.__dict__)
    return module.MCTS


def make_engine(
    mcts_cls: type,
    network: Any,
    args: argparse.Namespace,
    extra: dict[str, Any] | None = None,
) -> Any:
    """Create an MCTS engine with the shared match parameters."""
    mcts = mcts_cls(
        network,
        num_simulations=args.simulations,
        batch_size=args.batch_size,
        c_puct=args.c_puct,
        # Same WDL settings as the play UI
        contempt=0.5,
        uncertainty_weight=0.2,
        draw_sibling_fpu=True,
        **(extra or {}),
    )
    mcts.temperature = 0.0
    return mcts


def play_game(
    engines: dict[chess.Color, Any],
    opening: str,
    max_plies: int,
) -> tuple[str, chess.Board, float, float]:
    """Play one game. Returns (result, board, time_white, time_black)."""
    board = chess.Board()
    for san in opening.split():
        board.push_san(san)

    for engine in engines.values():
        engine.clear_cache(reset_rng=True)

    positions: list[chess.Board] = [chess.Board()]
    tmp = chess.Board()
    for move in board.move_stack:
        tmp.push(move)
        positions.append(tmp.copy())

    times: dict[chess.Color, float] = {chess.WHITE: 0.0, chess.BLACK: 0.0}

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        engine = engines[board.turn]
        start = time.perf_counter()
        move = engine.get_best_move(
            board, add_noise=False, history_boards=positions[:-1]
        )
        times[board.turn] += time.perf_counter() - start
        if move is None or move not in board.legal_moves:
            # Illegal or missing move loses the game
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            return result, board, times[chess.WHITE], times[chess.BLACK]
        board.push(move)
        positions.append(board.copy())
        for e in engines.values():
            e.advance_root(board)

    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
    else:
        # Move limit reached
        result = "1/2-1/2"
    return result, board, times[chess.WHITE], times[chess.BLACK]


def worker(task_queue: Any, result_queue: Any, args: argparse.Namespace) -> None:
    """Worker process: loads network and both engines, plays games."""
    from src.alphazero.network import DualHeadNetwork
    from src.alphazero.mcts import MCTS as CandidateMCTS

    BaselineMCTS = load_baseline_mcts_class(args.baseline_ref)
    network = DualHeadNetwork.load(args.network)

    candidate = make_engine(CandidateMCTS, network, args, args.candidate_options)
    baseline = make_engine(BaselineMCTS, network, args)

    while True:
        task = task_queue.get()
        if task is None:
            break
        game_idx, opening, candidate_color = task
        engines = {
            candidate_color: candidate,
            not candidate_color: baseline,
        }
        try:
            result, board, t_white, t_black = play_game(
                engines, opening, args.max_plies
            )
            pgn = chess.pgn.Game.from_board(board)
            pgn.headers["Event"] = "MCTS match"
            pgn.headers["Round"] = str(game_idx + 1)
            pgn.headers["White"] = "new" if candidate_color == chess.WHITE else "old"
            pgn.headers["Black"] = "old" if candidate_color == chess.WHITE else "new"
            pgn.headers["Result"] = result
            result_queue.put(
                {
                    "idx": game_idx,
                    "opening": opening,
                    "candidate_color": candidate_color,
                    "result": result,
                    "plies": board.ply(),
                    "pgn": str(pgn),
                    "time_new": t_white if candidate_color == chess.WHITE else t_black,
                    "time_old": t_black if candidate_color == chess.WHITE else t_white,
                }
            )
        except Exception as exc:  # noqa: BLE001 - report and keep going
            result_queue.put({"idx": game_idx, "error": repr(exc)})


def elo_from_score(score: float) -> float:
    """Elo difference from an expected score."""
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


def summarize(w: int, d: int, l: int) -> str:
    """Score, Elo with 95% confidence interval and LOS."""
    n = w + d + l
    if n == 0:
        return "no games"
    score = (w + 0.5 * d) / n
    var = (w * (1 - score) ** 2 + d * (0.5 - score) ** 2 + l * (0 - score) ** 2) / n
    stderr = math.sqrt(var / n) if n > 1 else 0.0
    elo = elo_from_score(score)
    lo = elo_from_score(score - 1.96 * stderr)
    hi = elo_from_score(score + 1.96 * stderr)
    los = 0.5 * (1 + math.erf((w - l) / math.sqrt(2 * (w + l)))) if w + l > 0 else 0.5
    return (
        f"+{w} ={d} -{l}  score {score:.1%}  "
        f"Elo {elo:+.0f} [{lo:+.0f}, {hi:+.0f}]  LOS {los:.1%}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="New MCTS vs baseline MCTS")
    parser.add_argument("-n", "--network", default="models/pretrained_best_network.pt")
    parser.add_argument("-s", "--simulations", type=int, default=200)
    parser.add_argument("-o", "--openings", type=int, default=len(OPENINGS))
    parser.add_argument("-w", "--workers", type=int, default=4)
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--baseline-ref", default="main")
    parser.add_argument("--pgn", default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Extra MCTS option for the new engine only (e.g. policy_temperature=1.6)",
    )
    args = parser.parse_args()
    args.candidate_options = {}
    for item in args.set:
        name, value = item.split("=", 1)
        args.candidate_options[name] = float(value)
    args.network = os.path.join(project_root, args.network)

    openings = OPENINGS[: args.openings]
    tasks = []
    for i, opening in enumerate(openings):
        tasks.append((2 * i, opening, chess.WHITE))
        tasks.append((2 * i + 1, opening, chess.BLACK))

    print(
        f"Match: new (working tree) vs old ({args.baseline_ref})\n"
        f"Network: {os.path.relpath(args.network, project_root)}  "
        f"sims={args.simulations}  batch={args.batch_size}  c_puct={args.c_puct}  "
        f"games={len(tasks)}  workers={args.workers}\n"
        f"New engine options: {args.candidate_options or 'defaults'}\n",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in range(args.workers):
        task_queue.put(None)

    procs = [
        ctx.Process(target=worker, args=(task_queue, result_queue, args))
        for _ in range(args.workers)
    ]
    for p in procs:
        p.start()

    w = d = l = 0
    time_new = time_old = 0.0
    games: list[dict[str, Any]] = []
    start = time.time()
    for _ in range(len(tasks)):
        res = result_queue.get()
        if "error" in res:
            print(f"Game {res['idx'] + 1}: ERROR {res['error']}")
            continue
        games.append(res)
        color = res["candidate_color"]
        r = res["result"]
        if r == "1/2-1/2":
            d += 1
            outcome = "draw"
        elif (r == "1-0") == (color == chess.WHITE):
            w += 1
            outcome = "NEW wins"
        else:
            l += 1
            outcome = "OLD wins"
        time_new += res["time_new"]
        time_old += res["time_old"]
        print(
            f"[{len(games):3d}/{len(tasks)}] {res['opening']:<26} "
            f"new={'W' if color == chess.WHITE else 'B'}  {r:<7} {outcome:<9} "
            f"({res['plies']} plies)  | {summarize(w, d, l)}",
            flush=True,
        )

    for p in procs:
        p.join()

    print("\n" + "=" * 70)
    print(f"FINAL (new vs old): {summarize(w, d, l)}")
    print(f"Search time: new {time_new:.0f}s, old {time_old:.0f}s")
    print(f"Wall time: {time.time() - start:.0f}s")

    if args.pgn:
        games.sort(key=lambda g: g["idx"])
        with open(args.pgn, "w", encoding="utf-8") as f:
            f.write("\n\n".join(g["pgn"] for g in games) + "\n")
        print(f"PGN saved to {args.pgn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
