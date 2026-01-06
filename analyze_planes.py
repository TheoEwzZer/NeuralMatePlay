#!/usr/bin/env python3
"""
Analyze and visualize the 60 input planes of the neural network.

Usage:
    python analyze_planes.py                     # Analyze starting position
    python analyze_planes.py --fen "fen_string"  # Analyze specific position
    python analyze_planes.py --pgn games.pgn     # Analyze positions from PGN
    python analyze_planes.py --gui               # Show matplotlib visualization
    python analyze_planes.py --show-planes       # Show each plane as ASCII grid
"""

import argparse
import sys
from typing import Optional
from pathlib import Path

import chess
import chess.pgn
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from alphazero.spatial_encoding import encode_board_with_history, PositionHistory


# Plane descriptions (from perspective of player to move)
# Planes 0-5: "My pieces", Planes 6-11: "Opponent pieces"
def get_plane_descriptions(white_to_move: bool = True) -> dict:
    """Get plane descriptions based on whose turn it is."""
    if white_to_move:
        my, opp = "White", "Black"
    else:
        my, opp = "Black", "White"

    descs = {}
    for t_idx, t_name in enumerate(["T", "T-1", "T-2", "T-3"]):
        offset = t_idx * 12
        descs[offset + 0] = f"{t_name}: {my} Pawns (mine)"
        descs[offset + 1] = f"{t_name}: {my} Knights (mine)"
        descs[offset + 2] = f"{t_name}: {my} Bishops (mine)"
        descs[offset + 3] = f"{t_name}: {my} Rooks (mine)"
        descs[offset + 4] = f"{t_name}: {my} Queens (mine)"
        descs[offset + 5] = f"{t_name}: {my} King (mine)"
        descs[offset + 6] = f"{t_name}: {opp} Pawns (opp)"
        descs[offset + 7] = f"{t_name}: {opp} Knights (opp)"
        descs[offset + 8] = f"{t_name}: {opp} Bishops (opp)"
        descs[offset + 9] = f"{t_name}: {opp} Rooks (opp)"
        descs[offset + 10] = f"{t_name}: {opp} Queens (opp)"
        descs[offset + 11] = f"{t_name}: {opp} King (opp)"

    descs[48] = "Meta: Side to Move"
    descs[49] = "Meta: Move Count (norm)"
    descs[50] = "Meta: My Kingside Castle"
    descs[51] = "Meta: My Queenside Castle"
    descs[52] = "Meta: Opp Kingside Castle"
    descs[53] = "Meta: Opp Queenside Castle"
    descs[54] = "Meta: En Passant Square"
    descs[55] = "Meta: Halfmove Clock (norm)"
    descs[56] = "Meta: Repetition Count"
    descs[57] = "Meta: In Check"
    descs[58] = "Meta: My Attack Map"
    descs[59] = "Meta: Opp Attack Map"

    return descs


# Default descriptions (for backward compatibility)
PLANE_DESCRIPTIONS = get_plane_descriptions(True)


def plane_to_ascii(plane: np.ndarray) -> str:
    """Convert an 8x8 plane to ASCII representation."""
    lines = ["  a b c d e f g h"]
    for rank in range(8):
        row = f"{8 - rank} "
        for file in range(8):
            val = plane[rank, file]
            if val == 0:
                row += ". "
            elif val == 1.0:
                row += "# "
            elif val == 0.5:
                row += "o "
            else:
                # Show as decimal for normalized values
                row += f"{val:.1f}"[1:] + " " if val < 1 else "1 "
        lines.append(row)
    return "\n".join(lines)


def analyze_planes(
    planes: np.ndarray, verbose: bool = False, white_to_move: bool = True
) -> dict:
    """
    Analyze plane utilization.

    Returns dict with statistics per plane.
    """
    descriptions = get_plane_descriptions(white_to_move)
    stats = {}
    for i in range(planes.shape[0]):
        plane = planes[i]
        desc = descriptions.get(i, f"Plane {i}")
        nonzero = np.count_nonzero(plane)
        stats[i] = {
            "description": desc,
            "min": float(plane.min()),
            "max": float(plane.max()),
            "mean": float(plane.mean()),
            "sum": float(plane.sum()),
            "nonzero": nonzero,
            "nonzero_pct": nonzero / 64 * 100,
        }
        if verbose:
            print(f"\n[Plane {i:2d}] {desc}")
            print(plane_to_ascii(plane))

    return stats


def print_summary(stats: dict) -> None:
    """Print summary table of plane utilization."""
    print("\n" + "=" * 80)
    print(
        f"{'Plane':>6} | {'Description':<26} | {'NonZero':>8} | {'Sum':>8} | {'Max':>5}"
    )
    print("-" * 80)

    empty_planes = []
    for i, s in stats.items():
        is_empty = s["nonzero"] == 0
        marker = "  EMPTY" if is_empty else ""
        print(
            f"{i:>6} | {s['description']:<26} | "
            f"{s['nonzero']:>6}/64 | {s['sum']:>8.3f} | {s['max']:>5.2f}{marker}"
        )
        if is_empty:
            empty_planes.append((i, s["description"]))

    print("=" * 80)

    # Summary
    total_planes = len(stats)
    used_planes = sum(1 for s in stats.values() if s["nonzero"] > 0)

    print(
        f"\nPlanes used: {used_planes}/{total_planes} ({used_planes/total_planes*100:.1f}%)"
    )

    if empty_planes:
        print(f"\nEmpty planes ({len(empty_planes)}):")
        for idx, desc in empty_planes:
            print(f"  [{idx:2d}] {desc}")


def analyze_from_pgn(pgn_path: str, max_positions: int = 1000) -> dict:
    """
    Analyze plane utilization across many positions from a PGN file.

    Returns aggregated statistics.
    """
    print(f"Analyzing positions from: {pgn_path}")

    # Track per-plane stats across all positions
    plane_usage = {
        i: {"nonzero_count": 0, "total_sum": 0.0, "positions": 0} for i in range(60)
    }

    positions_analyzed = 0
    games_processed = 0

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while positions_analyzed < max_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            games_processed += 1
            board = game.board()
            history = PositionHistory()

            for move in game.mainline_moves():
                # Update history
                history.push(board.copy())
                board.push(move)

                # Skip first few moves (opening book)
                if board.fullmove_number < 5:
                    continue

                # Encode position
                history.push(board.copy())
                planes = history.encode()

                # Update stats
                for i in range(60):
                    plane = planes[i]
                    nonzero = np.count_nonzero(plane)
                    plane_usage[i]["nonzero_count"] += 1 if nonzero > 0 else 0
                    plane_usage[i]["total_sum"] += float(plane.sum())
                    plane_usage[i]["positions"] += 1

                positions_analyzed += 1
                if positions_analyzed >= max_positions:
                    break

            if games_processed % 10 == 0:
                print(
                    f"  Processed {games_processed} games, {positions_analyzed} positions..."
                )

    print(f"\nAnalyzed {positions_analyzed} positions from {games_processed} games")
    return plane_usage


def get_random_position_from_pgn(
    pgn_path: str,
    min_move: int = 15,
    max_move: int = 50,
) -> tuple[chess.Board, "PositionHistory", str]:
    """
    Get a random interesting position from a PGN file for GUI display.

    Args:
        pgn_path: Path to PGN file.
        min_move: Minimum move number to consider.
        max_move: Maximum move number to consider.

    Returns:
        Tuple of (board, history, game_info_string).
    """
    import random

    print(f"Loading random position from: {pgn_path}")

    # Collect candidate positions
    candidates = []
    games_scanned = 0
    max_games_to_scan = 100

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while games_scanned < max_games_to_scan:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            games_scanned += 1

            # Get game info
            white = game.headers.get("White", "?")
            black = game.headers.get("Black", "?")
            result = game.headers.get("Result", "?")

            board = game.board()
            history = PositionHistory()
            move_num = 0

            for move in game.mainline_moves():
                history.push(board.copy())
                board.push(move)
                move_num += 1

                # Collect positions in the interesting range
                if min_move <= board.fullmove_number <= max_move:
                    # Create a snapshot of history
                    history_copy = PositionHistory()
                    for b in reversed(history.get_boards()):
                        history_copy.push(b)
                    history_copy.push(board.copy())

                    info = f"{white} vs {black} (move {board.fullmove_number})"
                    candidates.append((board.copy(), history_copy, info))

            if games_scanned % 20 == 0:
                print(
                    f"  Scanned {games_scanned} games, {len(candidates)} candidate positions..."
                )

    if not candidates:
        # Fallback to starting position
        print("No suitable positions found, using starting position")
        board = chess.Board()
        history = PositionHistory()
        history.push(board)
        return board, history, "Starting Position"

    # Pick a random one
    board, history, info = random.choice(candidates)
    print(f"Selected from {len(candidates)} candidates")
    return board, history, info


def print_pgn_summary(plane_usage: dict) -> None:
    """Print summary of PGN analysis."""
    print("\n" + "=" * 90)
    print(
        f"{'Plane':>6} | {'Description':<26} | {'Used %':>8} | {'Avg Sum':>10} | {'Status':<10}"
    )
    print("-" * 90)

    rarely_used = []
    never_used = []

    for i in range(60):
        usage = plane_usage[i]
        desc = PLANE_DESCRIPTIONS.get(i, f"Plane {i}")
        total = usage["positions"]

        if total == 0:
            used_pct = 0
            avg_sum = 0
        else:
            used_pct = usage["nonzero_count"] / total * 100
            avg_sum = usage["total_sum"] / total

        if used_pct == 0:
            status = "NEVER USED"
            never_used.append((i, desc))
        elif used_pct < 5:
            status = "RARE (<5%)"
            rarely_used.append((i, desc, used_pct))
        elif used_pct < 50:
            status = "PARTIAL"
        else:
            status = "OK"

        print(
            f"{i:>6} | {desc:<26} | {used_pct:>7.1f}% | {avg_sum:>10.3f} | {status:<10}"
        )

    print("=" * 90)

    # Summary
    print(f"\nTotal planes: 60")
    print(
        f"Always used (>50%): {sum(1 for u in plane_usage.values() if u['positions'] > 0 and u['nonzero_count']/u['positions'] >= 0.5)}"
    )
    print(
        f"Partial (5-50%): {sum(1 for u in plane_usage.values() if u['positions'] > 0 and 0.05 <= u['nonzero_count']/u['positions'] < 0.5)}"
    )
    print(f"Rare (<5%): {len(rarely_used)}")
    print(f"Never used: {len(never_used)}")

    if rarely_used:
        print(f"\nRarely used planes (< 5%):")
        for idx, desc, pct in rarely_used:
            print(f"  [{idx:2d}] {desc} ({pct:.1f}%)")

    if never_used:
        print(f"\nNever used planes:")
        for idx, desc in never_used:
            print(f"  [{idx:2d}] {desc}")


def show_gui(
    planes: np.ndarray, title: str = "Position", white_to_move: bool = True
) -> None:
    """Display planes using tkinter with a slider to navigate."""
    import tkinter as tk
    from tkinter import ttk

    # Import styles from src/ui
    try:
        from src.ui.styles import COLORS, FONTS, apply_theme, create_panel
    except ImportError:
        # Fallback colors if styles not available
        COLORS = {
            "bg_primary": "#1a1a2e",
            "bg_secondary": "#16213e",
            "bg_tertiary": "#0f3460",
            "text_primary": "#ffffff",
            "text_secondary": "#a0a0a0",
            "text_muted": "#666666",
            "accent": "#e94560",
            "border": "#2a2a4e",
            "light_square": "#eeeed2",
            "dark_square": "#769656",
        }
        FONTS = {
            "title": ("Segoe UI", 24, "bold"),
            "heading": ("Segoe UI", 16, "bold"),
            "subheading": ("Segoe UI", 14, "bold"),
            "body": ("Segoe UI", 12),
            "body_bold": ("Segoe UI", 12, "bold"),
            "small": ("Segoe UI", 10),
            "mono": ("Consolas", 12),
        }
        apply_theme = None

    descriptions = get_plane_descriptions(white_to_move)

    # Create main window
    root = tk.Tk()
    root.title(f"Plane Analyzer - {title}")
    root.configure(bg=COLORS["bg_primary"])
    root.geometry("700x800")

    if apply_theme:
        apply_theme(root)

    # Current plane index
    current_plane = tk.IntVar(value=0)

    # Title
    title_label = tk.Label(
        root,
        text=f"60 Input Planes - {title}",
        font=FONTS["title"],
        fg=COLORS["text_primary"],
        bg=COLORS["bg_primary"],
    )
    title_label.pack(pady=10)

    # Plane description label
    desc_label = tk.Label(
        root,
        text=f"[0] {descriptions.get(0, 'Plane 0')}",
        font=FONTS["heading"],
        fg=COLORS["accent"],
        bg=COLORS["bg_primary"],
    )
    desc_label.pack(pady=5)

    # Stats label
    stats_label = tk.Label(
        root,
        text="",
        font=FONTS["mono"],
        fg=COLORS["text_secondary"],
        bg=COLORS["bg_primary"],
    )
    stats_label.pack(pady=5)

    # Board canvas frame
    board_frame = tk.Frame(root, bg=COLORS["bg_primary"])
    board_frame.pack(pady=10)

    SQUARE_SIZE = 60
    BOARD_SIZE = SQUARE_SIZE * 8

    # Create canvas for the board
    canvas = tk.Canvas(
        board_frame,
        width=BOARD_SIZE + 40,  # Extra space for coordinates
        height=BOARD_SIZE + 40,
        bg=COLORS["bg_primary"],
        highlightthickness=0,
    )
    canvas.pack()

    # Colors for plane values
    def value_to_color(val: float) -> str:
        """Convert plane value to color."""
        if val == 0:
            return "#3a3a5e"  # Dark gray for empty
        elif val == 1.0:
            return "#e94560"  # Accent red for 1.0
        elif val == 0.5:
            return "#f6a623"  # Orange for 0.5
        else:
            # Gradient from blue to cyan for intermediate values
            intensity = int(val * 255)
            return f"#{intensity:02x}{intensity:02x}ff"

    def draw_board(plane_idx: int) -> None:
        """Draw the board with plane values."""
        canvas.delete("all")
        plane = planes[plane_idx]

        # Draw coordinates
        for i in range(8):
            # Rank numbers (left side)
            canvas.create_text(
                15,
                20 + 30 + i * SQUARE_SIZE,
                text=str(8 - i),
                font=FONTS["small"],
                fill=COLORS["text_secondary"],
            )
            # File letters (bottom)
            canvas.create_text(
                40 + i * SQUARE_SIZE + SQUARE_SIZE // 2,
                BOARD_SIZE + 30,
                text=chr(ord("a") + i),
                font=FONTS["small"],
                fill=COLORS["text_secondary"],
            )

        # Draw squares
        for rank in range(8):
            for file in range(8):
                x = 40 + file * SQUARE_SIZE
                y = 20 + rank * SQUARE_SIZE

                # Base square color (checkerboard pattern)
                is_light = (rank + file) % 2 == 0
                base_color = (
                    COLORS["light_square"] if is_light else COLORS["dark_square"]
                )

                # Get plane value at this position
                val = plane[rank, file]

                # Determine fill color based on value
                if val == 0:
                    fill_color = base_color
                else:
                    fill_color = value_to_color(val)

                # Draw square
                canvas.create_rectangle(
                    x,
                    y,
                    x + SQUARE_SIZE,
                    y + SQUARE_SIZE,
                    fill=fill_color,
                    outline=COLORS["border"],
                    width=1,
                )

                # Show value text if non-zero
                if val > 0:
                    # Format value
                    if val == 1.0:
                        val_text = "1"
                    elif val == 0.5:
                        val_text = ".5"
                    else:
                        val_text = f"{val:.2f}"[1:]  # Remove leading 0

                    canvas.create_text(
                        x + SQUARE_SIZE // 2,
                        y + SQUARE_SIZE // 2,
                        text=val_text,
                        font=FONTS["body_bold"],
                        fill=COLORS["text_primary"],
                    )

        # Update description
        desc = descriptions.get(plane_idx, f"Plane {plane_idx}")
        desc_label.config(text=f"[{plane_idx}] {desc}")

        # Update stats
        nonzero = np.count_nonzero(plane)
        total_sum = float(plane.sum())
        max_val = float(plane.max())
        min_val = float(plane.min())

        if nonzero == 0:
            stats_text = "EMPTY PLANE"
        else:
            stats_text = (
                f"NonZero: {nonzero}/64  |  Sum: {total_sum:.3f}  |  Max: {max_val:.2f}"
            )

        stats_label.config(text=stats_text)

    # Slider frame
    slider_frame = tk.Frame(root, bg=COLORS["bg_primary"])
    slider_frame.pack(pady=10, fill=tk.X, padx=20)

    # Plane number display
    plane_num_label = tk.Label(
        slider_frame,
        text="Plane: 0",
        font=FONTS["body_bold"],
        fg=COLORS["text_primary"],
        bg=COLORS["bg_primary"],
        width=12,
    )
    plane_num_label.pack(side=tk.LEFT)

    def on_slider_change(val):
        """Handle slider value change."""
        idx = int(float(val))
        current_plane.set(idx)
        plane_num_label.config(text=f"Plane: {idx}")
        draw_board(idx)

    # Slider
    slider = ttk.Scale(
        slider_frame,
        from_=0,
        to=59,
        orient=tk.HORIZONTAL,
        variable=current_plane,
        command=on_slider_change,
    )
    slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

    # Style the slider
    style = ttk.Style()
    style.configure(
        "TScale",
        background=COLORS["bg_primary"],
        troughcolor=COLORS["bg_tertiary"],
    )

    # Navigation buttons
    nav_frame = tk.Frame(root, bg=COLORS["bg_primary"])
    nav_frame.pack(pady=10)

    def prev_plane():
        idx = max(0, current_plane.get() - 1)
        current_plane.set(idx)
        slider.set(idx)
        draw_board(idx)

    def next_plane():
        idx = min(59, current_plane.get() + 1)
        current_plane.set(idx)
        slider.set(idx)
        draw_board(idx)

    prev_btn = tk.Button(
        nav_frame,
        text="◀ Previous",
        font=FONTS["body_bold"],
        fg=COLORS["text_primary"],
        bg=COLORS["bg_tertiary"],
        activebackground=COLORS["accent"],
        activeforeground=COLORS["text_primary"],
        relief=tk.FLAT,
        padx=20,
        pady=8,
        command=prev_plane,
    )
    prev_btn.pack(side=tk.LEFT, padx=10)

    next_btn = tk.Button(
        nav_frame,
        text="Next ▶",
        font=FONTS["body_bold"],
        fg=COLORS["text_primary"],
        bg=COLORS["bg_tertiary"],
        activebackground=COLORS["accent"],
        activeforeground=COLORS["text_primary"],
        relief=tk.FLAT,
        padx=20,
        pady=8,
        command=next_plane,
    )
    next_btn.pack(side=tk.LEFT, padx=10)

    # Legend
    legend_frame = tk.Frame(root, bg=COLORS["bg_primary"])
    legend_frame.pack(pady=15)

    legend_items = [
        ("#3a3a5e", "0 (empty)"),
        ("#f6a623", "0.5"),
        ("#e94560", "1.0"),
    ]

    legend_label = tk.Label(
        legend_frame,
        text="Legend: ",
        font=FONTS["body"],
        fg=COLORS["text_secondary"],
        bg=COLORS["bg_primary"],
    )
    legend_label.pack(side=tk.LEFT)

    for color, text in legend_items:
        color_box = tk.Canvas(
            legend_frame,
            width=20,
            height=20,
            bg=COLORS["bg_primary"],
            highlightthickness=0,
        )
        color_box.create_rectangle(0, 0, 20, 20, fill=color, outline=COLORS["border"])
        color_box.pack(side=tk.LEFT, padx=(10, 2))

        item_label = tk.Label(
            legend_frame,
            text=text,
            font=FONTS["small"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_primary"],
        )
        item_label.pack(side=tk.LEFT, padx=(0, 10))

    # Keyboard bindings
    def on_key(event):
        if event.keysym in ("Left", "Up"):
            prev_plane()
        elif event.keysym in ("Right", "Down"):
            next_plane()
        elif event.keysym == "Home":
            current_plane.set(0)
            slider.set(0)
            draw_board(0)
        elif event.keysym == "End":
            current_plane.set(59)
            slider.set(59)
            draw_board(59)

    root.bind("<Key>", on_key)

    # Initial draw
    draw_board(0)

    # Start main loop
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize the 60 input planes"
    )
    parser.add_argument(
        "--fen",
        type=str,
        default=None,
        help="FEN string to analyze (default: starting position)",
    )
    parser.add_argument(
        "--pgn",
        type=str,
        default=None,
        help="PGN file to analyze (aggregates statistics across games)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=1000,
        help="Max positions to analyze from PGN (default: 1000)",
    )
    parser.add_argument(
        "--gui", action="store_true", help="Show matplotlib visualization"
    )
    parser.add_argument(
        "--show-planes", action="store_true", help="Show each plane as ASCII grid"
    )
    parser.add_argument(
        "--moves",
        type=str,
        default=None,
        help="Comma-separated moves to play from start (e.g., 'e4,e5,Nf3,Nc6')",
    )

    args = parser.parse_args()

    # Analyze PGN file
    if args.pgn:
        if args.gui:
            # GUI mode: pick a random interesting position from PGN
            board, history, game_info = get_random_position_from_pgn(args.pgn)
            title = f"PGN: {game_info}"
            white_to_move = board.turn == chess.WHITE

            print(f"\nSelected position: {title}")
            print(f"Board:\n{board}\n")
            print(f"FEN: {board.fen()}")
            print(f"Turn: {'White' if white_to_move else 'Black'}")

            planes = history.encode()
            stats = analyze_planes(
                planes, verbose=args.show_planes, white_to_move=white_to_move
            )
            print_summary(stats)
            show_gui(planes, title, white_to_move)
            return

        # Non-GUI: aggregate statistics
        plane_usage = analyze_from_pgn(args.pgn, args.max_positions)
        print_pgn_summary(plane_usage)
        return

    # Single position analysis
    history = PositionHistory()

    if args.fen:
        board = chess.Board(args.fen)
        history.push(board)
        title = f"FEN: {args.fen[:40]}..."
    elif args.moves:
        board = chess.Board()
        history.push(board.copy())  # Start position
        moves = args.moves.split(",")
        for move_str in moves:
            move = board.parse_san(move_str.strip())
            board.push(move)
            history.push(board.copy())  # Each position after a move
        title = f"After: {args.moves}"
    else:
        board = chess.Board()
        history.push(board)
        title = "Starting Position"

    print(f"\nAnalyzing: {title}")
    print(f"Board:\n{board}\n")
    print(f"FEN: {board.fen()}")
    white_to_move = board.turn == chess.WHITE
    print(
        f"Turn: {'White' if white_to_move else 'Black'} (planes encoded from {'White' if white_to_move else 'Black'}'s perspective)"
    )

    # Encode with history
    planes = history.encode()
    print(f"Encoded shape: {planes.shape}")

    # Analyze
    stats = analyze_planes(
        planes, verbose=args.show_planes, white_to_move=white_to_move
    )
    print_summary(stats)

    # GUI
    if args.gui:
        show_gui(planes, title, white_to_move)


if __name__ == "__main__":
    main()
