"""Test: MCTS Behavior."""

import time
import numpy as np
import chess

from alphazero.move_encoding import encode_move, decode_move
from alphazero.mcts import MCTS
from ..core import (
    Colors,
    TestResults,
    header,
    subheader,
    ok,
    fail,
    warn,
    encode_for_network,
    get_history_length,
)


def test_mcts_behavior(network, results: TestResults):
    """Test MCTS behavior on a simple position."""
    print(header("TEST: MCTS Behavior"))

    fen = "rnb1kbnr/pppppppp/8/8/3q4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1"
    board = chess.Board(fen)

    print(subheader("Position: Black Queen Hanging on d4"))
    print(board)
    print("\n  This is a trivial position - capturing the queen is clearly best.")

    # First, get raw policy
    state = encode_for_network(board, network)
    raw_policy, raw_value = network.predict_single(state)

    raw_top_idx = np.argmax(raw_policy)
    raw_top_move = decode_move(raw_top_idx, board)

    print(subheader("Raw Network Output (no search)"))
    print(f"  Value: {raw_value:+.4f}")
    print(
        f"  Top move: {raw_top_move.uci() if raw_top_move else 'None'} ({raw_policy[raw_top_idx]*100:.1f}%)"
    )

    # Find queen capture
    capture_move = None
    for move in board.legal_moves:
        if board.is_capture(move) and board.piece_at(move.to_square):
            if board.piece_at(move.to_square).piece_type == chess.QUEEN:
                capture_move = move
                break

    if capture_move:
        capture_idx = encode_move(capture_move)
        capture_prob = raw_policy[capture_idx] if capture_idx else 0
        print(f"  Queen capture ({capture_move.uci()}): {capture_prob*100:.1f}%")

        # Check value AFTER capturing the queen
        board_after = board.copy()
        board_after.push(capture_move)
        state_after = encode_for_network(board_after, network)
        _, value_after_capture = network.predict_single(state_after)
        print(f"  Value after capture: {value_after_capture:+.4f} (should be negative = bad for Black)")

        # Check value after b2b3 for comparison
        board_b3 = board.copy()
        board_b3.push(chess.Move.from_uci("b2b3"))
        state_b3 = encode_for_network(board_b3, network)
        _, value_after_b3 = network.predict_single(state_b3)
        print(f"  Value after b2b3:    {value_after_b3:+.4f} (for comparison)")

        results.add_diagnostic("mcts", "raw_capture_prob", float(capture_prob))
        results.add_diagnostic("mcts", "value_after_capture", float(value_after_capture))
        results.add_diagnostic(
            "mcts",
            "raw_capture_is_top",
            raw_top_move == capture_move if raw_top_move else False,
        )

    # Now test MCTS
    print(subheader("MCTS Search (100 simulations)"))

    history_length = get_history_length(network)
    mcts = MCTS(
        network=network, c_puct=1.0, num_simulations=100, history_length=history_length
    )
    mcts.temperature = 0.1

    start = time.time()
    policy = mcts.search(board, add_noise=False)
    elapsed = time.time() - start

    print(f"  Search time: {elapsed*1000:.0f}ms")
    print(f"  Simulations per second: {100/elapsed:.0f}")

    top_indices = np.argsort(policy)[::-1][:5]

    print(f"\n  {'Rank':<6} {'Move':<8} {'Visits':>10} {'Type':<20}")
    print("  " + "-" * 50)

    found_capture = False
    capture_rank = None
    capture_visits = None

    for i, idx in enumerate(top_indices):
        move = decode_move(idx, board)
        visits = policy[idx]
        if move:
            is_capture = board.is_capture(move)
            piece = board.piece_at(move.to_square) if is_capture else None

            if piece and piece.piece_type == chess.QUEEN:
                move_type = "*** QUEEN CAPTURE ***"
                if i == 0:
                    found_capture = True
                if capture_rank is None:
                    capture_rank = i + 1
                    capture_visits = visits
                color = Colors.GREEN
            elif is_capture:
                move_type = "(capture)"
                color = Colors.YELLOW
            else:
                move_type = ""
                color = ""

            end_color = Colors.ENDC if color else ""
            print(
                f"  {color}{i+1:<6} {move.uci():<8} {visits*100:>9.1f}% {move_type}{end_color}"
            )

    results.add_diagnostic("mcts", "search_time_ms", elapsed * 1000)
    results.add_diagnostic("mcts", "capture_rank", capture_rank)
    results.add_diagnostic(
        "mcts", "capture_visits", float(capture_visits) if capture_visits else 0
    )
    results.add_diagnostic("mcts", "top_move_visits", float(policy[top_indices[0]]))

    # Analysis
    print(subheader("MCTS Analysis"))

    if found_capture:
        print(f"  {ok('MCTS correctly prioritizes queen capture')}")
    else:
        if capture_rank:
            print(
                f"  {warn(f'Queen capture at rank {capture_rank} ({capture_visits*100:.1f}%)')}"
            )
            print(
                f"  Top move ({decode_move(top_indices[0], board).uci()}) has {policy[top_indices[0]]*100:.1f}%"
            )
        else:
            print(f"  {fail('Queen capture not in top 5!')}")

        results.add_issue(
            "CRITICAL",
            "mcts",
            "MCTS fails to find obvious queen capture",
            f"Capture at rank {capture_rank}, value head may be broken",
        )

        # Diagnose why
        print(f"\n  {Colors.BOLD}Diagnosis:{Colors.ENDC}")
        if abs(raw_value) < 0.1:
            print(f"    • Value head output near zero ({raw_value:+.4f})")
            print(f"    • MCTS can't distinguish good from bad positions")
            print(f"    • Recommendation: Fix value head training")
        if capture_prob < 0.3:
            print(f"    • Policy gives low prob to capture ({capture_prob*100:.1f}%)")
            print(f"    • MCTS explores other moves more initially")

    results.add("MCTS Behavior", found_capture, 1.0 if found_capture else 0.0, 1.0)
    results.add_timing("MCTS (100 sims)", elapsed)

    return 1.0 if found_capture else 0.0
