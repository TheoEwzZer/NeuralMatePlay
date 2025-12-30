"""Test: Inference Speed."""

import time
import numpy as np
import chess

from ..core import (
    TestResults,
    header,
    subheader,
    ok,
    warn,
    encode_for_network,
)


def test_inference_speed(network, results: TestResults):
    """Measure network inference speed."""
    print(header("TEST: Inference Speed"))

    board = chess.Board()
    state = encode_for_network(board, network)

    # Warmup
    print("  Warming up (5 iterations)...")
    for _ in range(5):
        network.predict_single(state)

    # Measure single inference
    n_single = 100
    start = time.time()
    for _ in range(n_single):
        network.predict_single(state)
    single_elapsed = time.time() - start
    single_avg = single_elapsed / n_single

    # Measure batch inference if available
    batch_size = 32
    states = np.stack([state] * batch_size)

    n_batch = 50
    start = time.time()
    for _ in range(n_batch):
        network.predict_batch(states)
    batch_elapsed = time.time() - start
    batch_avg = batch_elapsed / n_batch
    per_sample_batch = batch_avg / batch_size

    print(subheader("Performance Results"))
    print(f"  {'Metric':<35} {'Value':>15}")
    print("  " + "-" * 55)
    print(f"  {'Single inference (avg of 100)':<35} {single_avg*1000:>12.2f} ms")
    print(f"  {'Single inferences per second':<35} {1/single_avg:>12.0f}")
    print(f"  {'Batch inference (32 samples)':<35} {batch_avg*1000:>12.2f} ms")
    print(f"  {'Per-sample in batch':<35} {per_sample_batch*1000:>12.3f} ms")
    print(f"  {'Batch speedup factor':<35} {single_avg/per_sample_batch:>12.1f}x")

    results.add_diagnostic("inference", "single_avg_ms", single_avg * 1000)
    results.add_diagnostic("inference", "batch_avg_ms", batch_avg * 1000)
    results.add_diagnostic("inference", "inferences_per_sec", 1 / single_avg)
    results.add_diagnostic("inference", "batch_speedup", single_avg / per_sample_batch)

    # Benchmark assessment
    if single_avg < 0.005:
        status = "Excellent"
        passed = True
    elif single_avg < 0.02:
        status = "Good"
        passed = True
    elif single_avg < 0.05:
        status = "Acceptable"
        passed = True
    else:
        status = "Slow"
        passed = False
        results.add_issue(
            "MEDIUM",
            "performance",
            f"Slow inference: {single_avg*1000:.1f}ms per sample",
            "May impact MCTS search speed",
        )

    print(f"\n  {ok(f'Status: {status}') if passed else warn(f'Status: {status}')}")

    # Calculate score based on speed (faster = better)
    # Excellent (<5ms) = 1.0, Good (<20ms) = 0.8, Acceptable (<50ms) = 0.6, Slow = 0.4
    if single_avg < 0.005:
        score = 1.0
    elif single_avg < 0.02:
        score = 0.8
    elif single_avg < 0.05:
        score = 0.6
    else:
        score = 0.4

    results.add("Inference Speed", passed, score, 1.0)
    results.add_timing("Single inference", single_avg)
    results.add_timing("Batch inference (32)", batch_avg)

    return score
