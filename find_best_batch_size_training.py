#!/usr/bin/env python3
"""
Find the optimal batch_size for RL training (self-play + network training).

Tests batch sizes while simulating MCTS memory usage to find
the best batch_size for the training phase.
"""

import sys
import time
import gc

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

sys.path.insert(0, "src")
from alphazero import DualHeadNetwork, get_device, supports_mixed_precision
from alphazero.move_encoding import MOVE_ENCODING_SIZE


def test_mcts_batch_size(network, mcts_batch_size, device, use_amp, num_iterations=50):
    """Test MCTS inference throughput for a given batch size."""
    network.eval()
    num_planes = network.num_input_planes

    # Simulate MCTS batch inference
    dummy_states = torch.randn(mcts_batch_size, num_planes, 8, 8, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            if use_amp:
                with autocast(device_type="cuda"):
                    network(dummy_states)
            else:
                network(dummy_states)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            if use_amp:
                with autocast(device_type="cuda"):
                    network(dummy_states)
            else:
                network(dummy_states)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    throughput = (mcts_batch_size * num_iterations) / elapsed
    latency = (elapsed / num_iterations) * 1000  # ms
    return throughput, latency


def simulate_mcts_memory(network, device, num_simulations=100, batch_size=32):
    """Simulate MCTS memory usage by allocating typical MCTS buffers."""
    network.eval()

    # MCTS stores node data: visits, values, priors, children
    # Simulate memory for a typical search tree
    num_nodes = num_simulations * 20  # ~20 nodes per simulation on average

    # Node storage (visits, values, etc.)
    visits = torch.zeros(num_nodes, dtype=torch.int32, device=device)
    values = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    priors = torch.zeros(
        num_nodes, MOVE_ENCODING_SIZE, dtype=torch.float32, device=device
    )

    # Batch inference during MCTS
    num_planes = network.num_input_planes
    mcts_batch = torch.randn(batch_size, num_planes, 8, 8, device=device)

    with torch.no_grad():
        if device.type == "cuda":
            with autocast(device_type="cuda"):
                policy, value, _ = network(mcts_batch)
        else:
            policy, value, _ = network(mcts_batch)

    return visits, values, priors, mcts_batch, policy, value


def test_batch_size_with_mcts(
    network,
    batch_size,
    device,
    use_amp,
    scaler,
    mcts_simulations=100,
    mcts_batch_size=32,
    num_iterations=10,
):
    """Test a batch size while MCTS buffers are allocated."""

    # First allocate MCTS memory (simulating self-play)
    mcts_data = simulate_mcts_memory(network, device, mcts_simulations, mcts_batch_size)

    # Now test training with MCTS memory still allocated
    num_planes = network.num_input_planes
    dummy_states = torch.randn(batch_size, num_planes, 8, 8, device=device)
    dummy_policies = torch.zeros(batch_size, MOVE_ENCODING_SIZE, device=device)
    dummy_policies[:, 0] = 1.0
    dummy_values = torch.randn(batch_size, device=device)
    dummy_phases = torch.randint(0, 3, (batch_size,), device=device)

    network.train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.01)

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        if use_amp:
            with autocast(device_type="cuda"):
                pred_p, pred_v, pred_phase = network(dummy_states)
                loss = (
                    nn.functional.cross_entropy(pred_p, dummy_policies)
                    + nn.functional.mse_loss(pred_v, dummy_values)
                    + 0.1 * nn.functional.cross_entropy(pred_phase, dummy_phases)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_p, pred_v, pred_phase = network(dummy_states)
            loss = (
                nn.functional.cross_entropy(pred_p, dummy_policies)
                + nn.functional.mse_loss(pred_v, dummy_values)
                + 0.1 * nn.functional.cross_entropy(pred_phase, dummy_phases)
            )
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        optimizer.zero_grad()
        if use_amp:
            with autocast(device_type="cuda"):
                pred_p, pred_v, pred_phase = network(dummy_states)
                loss = (
                    nn.functional.cross_entropy(pred_p, dummy_policies)
                    + nn.functional.mse_loss(pred_v, dummy_values)
                    + 0.1 * nn.functional.cross_entropy(pred_phase, dummy_phases)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_p, pred_v, pred_phase = network(dummy_states)
            loss = (
                nn.functional.cross_entropy(pred_p, dummy_policies)
                + nn.functional.mse_loss(pred_v, dummy_values)
                + 0.1 * nn.functional.cross_entropy(pred_phase, dummy_phases)
            )
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Clean up MCTS data
    del mcts_data

    throughput = (batch_size * num_iterations) / elapsed
    return throughput, elapsed / num_iterations


def main():
    print("=" * 60)
    print("GPU Batch Size Optimizer for NeuralMate2 RL Training")
    print("=" * 60)

    device = get_device()
    use_amp = supports_mixed_precision()

    print(f"\nDevice: {device}")
    print(f"Mixed Precision (AMP): {use_amp}")

    if device.type != "cuda":
        print("\nERROR: This script requires a CUDA GPU.")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_mem:.1f} GB")

    # =========================================================
    # PART 1: Find optimal mcts_batch_size (inference)
    # =========================================================
    print("\n" + "=" * 60)
    print("PART 1: Finding optimal mcts_batch_size (MCTS inference)")
    print("=" * 60)

    mcts_batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    mcts_results = []

    for mcts_bs in mcts_batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()

        network = DualHeadNetwork()
        network.to(device)

        try:
            throughput, latency = test_mcts_batch_size(
                network, mcts_bs, device, use_amp
            )

            mem_used = torch.cuda.max_memory_allocated() / (1024**3)
            mem_pct = (mem_used / gpu_mem) * 100

            mcts_results.append(
                {
                    "batch_size": mcts_bs,
                    "throughput": throughput,
                    "latency": latency,
                    "mem_used": mem_used,
                    "mem_pct": mem_pct,
                }
            )

            print(
                f"mcts_batch_size={mcts_bs:4d} | {throughput:7.0f} pos/s | "
                f"{latency:5.2f}ms/batch | VRAM: {mem_used:.1f}GB ({mem_pct:.0f}%)"
            )

        except torch.cuda.OutOfMemoryError:
            print(f"mcts_batch_size={mcts_bs:4d} | OUT OF MEMORY")
            break
        finally:
            del network
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # Find best mcts_batch_size
    if mcts_results:
        best_mcts = max(mcts_results, key=lambda x: x["throughput"])
        recommended_mcts_batch_size = best_mcts["batch_size"]
        print(
            f"\n→ Best mcts_batch_size: {recommended_mcts_batch_size} ({best_mcts['throughput']:.0f} pos/s)"
        )
    else:
        recommended_mcts_batch_size = 32
        print("\n→ Using default mcts_batch_size: 32")

    # =========================================================
    # PART 2: Find optimal training batch_size
    # =========================================================
    print("\n" + "=" * 60)
    print("PART 2: Finding optimal training batch_size")
    print("=" * 60)

    # RL training config values
    mcts_simulations = 100

    print("\nSimulating MCTS with:")
    print(f"  num_simulations: {mcts_simulations}")
    print(f"  mcts_batch_size: {recommended_mcts_batch_size}")

    batch_sizes = [
        64,
        128,
        192,
        256,
        384,
        512,
        640,
        768,
        896,
        1024,
        1152,
        1280,
    ]

    print(f"\nTesting batch sizes: {batch_sizes}")
    print("-" * 60)

    results = []
    scaler = GradScaler("cuda") if use_amp else None

    for batch_size in batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()

        network = DualHeadNetwork()
        network.to(device)

        try:
            throughput, time_per_batch = test_batch_size_with_mcts(
                network,
                batch_size,
                device,
                use_amp,
                scaler,
                mcts_simulations,
                recommended_mcts_batch_size,
            )

            mem_used = torch.cuda.max_memory_allocated() / (1024**3)
            mem_pct = (mem_used / gpu_mem) * 100

            results.append(
                {
                    "batch_size": batch_size,
                    "throughput": throughput,
                    "time_per_batch": time_per_batch,
                    "mem_used": mem_used,
                    "mem_pct": mem_pct,
                }
            )

            print(
                f"batch_size={batch_size:4d} | {throughput:7.0f} samples/s | "
                f"{time_per_batch*1000:6.1f}ms/batch | "
                f"VRAM: {mem_used:.1f}GB ({mem_pct:.0f}%)"
            )

        except torch.cuda.OutOfMemoryError:
            print(f"batch_size={batch_size:4d} | OUT OF MEMORY")
            break
        except Exception as e:
            print(f"batch_size={batch_size:4d} | ERROR: {e}")
            break
        finally:
            del network
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    if not results:
        print("\nNo valid batch sizes found!")
        return 1

    best = max(results, key=lambda x: x["throughput"])

    safe_results = [r for r in results if r["mem_pct"] < 85]  # More conservative for RL
    if safe_results:
        largest_safe = max(safe_results, key=lambda x: x["batch_size"])
    else:
        largest_safe = results[0]

    print("\n" + "=" * 60)
    print("RESULTS (with MCTS memory overhead)")
    print("=" * 60)

    print(
        f"\nBest throughput:     batch_size={best['batch_size']} ({best['throughput']:.0f} samples/s)"
    )
    print(
        f"Largest safe (<85%): batch_size={largest_safe['batch_size']} ({largest_safe['mem_pct']:.0f}% VRAM)"
    )

    recommended = min(best["batch_size"], largest_safe["batch_size"])

    print(f"\n{'='*60}")
    print("RECOMMENDED for RL training:")
    print(f"  batch_size      = {recommended}")
    print(f"  mcts_batch_size = {recommended_mcts_batch_size}")
    print(f"{'='*60}")

    print("\nTo use this, update config/config.json [training] section:")
    print(f'  "batch_size": {recommended},')
    print(f'  "mcts_batch_size": {recommended_mcts_batch_size}')

    return 0


if __name__ == "__main__":
    sys.exit(main())
