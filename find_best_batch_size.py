#!/usr/bin/env python3
"""
Find the optimal batch_size for your GPU.

Tests increasing batch sizes until CUDA runs out of memory,
then reports the best performing batch size.
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


def test_batch_size(network, batch_size, device, use_amp, scaler, num_iterations=10):
    """Test a batch size and return throughput (samples/sec)."""

    num_planes = network.num_input_planes
    dummy_states = torch.randn(batch_size, num_planes, 8, 8, device=device)
    dummy_policies = torch.zeros(batch_size, MOVE_ENCODING_SIZE, device=device)
    dummy_policies[:, 0] = 1.0
    dummy_values = torch.randn(batch_size, device=device)
    dummy_phases = torch.randint(0, 3, (batch_size,), device=device)

    network.train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.01)

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

    throughput = (batch_size * num_iterations) / elapsed
    return throughput, elapsed / num_iterations


def main():
    print("=" * 60)
    print("GPU Batch Size Optimizer for NeuralMate2")
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
            throughput, time_per_batch = test_batch_size(
                network, batch_size, device, use_amp, scaler
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

    safe_results = [r for r in results if r["mem_pct"] < 90]
    if safe_results:
        largest_safe = max(safe_results, key=lambda x: x["batch_size"])
    else:
        largest_safe = results[0]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"\nBest throughput:     batch_size={best['batch_size']} ({best['throughput']:.0f} samples/s)"
    )
    print(
        f"Largest safe (<90%): batch_size={largest_safe['batch_size']} ({largest_safe['mem_pct']:.0f}% VRAM)"
    )

    recommended = min(best["batch_size"], largest_safe["batch_size"])

    print(f"\n{'='*60}")
    print(f"RECOMMENDED: batch_size = {recommended}")
    print(f"{'='*60}")

    print("\nTo use this, update your config/config.json:")
    print(f'  "batch_size": {recommended}')

    return 0


if __name__ == "__main__":
    sys.exit(main())
