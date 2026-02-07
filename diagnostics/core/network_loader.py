"""Network loading and architecture analysis utilities."""

from __future__ import annotations

import os
import re
from typing import Any, TYPE_CHECKING

import numpy as np

from .colors import Colors, header, subheader
from .results import TestResults

if TYPE_CHECKING:
    from alphazero import DualHeadNetwork


def load_network_from_path(path: str) -> DualHeadNetwork | None:
    """Load a network from a specific path."""
    from alphazero import DualHeadNetwork

    if not os.path.exists(path):
        print(f"{Colors.RED}File not found: {path}{Colors.ENDC}")
        return None
    return DualHeadNetwork.load(path)


def get_available_checkpoints(
    checkpoint_dir: str, checkpoint_type: str = "train"
) -> list[tuple[int, str]]:
    """
    Get all available checkpoints sorted by iteration/epoch.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        checkpoint_type: "train" for iteration_X, "pretrain" for pretrained_best_X.

    Returns:
        List of (number, path) tuples sorted by number.
    """
    if not os.path.exists(checkpoint_dir):
        return []

    pattern: re.Pattern[str]
    if checkpoint_type == "pretrain":
        pattern = re.compile(r"pretrained_best_(\d+)_network\.pt")
    else:
        pattern = re.compile(r"iteration_(\d+)_network\.pt")

    checkpoints: list[tuple[int, str]] = []

    for filename in os.listdir(checkpoint_dir):
        match: re.Match[str] | None = pattern.match(filename)
        if match:
            num: int = int(match.group(1))
            path: str = os.path.join(checkpoint_dir, filename)
            checkpoints.append((num, path))

    return sorted(checkpoints, key=lambda x: x[0])


def load_latest_network(
    checkpoint_dir: str = "checkpoints", checkpoint_type: str = "train"
) -> tuple[DualHeadNetwork, int] | None:
    """Load the most recent checkpoint from the specified directory."""
    from alphazero import DualHeadNetwork

    checkpoints: list[tuple[int, str]] = get_available_checkpoints(
        checkpoint_dir, checkpoint_type
    )

    if not checkpoints:
        type_name: str = "pretrain" if checkpoint_type == "pretrain" else "train"
        print(
            f"{Colors.RED}No {type_name} checkpoints found in {checkpoint_dir}!{Colors.ENDC}"
        )
        return None

    latest_num: int
    latest_path: str
    latest_num, latest_path = checkpoints[-1]
    type_label: str = "best" if checkpoint_type == "pretrain" else "iteration"
    print(f"Loading network from {type_label} {latest_num}...")
    return DualHeadNetwork.load(latest_path), latest_num


def analyze_network_architecture(
    network: DualHeadNetwork, results: TestResults
) -> None:
    """Analyze and log network architecture details."""
    print(header("NETWORK ARCHITECTURE ANALYSIS"))

    config: dict[str, Any] = network.get_config()

    print(subheader("Configuration"))
    print(f"  Input planes:       {network.num_input_planes}")
    print(f"  Filters:            {config['num_filters']}")
    print(f"  Residual blocks:    {config['num_residual_blocks']}")
    print(f"  Policy size:        {config['policy_size']}")

    total_params: int = sum(p.numel() for p in network.parameters())
    trainable_params: int = sum(
        p.numel() for p in network.parameters() if p.requires_grad
    )

    print(subheader("Parameters"))
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    print(f"  Model size (approx):   {total_params * 4 / 1024 / 1024:.1f} MB")

    results.add_diagnostic("architecture", "input_planes", network.num_input_planes)
    results.add_diagnostic("architecture", "filters", config["num_filters"])
    results.add_diagnostic(
        "architecture", "residual_blocks", config["num_residual_blocks"]
    )
    results.add_diagnostic("architecture", "policy_size", config["policy_size"])
    results.add_diagnostic("architecture", "total_params", total_params)
    results.add_diagnostic("architecture", "trainable_params", trainable_params)

    print(subheader("Weight Statistics"))

    weight_stats: dict[str, dict[str, float]] = {}
    for name, param in network.named_parameters():
        if param.requires_grad:
            data: np.ndarray = param.data.cpu().numpy().flatten()
            stats: dict[str, float] = {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "zeros_pct": float(np.sum(np.abs(data) < 1e-6) / len(data) * 100),
            }
            weight_stats[name] = stats

    key_layers: list[str] = [
        # Policy head
        "policy_head.fc.weight",
        "policy_head.fc.bias",
        # WDL head (win/draw/loss)
        "wdl_head.fc1.weight",
        "wdl_head.fc1.bias",
        "wdl_head.fc2.weight",
        "wdl_head.fc2.bias",
    ]

    print(f"  {'Layer':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("  " + "-" * 70)

    for layer in key_layers:
        if layer in weight_stats:
            s: dict[str, float] = weight_stats[layer]
            print(
                f"  {layer:<30} {s['mean']:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f}"
            )
            results.add_diagnostic("weights", f"{layer}_mean", s["mean"])
            results.add_diagnostic("weights", f"{layer}_std", s["std"])

    for layer, stats in weight_stats.items():
        if stats["std"] < 0.001 and layer != "wdl_head.fc2.bias":
            results.add_issue(
                "HIGH",
                "weights",
                f"Layer '{layer}' has very low std ({stats['std']:.6f})",
                "Weights may not be learning or are collapsed",
            )
        if stats["zeros_pct"] > 50:
            results.add_issue(
                "MEDIUM",
                "weights",
                f"Layer '{layer}' has {stats['zeros_pct']:.1f}% near-zero weights",
                "May indicate dead neurons or insufficient training",
            )
