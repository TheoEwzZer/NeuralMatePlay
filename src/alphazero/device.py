"""Device management for PyTorch (CPU/GPU selection)."""

import torch


_device: torch.device | None = None


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch operations.

    Returns:
        torch.device: CUDA if available, otherwise CPU.
    """
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")
    return _device


def set_device(device: str | torch.device) -> None:
    """
    Manually set the device to use.

    Args:
        device: Device string ("cuda", "cpu") or torch.device object.
    """
    global _device
    if isinstance(device, str):
        _device = torch.device(device)
    else:
        _device = device


def supports_mixed_precision() -> bool:
    """
    Check if the current device supports mixed precision (FP16) training.

    Returns:
        True if GPU supports FP16, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    # Check compute capability (need >= 7.0 for good FP16 support)
    capability = torch.cuda.get_device_capability()
    return capability[0] >= 7


def get_device_info() -> dict:
    """
    Get information about the current device.

    Returns:
        Dictionary with device information.
    """
    device = get_device()
    info = {
        "device": str(device),
        "type": device.type,
    }

    if device.type == "cuda":
        info.update(
            {
                "name": torch.cuda.get_device_name(),
                "memory_total": torch.cuda.get_device_properties(0).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(),
                "compute_capability": torch.cuda.get_device_capability(),
                "supports_fp16": supports_mixed_precision(),
            }
        )

    return info
