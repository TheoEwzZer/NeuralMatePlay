"""
Neural network architecture for AlphaZero-style chess.

Architecture: SE-ResNet with Spatial Attention
- 72 input planes (48 history + 12 metadata + 8 semantic + 4 tactical)
- Squeeze-and-Excitation blocks for channel recalibration
- Spatial attention in final blocks for global awareness
- Group Normalization for stability with small batches
- GELU activation for smoother gradients
- WDL head (Win/Draw/Loss probabilities)
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .device import get_device, supports_mixed_precision
from .move_encoding import MOVE_ENCODING_SIZE


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel recalibration.

    Adaptively weights channel importance based on global information.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Squeeze: global average pooling
        y = self.squeeze(x).view(b, c)
        # Excite: FC -> GELU -> FC -> Sigmoid
        y = self.excite(y).view(b, c, 1, 1)
        # Scale channels
        return x * y


class SEResBlock(nn.Module):
    """
    Squeeze-and-Excitation Residual Block.

    Conv -> GroupNorm -> GELU -> Conv -> GroupNorm -> SE -> Add -> GELU
    """

    def __init__(self, channels: int, se_reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, channels)
        self.se = SEBlock(channels, se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = F.gelu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.se(out)

        out = out + residual
        out = F.gelu(out)

        return out


class SpatialAttention(nn.Module):
    """
    Spatial attention module for global board awareness.

    Uses multi-head self-attention over spatial positions.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.gn = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x

        # Compute Q, K, V
        qkv = self.qkv(x)  # (b, 3*c, h, w)
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, h * w)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, b, heads, hw, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention with numerical stability
        scale = self.head_dim**-0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        # Subtract max for numerical stability before softmax
        attn = attn - attn.max(dim=-1, keepdim=True).values
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = attn @ v  # (b, heads, hw, head_dim)
        out = out.permute(0, 1, 3, 2)  # (b, heads, head_dim, hw)
        out = out.reshape(b, c, h, w)

        out = self.proj(out)
        out = self.gn(out)
        out = out + residual

        return out


class SEResBlockWithAttention(nn.Module):
    """
    SE-ResBlock followed by spatial attention.
    """

    def __init__(self, channels: int, se_reduction: int = 8, num_heads: int = 4):
        super().__init__()
        self.resblock = SEResBlock(channels, se_reduction)
        self.attention = SpatialAttention(channels, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resblock(x)
        x = self.attention(x)
        return x


class PolicyHead(nn.Module):
    """
    Policy head: outputs probability distribution over moves.
    """

    def __init__(self, in_channels: int, policy_size: int = MOVE_ENCODING_SIZE):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 32, 1, bias=False)
        self.gn = nn.GroupNorm(8, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32 * 8 * 8, policy_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        x = F.gelu(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        # Clamp logits to prevent softmax overflow (|logit| > 88 causes overflow in float16)
        x = torch.clamp(x, min=-50, max=50)
        return x  # Clamped logits (softmax applied externally)


class WDLHead(nn.Module):
    """
    Win/Draw/Loss head: outputs probability distribution over outcomes.

    Predicts:
    - P(win): probability of winning
    - P(draw): probability of drawing
    - P(loss): probability of losing

    The expected value can be computed as: E[V] = P(win) - P(loss)

    This provides better calibration, especially for positions with high draw probability.
    Used by Leela Chess Zero with proven success.
    """

    def __init__(self, in_channels: int, hidden_size: int = 512):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 8, 1, bias=False)
        self.gn = nn.GroupNorm(4, 8)
        self.fc1 = nn.Linear(8 * 8 * 8, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, 3)  # 3 classes: win, draw, loss

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for balanced initial predictions."""
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor from backbone.

        Returns:
            WDL logits of shape (batch, 3) - [win, draw, loss]
        """
        x = self.conv(x)
        x = self.gn(x)
        x = F.gelu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # logits, softmax applied externally

    @staticmethod
    def logits_to_value(wdl_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert WDL logits to expected value in [-1, 1].

        Args:
            wdl_logits: WDL logits of shape (..., 3)

        Returns:
            Expected value: P(win) - P(loss), shape (...)
        """
        wdl_probs = F.softmax(wdl_logits, dim=-1)
        # value = P(win) * 1 + P(draw) * 0 + P(loss) * (-1) = P(win) - P(loss)
        return wdl_probs[..., 0] - wdl_probs[..., 2]

    @staticmethod
    def probs_to_value(wdl_probs: torch.Tensor) -> torch.Tensor:
        """
        Convert WDL probabilities to expected value.

        Args:
            wdl_probs: WDL probabilities of shape (..., 3)

        Returns:
            Expected value: P(win) - P(loss), shape (...)
        """
        return wdl_probs[..., 0] - wdl_probs[..., 2]


class DualHeadNetwork(nn.Module):
    """
    AlphaZero-style network with SE-ResNet and attention.

    Architecture:
    - Input: 72 planes (48 history + 12 metadata + 8 semantic + 4 tactical)
    - 10 SE-ResBlocks
    - 2 SE-ResBlocks with spatial attention
    - Policy head (move probabilities)
    - WDL head (win/draw/loss probabilities)
    """

    def __init__(
        self,
        num_input_planes: int = 72,
        num_filters: int = 192,
        num_residual_blocks: int = 12,
        policy_size: int = MOVE_ENCODING_SIZE,
        se_reduction: int = 8,
        attention_heads: int = 4,
    ):
        super().__init__()

        self._num_input_planes = num_input_planes
        self._num_filters = num_filters
        self._num_residual_blocks = num_residual_blocks
        self._policy_size = policy_size
        self._se_reduction = se_reduction
        self._attention_heads = attention_heads

        # Input convolution
        self.input_conv = nn.Conv2d(
            num_input_planes, num_filters, 3, padding=1, bias=False
        )
        self.input_gn = nn.GroupNorm(8, num_filters)

        # Residual tower
        self.res_blocks = nn.ModuleList()

        # First N-2 blocks: SE-ResBlocks only
        num_plain_blocks = max(0, num_residual_blocks - 2)
        for _ in range(num_plain_blocks):
            self.res_blocks.append(SEResBlock(num_filters, se_reduction))

        # Last 2 blocks: SE-ResBlocks with attention
        num_attention_blocks = min(2, num_residual_blocks)
        for _ in range(num_attention_blocks):
            self.res_blocks.append(
                SEResBlockWithAttention(num_filters, se_reduction, attention_heads)
            )

        # Output heads
        self.policy_head = PolicyHead(num_filters, policy_size)
        self.wdl_head = WDLHead(num_filters)

        # Move to device
        self.to(get_device())

    @property
    def num_input_planes(self) -> int:
        """Number of input planes."""
        return self._num_input_planes

    @property
    def num_filters(self) -> int:
        """Number of filters in residual blocks."""
        return self._num_filters

    @property
    def num_residual_blocks(self) -> int:
        """Number of residual blocks."""
        return self._num_residual_blocks

    def get_config(self) -> dict:
        """Get network configuration for serialization."""
        return {
            "num_input_planes": self._num_input_planes,
            "num_filters": self._num_filters,
            "num_residual_blocks": self._num_residual_blocks,
            "policy_size": self._policy_size,
            "se_reduction": self._se_reduction,
            "attention_heads": self._attention_heads,
        }

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 72, 8, 8).

        Returns:
            Tuple of (policy_logits, values, wdl_logits):
            - policy_logits: (batch, policy_size)
            - values: (batch,) - scalar values computed from WDL
            - wdl_logits: (batch, 3) - [P(win), P(draw), P(loss)]
        """
        # Input processing
        x = self.input_conv(x)
        x = self.input_gn(x)
        x = F.gelu(x)

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # Output heads
        policy = self.policy_head(x)
        wdl_logits = self.wdl_head(x)
        value = WDLHead.logits_to_value(wdl_logits)

        return policy, value, wdl_logits

    def predict_single(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Predict policy and value for a single position.

        Args:
            state: Board encoding of shape (72, 8, 8).

        Returns:
            Tuple of (policy, value):
            - policy: numpy array of shape (policy_size,) with probabilities
            - value: float in [-1, +1]
        """
        self.eval()
        device = get_device()
        use_amp = device.type == "cuda" and supports_mixed_precision()

        with torch.inference_mode():
            # Add batch dimension, use non_blocking for async transfer
            x = torch.from_numpy(state).unsqueeze(0)
            dtype = torch.float16 if use_amp else torch.float32
            x = x.to(device, dtype=dtype, non_blocking=True)

            # FP16 inference for speed (if available)
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    policy_logits, value, _ = self(x)
            else:
                policy_logits, value, _ = self(x)

            # Clamp logits for numerical stability and apply softmax
            policy_logits = torch.clamp(policy_logits, min=-50, max=50)
            policy = F.softmax(policy_logits, dim=-1)

            return policy[0].cpu().numpy(), value[0].item()

    def predict_batch(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict policy and value for a batch of positions.

        Args:
            states: Board encodings of shape (batch, 72, 8, 8).

        Returns:
            Tuple of (policies, values):
            - policies: numpy array of shape (batch, policy_size) with probabilities
            - values: numpy array of shape (batch,)
        """
        self.eval()
        device = get_device()
        use_amp = device.type == "cuda" and supports_mixed_precision()

        with torch.inference_mode():
            # Use non_blocking for async transfer
            x = torch.from_numpy(states)
            dtype = torch.float16 if use_amp else torch.float32
            x = x.to(device, dtype=dtype, non_blocking=True)

            # FP16 inference for speed (if available)
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    policy_logits, values, _ = self(x)
            else:
                policy_logits, values, _ = self(x)

            # Clamp logits for numerical stability and apply softmax
            policy_logits = torch.clamp(policy_logits, min=-50, max=50)
            policies = F.softmax(policy_logits, dim=-1)

            return policies.cpu().numpy(), values.cpu().numpy()

    def predict_single_with_wdl(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Predict policy, value, and WDL probabilities for a single position.

        Args:
            state: Board encoding of shape (72, 8, 8).

        Returns:
            Tuple of (policy, value, wdl_probs):
            - policy: numpy array of shape (policy_size,) with probabilities
            - value: float in [-1, +1]
            - wdl_probs: numpy array of shape (3,) with [P(win), P(draw), P(loss)]
        """
        self.eval()
        device = get_device()
        use_amp = device.type == "cuda" and supports_mixed_precision()

        with torch.inference_mode():
            x = torch.from_numpy(state).unsqueeze(0)
            dtype = torch.float16 if use_amp else torch.float32
            x = x.to(device, dtype=dtype, non_blocking=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    policy_logits, value, wdl_logits = self(x)
            else:
                policy_logits, value, wdl_logits = self(x)

            policy_logits = torch.clamp(policy_logits, min=-50, max=50)
            policy = F.softmax(policy_logits, dim=-1)
            wdl_probs = F.softmax(wdl_logits, dim=-1)

            return (
                policy[0].cpu().numpy(),
                value[0].item(),
                wdl_probs[0].cpu().numpy(),
            )

    def predict_batch_with_wdl(
        self, states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict policy, value, and WDL probabilities for a batch of positions.

        Args:
            states: Board encodings of shape (batch, 72, 8, 8).

        Returns:
            Tuple of (policies, values, wdl_probs):
            - policies: numpy array of shape (batch, policy_size) with probabilities
            - values: numpy array of shape (batch,)
            - wdl_probs: numpy array of shape (batch, 3) with [P(win), P(draw), P(loss)]
        """
        self.eval()
        device = get_device()
        use_amp = device.type == "cuda" and supports_mixed_precision()

        with torch.inference_mode():
            x = torch.from_numpy(states)
            dtype = torch.float16 if use_amp else torch.float32
            x = x.to(device, dtype=dtype, non_blocking=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    policy_logits, values, wdl_logits = self(x)
            else:
                policy_logits, values, wdl_logits = self(x)

            policy_logits = torch.clamp(policy_logits, min=-50, max=50)
            policies = F.softmax(policy_logits, dim=-1)
            wdl_probs = F.softmax(wdl_logits, dim=-1)

            return (
                policies.cpu().numpy(),
                values.cpu().numpy(),
                wdl_probs.cpu().numpy(),
            )

    def save(self, path: str) -> None:
        """
        Save network to file.

        Args:
            path: Path to save file (.pt extension recommended).
        """
        # Ensure directory exists
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )

        torch.save(
            {
                "config": self.get_config(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "DualHeadNetwork":
        """
        Load network from file.

        Args:
            path: Path to saved file.
            device: Optional device override.

        Returns:
            Loaded DualHeadNetwork instance.
        """
        if device is None:
            device = get_device()
        elif isinstance(device, str):
            device = torch.device(device)

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        if "config" not in checkpoint:
            raise ValueError(
                f"Invalid checkpoint format: {path}. "
                "Only 72-plane WDL models are supported."
            )

        config = checkpoint["config"]
        state_dict = checkpoint["state_dict"]

        # Filter out deprecated auxiliary head weights (phase_head, moves_left_head)
        # This allows loading old checkpoints that had these heads
        deprecated_prefixes = ("phase_head.", "moves_left_head.")
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith(deprecated_prefixes)
        }

        # Create network with config
        network = cls(**config)
        network.load_state_dict(filtered_state_dict)
        network.to(device)
        network.eval()

        return network


# Alias for compatibility with some UI code
DualHeadResNet = DualHeadNetwork
