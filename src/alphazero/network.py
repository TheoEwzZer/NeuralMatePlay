"""
Neural network architecture for AlphaZero-style chess.

Architecture: SE-ResNet with Spatial Attention
- Squeeze-and-Excitation blocks for channel recalibration
- Spatial attention in final blocks for global awareness
- Group Normalization for stability with small batches
- GELU activation for smoother gradients
- Dual heads: Policy (1858 moves) and Value (-1 to +1)
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .device import get_device
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
        self.dropout = nn.Dropout(0.3)  # Increased regularization (was 0.2)
        self.fc = nn.Linear(32 * 8 * 8, policy_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        x = F.gelu(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)  # Apply dropout before FC
        x = self.fc(x)
        # Clamp logits to prevent softmax overflow (|logit| > 88 causes overflow in float16)
        x = torch.clamp(x, min=-50, max=50)
        return x  # Clamped logits (softmax applied externally)


class ValueHead(nn.Module):
    """
    Value head: outputs position evaluation in [-1, +1].

    Architecture optimized to prevent collapse:
    - Increased capacity (8 conv channels, 512 hidden)
    - Reduced dropout (0.3 instead of 0.5)
    - Explicit weight initialization for output layer
    """

    def __init__(self, in_channels: int, hidden_size: int = 512):
        super().__init__()
        # Increased conv channels: 4 -> 8 for more capacity
        self.conv = nn.Conv2d(in_channels, 8, 1, bias=False)
        self.gn = nn.GroupNorm(4, 8)  # 4 groups for 8 channels
        # Increased hidden size: 256 -> 512
        self.fc1 = nn.Linear(8 * 8 * 8, hidden_size)
        # Reduced dropout: 0.5 -> 0.3 (matches policy head)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, 1)

        # Explicit initialization to prevent collapse
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to encourage full use of [-1, 1] range."""
        # Xavier init for fc2 to get reasonable initial outputs
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        x = F.gelu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        return x.squeeze(-1)


class PhaseHead(nn.Module):
    """
    Auxiliary phase prediction head: opening/middlegame/endgame.

    This is an auxiliary task that helps the network learn phase-specific features.
    The phase prediction is not used during inference, only for training.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 4, 1, bias=False)
        self.gn = nn.GroupNorm(2, 4)
        self.fc = nn.Linear(4 * 8 * 8, 3)  # 3 phases: opening, middlegame, endgame

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        x = F.gelu(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)  # logits for 3 classes


class DualHeadNetwork(nn.Module):
    """
    AlphaZero-style network with SE-ResNet, attention, and auxiliary phase head.

    Architecture:
    - Input conv: planes -> filters
    - 10 SE-ResBlocks
    - 2 SE-ResBlocks with spatial attention
    - Policy head (move probabilities)
    - Value head (position evaluation)
    - Phase head (auxiliary task: opening/middlegame/endgame)
    """

    def __init__(
        self,
        num_input_planes: int = 60,
        num_filters: int = 192,
        num_residual_blocks: int = 12,
        policy_size: int = MOVE_ENCODING_SIZE,
        se_reduction: int = 8,
        attention_heads: int = 4,
        use_phase_head: bool = True,
    ):
        super().__init__()

        self._num_input_planes = num_input_planes
        self._num_filters = num_filters
        self._num_residual_blocks = num_residual_blocks
        self._policy_size = policy_size
        self._se_reduction = se_reduction
        self._attention_heads = attention_heads
        self._use_phase_head = use_phase_head

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
        self.value_head = ValueHead(num_filters)

        # Auxiliary phase head (for multi-task learning during training)
        if use_phase_head:
            self.phase_head = PhaseHead(num_filters)
        else:
            self.phase_head = None

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
            "use_phase_head": self._use_phase_head,
        }

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, planes, 8, 8).

        Returns:
            Tuple of (policy_logits, values, phase_logits):
            - policy_logits: (batch, policy_size)
            - values: (batch,)
            - phase_logits: (batch, 3) or None if no phase head
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
        value = self.value_head(x)

        # Auxiliary phase head (only during training)
        if self.phase_head is not None:
            phase = self.phase_head(x)
        else:
            phase = None

        return policy, value, phase

    def predict_single(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Predict policy and value for a single position.

        Args:
            state: Board encoding of shape (planes, 8, 8).

        Returns:
            Tuple of (policy, value):
            - policy: numpy array of shape (policy_size,) with probabilities
            - value: float in [-1, +1]
        """
        self.eval()
        with torch.inference_mode():
            # Add batch dimension, use non_blocking for async transfer
            x = torch.from_numpy(state).unsqueeze(0)
            x = x.to(get_device(), dtype=torch.float16, non_blocking=True)

            # FP16 inference for speed
            with torch.amp.autocast(device_type="cuda"):
                policy_logits, value, _ = self(x)

            # Clamp logits for numerical stability and apply softmax
            policy_logits = torch.clamp(policy_logits, min=-50, max=50)
            policy = F.softmax(policy_logits, dim=-1)

            return policy[0].cpu().numpy(), value[0].item()

    def predict_batch(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict policy and value for a batch of positions.

        Args:
            states: Board encodings of shape (batch, planes, 8, 8).

        Returns:
            Tuple of (policies, values):
            - policies: numpy array of shape (batch, policy_size) with probabilities
            - values: numpy array of shape (batch,)
        """
        self.eval()
        with torch.inference_mode():
            # Use non_blocking for async transfer, FP16 for speed
            x = torch.from_numpy(states)
            x = x.to(get_device(), dtype=torch.float16, non_blocking=True)

            # FP16 inference for speed
            with torch.amp.autocast(device_type="cuda"):
                policy_logits, values, _ = self(x)

            # Clamp logits for numerical stability and apply softmax
            policy_logits = torch.clamp(policy_logits, min=-50, max=50)
            policies = F.softmax(policy_logits, dim=-1)

            return policies.cpu().numpy(), values.cpu().numpy()

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

        # Handle different save formats
        if "config" in checkpoint:
            config = checkpoint["config"]
            state_dict = checkpoint["state_dict"]
        else:
            # Legacy format: just state dict
            state_dict = checkpoint
            # Infer config from state dict
            input_conv_weight = state_dict.get("input_conv.weight")
            if input_conv_weight is not None:
                num_input_planes = input_conv_weight.shape[1]
                num_filters = input_conv_weight.shape[0]
            else:
                num_input_planes = 54
                num_filters = 192

            config = {
                "num_input_planes": num_input_planes,
                "num_filters": num_filters,
                "num_residual_blocks": 12,
                "policy_size": MOVE_ENCODING_SIZE,
                "use_phase_head": False,  # Legacy models don't have phase head
            }

        # Handle backward compatibility for use_phase_head
        if "use_phase_head" not in config:
            # Check if state_dict has phase_head weights
            has_phase_head = any(k.startswith("phase_head.") for k in state_dict.keys())
            config["use_phase_head"] = has_phase_head

        # Create network with config
        network = cls(**config)

        # Load state dict with strict=False to handle missing phase_head in old models
        network.load_state_dict(state_dict, strict=False)
        network.to(device)
        network.eval()

        return network


# Alias for compatibility with some UI code
DualHeadResNet = DualHeadNetwork
