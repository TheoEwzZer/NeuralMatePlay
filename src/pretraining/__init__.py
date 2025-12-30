"""
Pretraining module for supervised learning on master games.

Provides chunked PGN processing and PyTorch datasets for
training the network on human expert games before self-play.
"""

from .pgn_processor import PGNProcessor
from .dataset import ChessPositionDataset
from .pretrain import pretrain, create_dataloaders

__all__ = [
    "PGNProcessor",
    "ChessPositionDataset",
    "pretrain",
    "create_dataloaders",
]
