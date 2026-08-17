"""Optimal adjoint checkpoint schedules for PyTorch transformer chains."""

from torchrevolve.chain import BlockChain, ChainProfile, UnitProfile
from torchrevolve.executor import GradResult, run_scheduled_backward
from torchrevolve.memmodel import TransformerShape, transformer_activation_bytes
from torchrevolve.model import TinyGPT, TinyGPTConfig
from torchrevolve.schedules import Action, CostEstimate, LegalityReport, Schedule

__all__ = [
    "Action",
    "BlockChain",
    "ChainProfile",
    "CostEstimate",
    "GradResult",
    "LegalityReport",
    "Schedule",
    "TinyGPT",
    "TinyGPTConfig",
    "TransformerShape",
    "UnitProfile",
    "run_scheduled_backward",
    "transformer_activation_bytes",
]

__version__ = "0.1.0"
