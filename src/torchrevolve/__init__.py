"""Optimal adjoint checkpoint schedules for PyTorch transformer chains."""

from torchrevolve.chain import BlockChain, ChainProfile, UnitProfile
from torchrevolve.executor import GradResult, run_scheduled_backward
from torchrevolve.heuristics import (
    make_all_schedule,
    make_none_schedule,
    make_selective_schedule,
    make_uniform_schedule,
)
from torchrevolve.memmodel import (
    MemoryValidation,
    TransformerShape,
    transformer_activation_bytes,
)
from torchrevolve.model import TinyGPT, TinyGPTConfig
from torchrevolve.schedules import Action, CostEstimate, LegalityReport, Schedule
from torchrevolve.selection import SCHEDULERS, ScheduleSelection, select_schedule

__all__ = [
    "SCHEDULERS",
    "Action",
    "BlockChain",
    "ChainProfile",
    "CostEstimate",
    "GradResult",
    "LegalityReport",
    "MemoryValidation",
    "Schedule",
    "ScheduleSelection",
    "TinyGPT",
    "TinyGPTConfig",
    "TransformerShape",
    "UnitProfile",
    "make_all_schedule",
    "make_none_schedule",
    "make_selective_schedule",
    "make_uniform_schedule",
    "run_scheduled_backward",
    "select_schedule",
    "transformer_activation_bytes",
]

__version__ = "0.1.0"
