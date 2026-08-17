"""Small deterministic GPT used by tests and case studies."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TinyGPTConfig:
    vocab_size: int = 256
    max_sequence_length: int = 256
    depth: int = 4
    width: int = 128
    heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            "vocab_size",
            "max_sequence_length",
            "depth",
            "width",
            "heads",
            "mlp_ratio",
        )
        for name in integer_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.proj = nn.Linear(config.width, config.width)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, length, width = inputs.shape
        qkv = self.qkv(inputs)
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch, length, self.heads, self.head_width).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        scores = query @ key.transpose(-2, -1)
        scores = scores * (1.0 / math.sqrt(self.head_width))
        causal = torch.ones(length, length, device=inputs.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        probabilities = self.attention_dropout(F.softmax(scores, dim=-1))
        context = probabilities @ value
        context = context.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output_dropout(self.proj(context))


class ResidualAttention(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.width)
        self.attention = CausalSelfAttention(config)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.attention(self.norm(inputs))


class ResidualMLP(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        hidden = config.mlp_ratio * config.width
        self.norm = nn.LayerNorm(config.width)
        self.input = nn.Linear(config.width, hidden)
        self.output = nn.Linear(hidden, config.width)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = F.gelu(self.input(self.norm(inputs)), approximate="none")
        return inputs + self.dropout(self.output(hidden))


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.attention_unit = ResidualAttention(config)
        self.mlp_unit = ResidualMLP(config)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.mlp_unit(self.attention_unit(inputs))


class TinyGPT(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.width)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.depth))
        self.final_norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def prepare_inputs(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        length = token_ids.shape[1]
        if length > self.config.max_sequence_length:
            raise ValueError("sequence length exceeds model configuration")
        positions = torch.arange(length, device=token_ids.device)
        embeddings = self.token_embedding(token_ids) + self.position_embedding(positions)
        return self.embedding_dropout(embeddings)

    def finish(self, hidden: Tensor) -> Tensor:
        return self.lm_head(self.final_norm(hidden))

    def forward(self, token_ids: Tensor, targets: Tensor | None = None) -> Tensor:
        hidden = self.prepare_inputs(token_ids)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.finish(hidden)
        if targets is None:
            return logits
        if targets.shape != token_ids.shape:
            raise ValueError("targets must have the same shape as token_ids")
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

