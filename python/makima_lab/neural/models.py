"""
MakimaMindNet — cervello neurale operativo (PyTorch).

BiGRU + Self-Attention + memoria latente + attenzione episodica + action head.
FORECASTING_PATH=False: i prior suggeriti NON sono posteriors di produzione.
"""

from __future__ import annotations
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

INTENTS = ["query", "journal", "routine", "outcome", "fact"]
INTENT2IDX = {intent: i for i, intent in enumerate(INTENTS)}
IDX2INTENT = {i: intent for i, intent in enumerate(INTENTS)}

ACTIONS = ["SPEAK", "REQUEST_RUST_FORECAST", "REMEMBER", "ASK_CLARIFY", "IDLE"]
ACTION2IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX2ACTION = {i: a for i, a in enumerate(ACTIONS)}


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class MakimaMindNet(nn.Module):
    """Cervello neurale: encoding, memoria latente, attenzione episodica, azioni."""

    def __init__(
        self,
        vocab_size: int = 1024,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        memory_dim: int = 64,
        num_heads: int = 4,
        num_classes: int = len(INTENTS),
        num_actions: int = len(ACTIONS),
        target_embed_dim: int = 32,
        workspace_dim: int = 64,
        episodic_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.target_embed_dim = target_embed_dim
        self.workspace_dim = workspace_dim
        self.episodic_dim = episodic_dim
        self.num_actions = num_actions

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.drop = nn.Dropout(dropout)

        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0.0,
        )
        encoder_dim = hidden_dim * 2
        self.encoder_dim = encoder_dim
        self.self_attention = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(encoder_dim)
        self.memory_cell = nn.GRUCell(encoder_dim, memory_dim)

        self.episodic_proj = nn.Linear(episodic_dim, encoder_dim)
        self.episodic_attn = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        combined_dim = encoder_dim + memory_dim + encoder_dim
        self.combined_dim = combined_dim

        self.intent_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self.target_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, target_embed_dim),
        )
        self.suggestion_bridge = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )
        self.action_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_actions),
        )
        self.workspace_head = nn.Sequential(
            nn.Linear(combined_dim, workspace_dim),
            nn.Tanh(),
        )
        self.bayesian_bridge = self.suggestion_bridge

    def forward(
        self,
        token_ids: torch.Tensor,
        user_memory: torch.Tensor | None = None,
        episodic_memory: torch.Tensor | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        batch_size, _seq_len = token_ids.shape
        if user_memory is None:
            user_memory = torch.zeros(batch_size, self.memory_dim, device=token_ids.device)

        x = self.embedding(token_ids) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        x = self.drop(x)

        gru_out, _ = self.gru(x)
        attn_out, _ = self.self_attention(gru_out, gru_out, gru_out)
        context = self.norm(gru_out + attn_out)

        mask = (token_ids != 0).unsqueeze(-1).float()
        pooled = (context * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        new_memory = self.memory_cell(pooled, user_memory)

        if episodic_memory is not None and episodic_memory.numel() > 0 and episodic_memory.size(1) > 0:
            epi = episodic_memory
            if epi.size(-1) != self.episodic_dim:
                if epi.size(-1) > self.episodic_dim:
                    epi = epi[..., : self.episodic_dim]
                else:
                    pad = torch.zeros(
                        *epi.shape[:-1],
                        self.episodic_dim - epi.size(-1),
                        device=epi.device,
                        dtype=epi.dtype,
                    )
                    epi = torch.cat([epi, pad], dim=-1)
            epi_keys = self.episodic_proj(epi)
            query = pooled.unsqueeze(1)
            epi_ctx, epi_weights = self.episodic_attn(query, epi_keys, epi_keys)
            episodic_context = epi_ctx.squeeze(1)
            episodic_attn_weights = epi_weights.squeeze(1)
        else:
            episodic_context = torch.zeros(batch_size, self.encoder_dim, device=token_ids.device)
            episodic_attn_weights = None

        fused = torch.cat([pooled, new_memory, episodic_context], dim=-1)

        intent_logits = self.intent_head(fused)
        target_embedding = F.normalize(self.target_head(fused), p=2, dim=-1)
        action_logits = self.action_head(fused)
        workspace = self.workspace_head(fused)

        sug = self.suggestion_bridge(fused)
        polarity_prob = torch.sigmoid(sug[:, 0])
        uncertainty = torch.sigmoid(sug[:, 1])
        alpha_prior = F.softplus(sug[:, 2]) + 1.0
        beta_prior = F.softplus(sug[:, 3]) + 1.0
        lambda_rate = F.softplus(sug[:, 4])

        outputs: Dict[str, torch.Tensor] = {
            "intent_logits": intent_logits,
            "action_logits": action_logits,
            "target_embedding": target_embedding,
            "polarity": polarity_prob,
            "uncertainty": uncertainty,
            "alpha_prior": alpha_prior,
            "beta_prior": beta_prior,
            "lambda_rate": lambda_rate,
            "suggested_alpha": alpha_prior,
            "suggested_beta": beta_prior,
            "latent_representation": pooled,
            "workspace": workspace,
            "episodic_context": episodic_context,
        }
        if episodic_attn_weights is not None:
            outputs["episodic_attn_weights"] = episodic_attn_weights

        return outputs, new_memory
