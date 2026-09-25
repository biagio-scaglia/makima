"""
Deep Neural Network architectures for Makima's cognitive and NLP system.
Built with PyTorch, integrating Self-Attention, Bidirectional GRU, and Latent Memory.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

INTENTS = ["query", "journal", "routine", "outcome", "fact"]
INTENT2IDX = {intent: i for i, intent in enumerate(INTENTS)}
IDX2INTENT = {i: intent for i, intent in enumerate(INTENTS)}


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
    """
    Cognitive Deep Neural Network of Makima.
    Combines Self-Attention, Bidirectional GRU, and Continuous User Latent State Memory.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        memory_dim: int = 64,
        num_heads: int = 4,
        num_classes: int = len(INTENTS),
        target_embed_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.target_embed_dim = target_embed_dim

        # 1. Embedding & Positional Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.drop = nn.Dropout(dropout)

        # 2. Bidirectional GRU & Attention Layer
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0.0,
        )
        encoder_dim = hidden_dim * 2  # Bidirectional -> 128
        self.self_attention = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(encoder_dim)

        # 3. User Latent Memory Cell (Recurrent GRU cell for persistent state)
        self.memory_cell = nn.GRUCell(encoder_dim, memory_dim)

        # 4. Multi-Task Cognitive Heads
        # Feature representation combines sequence representation (128) + user memory state (64) = 192
        combined_dim = encoder_dim + memory_dim

        # Head A: Intent Classifier
        self.intent_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # Head B: Semantic Target Space Projection
        self.target_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, target_embed_dim),
        )

        # Head C: Bayesian Calibration & Polarity Bridge
        # Outputs: [polarity_prob, uncertainty, alpha_prior, beta_prior, lambda_rate]
        self.bayesian_bridge = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        user_memory: torch.Tensor | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward propagation.
        token_ids: [batch_size, seq_len]
        user_memory: [batch_size, memory_dim] (optional)
        """
        batch_size, seq_len = token_ids.shape
        if user_memory is None:
            user_memory = torch.zeros(batch_size, self.memory_dim, device=token_ids.device)

        # 1. Embeddings
        x = self.embedding(token_ids) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        x = self.drop(x)

        # 2. BiGRU Encoding
        gru_out, _ = self.gru(x)  # [batch_size, seq_len, encoder_dim]

        # 3. Multi-Head Self-Attention
        attn_out, _ = self.self_attention(gru_out, gru_out, gru_out)
        context = self.norm(gru_out + attn_out)  # Residual connection

        # Global average pooling over non-padded tokens
        mask = (token_ids != 0).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        sum_pooled = (context * mask).sum(dim=1)
        mask_sum = mask.sum(dim=1).clamp(min=1.0)
        pooled = sum_pooled / mask_sum  # [batch_size, encoder_dim]

        # 4. Update Latent User Memory
        new_memory = self.memory_cell(pooled, user_memory)

        # Combined representation
        fused = torch.cat([pooled, new_memory], dim=-1)

        # 5. Heads computation
        intent_logits = self.intent_head(fused)
        target_embedding = F.normalize(self.target_head(fused), p=2, dim=-1)

        # Bayesian Bridge outputs
        bayes_raw = self.bayesian_bridge(fused)
        polarity_prob = torch.sigmoid(bayes_raw[:, 0])
        uncertainty = torch.sigmoid(bayes_raw[:, 1])
        alpha_prior = F.softplus(bayes_raw[:, 2]) + 1.0
        beta_prior = F.softplus(bayes_raw[:, 3]) + 1.0
        lambda_rate = F.softplus(bayes_raw[:, 4]) + 0.1

        outputs = {
            "intent_logits": intent_logits,
            "target_embedding": target_embedding,
            "polarity": polarity_prob,
            "uncertainty": uncertainty,
            "alpha_prior": alpha_prior,
            "beta_prior": beta_prior,
            "lambda_rate": lambda_rate,
            "latent_representation": pooled,
        }

        return outputs, new_memory
