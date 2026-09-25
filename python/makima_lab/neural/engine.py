"""
Cognitive Neural Engine and continuous online learning manager for Makima.
Bridges PyTorch deep learning with SQLite Event Sourcing and Bayesian Core.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from .vocab import MakimaTokenizer
from .models import MakimaMindNet, INTENTS, INTENT2IDX, IDX2INTENT


@dataclass
class NeuralInferenceResult:
    """Structured perception result from Makima's neural network."""

    text: str
    intent: str
    intent_confidence: float
    intent_distribution: Dict[str, float]
    target_embedding: List[float]
    polarity: float
    uncertainty: float
    prior_alpha: float
    prior_beta: float
    lambda_rate: float
    memory_norm: float
    latent_vector: List[float] = field(default_factory=list)

    def format_report(self) -> str:
        """Formatted human-readable cognitive report."""
        intents_str = ", ".join(f"{k}: {v:.1%}" for k, v in sorted(self.intent_distribution.items(), key=lambda x: -x[1])[:3])
        return (
            f"--- [ Makima Neural Perception ] ---\n"
            f"Input Testo:        \"{self.text}\"\n"
            f"Intento Rilevato:   {self.intent.upper()} (confidenza {self.intent_confidence:.1%})\n"
            f"Top Distribuzione:  {intents_str}\n"
            f"Polarità / Conf:    {self.polarity:.3f} (Incertezza: {self.uncertainty:.3f})\n"
            f"Priors Bayesiani:   Beta(α={self.prior_alpha:.2f}, β={self.prior_beta:.2f}), Poisson(λ={self.lambda_rate:.2f})\n"
            f"Stato Memoria L2:   {self.memory_norm:.4f}\n"
            f"------------------------------------"
        )


class MakimaNeuralEngine:
    """High-level cognitive neural engine for Makima."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        learning_rate: float = 1e-3,
    ) -> None:
        self.device = torch.device(device)
        self.tokenizer = MakimaTokenizer()
        self.model = MakimaMindNet(vocab_size=max(self.tokenizer.vocab_size + 50, 256))
        self.model.to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.user_memory = torch.zeros(1, self.model.memory_dim, device=self.device)

        self.default_checkpoint = checkpoint_path or os.path.join(".makima", "makima_brain.pt")
        self.total_learning_steps = 0

        # Auto-load existing weights if available
        if os.path.exists(self.default_checkpoint):
            self.load_brain(self.default_checkpoint)

    def perceive(self, text: str, update_memory: bool = True) -> NeuralInferenceResult:
        """Processes a natural language string and produces a neural perception report."""
        self.model.eval()
        tokens = self.tokenizer.encode(text, max_length=32)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs, new_memory = self.model(input_tensor, self.user_memory)
            if update_memory:
                self.user_memory = new_memory

            # Parse intent distribution
            logits = outputs["intent_logits"][0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            intent_dist = {IDX2INTENT[i]: float(probs[i]) for i in range(len(INTENTS))}
            top_idx = int(torch.argmax(logits).item())
            top_intent = IDX2INTENT[top_idx]
            top_conf = float(probs[top_idx])

            target_emb = outputs["target_embedding"][0].cpu().tolist()
            latent_vec = outputs["latent_representation"][0].cpu().tolist()
            polarity = float(outputs["polarity"][0].item())
            uncertainty = float(outputs["uncertainty"][0].item())
            alpha_p = float(outputs["alpha_prior"][0].item())
            beta_p = float(outputs["beta_prior"][0].item())
            lambda_r = float(outputs["lambda_rate"][0].item())
            mem_norm = float(torch.norm(self.user_memory, p=2).item())

        return NeuralInferenceResult(
            text=text,
            intent=top_intent,
            intent_confidence=top_conf,
            intent_distribution=intent_dist,
            target_embedding=target_emb,
            polarity=polarity,
            uncertainty=uncertainty,
            prior_alpha=alpha_p,
            prior_beta=beta_p,
            lambda_rate=lambda_r,
            memory_norm=mem_norm,
            latent_vector=latent_vec,
        )

    def learn_step(
        self,
        text: str,
        intent_label: str | None = None,
        polarity_label: float | None = None,
        auto_save: bool = True,
    ) -> float:
        """
        Executes a single online backpropagation gradient update adapting to user interaction.
        Returns the scalar loss value.
        """
        self.model.train()
        tokens = self.tokenizer.encode(text, max_length=32)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

        self.optimizer.zero_grad()
        outputs, new_memory = self.model(input_tensor, self.user_memory.detach())
        self.user_memory = new_memory.detach()

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        if intent_label and intent_label in INTENT2IDX:
            target_idx = torch.tensor([INTENT2IDX[intent_label]], dtype=torch.long, device=self.device)
            loss = loss + nn.CrossEntropyLoss()(outputs["intent_logits"], target_idx)

        if polarity_label is not None:
            target_pol = torch.tensor([[polarity_label]], dtype=torch.float, device=self.device)
            loss = loss + nn.MSELoss()(outputs["polarity"].unsqueeze(-1), target_pol)

        # Regularization on priors to keep them well-conditioned
        reg_prior = 0.01 * (torch.mean((outputs["alpha_prior"] - 2.0) ** 2) + torch.mean((outputs["beta_prior"] - 2.0) ** 2))
        loss = loss + reg_prior

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.total_learning_steps += 1
        loss_val = float(loss.item())

        if auto_save:
            self.save_brain()

        return loss_val

    def save_brain(self, path: str | None = None) -> str:
        """Saves current neural weights, optimizer, and memory state."""
        save_path = path or self.default_checkpoint
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "user_memory": self.user_memory.cpu(),
                "total_learning_steps": self.total_learning_steps,
            },
            save_path,
        )
        return save_path

    def load_brain(self, path: str | None = None) -> bool:
        """Loads neural weights from checkpoint file."""
        load_path = path or self.default_checkpoint
        if not os.path.exists(load_path):
            return False
        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "user_memory" in checkpoint:
                self.user_memory = checkpoint["user_memory"].to(self.device)
            if "total_learning_steps" in checkpoint:
                self.total_learning_steps = checkpoint["total_learning_steps"]
            return True
        except Exception:
            return False

    def reset_memory(self) -> None:
        """Resets the continuous user latent memory vector to zero."""
        self.user_memory = torch.zeros(1, self.model.memory_dim, device=self.device)


_GLOBAL_ENGINE: Optional[MakimaNeuralEngine] = None


def get_neural_engine() -> MakimaNeuralEngine:
    """Singleton getter for Makima's cognitive neural engine."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = MakimaNeuralEngine()
    return _GLOBAL_ENGINE
