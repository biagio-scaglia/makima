"""
Cognitive Neural Engine — cervello operativo (FORECASTING_PATH=False).

Percezione soft, azioni, workspace. Non alimenta il forecast Rust.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .vocab import MakimaTokenizer
from .models import (
    ACTIONS,
    ACTION2IDX,
    IDX2ACTION,
    INTENTS,
    INTENT2IDX,
    IDX2INTENT,
    MakimaMindNet,
)


@dataclass
class NeuralInferenceResult:
    """Percezione + azione del cervello neurale (lab)."""

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
    action: str = "SPEAK"
    action_confidence: float = 0.0
    action_distribution: Dict[str, float] = field(default_factory=dict)
    workspace: List[float] = field(default_factory=list)
    for_forecasting: bool = False
    priors_are_suggestions: bool = True

    def format_report(self) -> str:
        intents_str = ", ".join(
            f"{k}: {v:.1%}"
            for k, v in sorted(self.intent_distribution.items(), key=lambda x: -x[1])[:3]
        )
        actions_str = ", ".join(
            f"{k}: {v:.1%}"
            for k, v in sorted(self.action_distribution.items(), key=lambda x: -x[1])[:3]
        )
        return (
            f"--- [ Makima Neural Brain — LAB / FORECASTING_PATH=False ] ---\n"
            f"Input Testo:        \"{self.text}\"\n"
            f"Intento soft:       {self.intent.upper()} ({self.intent_confidence:.1%})\n"
            f"Top Intent:         {intents_str}\n"
            f"Azione cervello:    {self.action} ({self.action_confidence:.1%})\n"
            f"Top Azioni:         {actions_str}\n"
            f"Polarità / Conf:    {self.polarity:.3f} (uncert={self.uncertainty:.3f})\n"
            f"Prior SUGGERITI:    Beta(α={self.prior_alpha:.2f}, β={self.prior_beta:.2f}), "
            f"λ≈{self.lambda_rate:.2f}\n"
            f"Workspace dim:      {len(self.workspace)} | Memoria L2: {self.memory_norm:.4f}\n"
            f"Forecast path:      NO — usa `makima query` se azione=REQUEST_RUST_FORECAST\n"
            f"----------------------------------------------------------------"
        )


class MakimaNeuralEngine:
    """Motore del cervello neurale (intent soft + azioni + workspace)."""

    FORECASTING_PATH = False

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        learning_rate: float = 1e-3,
    ) -> None:
        self.device = torch.device(device)
        self.tokenizer = MakimaTokenizer(max_vocab_size=2048)
        self.model = MakimaMindNet(vocab_size=2048)
        self.model.to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.user_memory = torch.zeros(1, self.model.memory_dim, device=self.device)
        self.workspace_state = torch.zeros(1, self.model.workspace_dim, device=self.device)

        self.default_checkpoint = checkpoint_path or os.path.join(".makima", "makima_brain.pt")
        self.total_learning_steps = 0

        if os.path.exists(self.default_checkpoint):
            self.load_brain(self.default_checkpoint)

    def _prep_episodic(self, episodic_embeddings: Optional[List[List[float]]]) -> Optional[torch.Tensor]:
        if not episodic_embeddings:
            return None
        # Normalizza a episodic_dim
        dim = self.model.episodic_dim
        rows = []
        for vec in episodic_embeddings[:8]:
            v = list(vec)[:dim]
            if len(v) < dim:
                v = v + [0.0] * (dim - len(v))
            rows.append(v)
        return torch.tensor([rows], dtype=torch.float32, device=self.device)

    def perceive(
        self,
        text: str,
        update_memory: bool = True,
        episodic_embeddings: Optional[List[List[float]]] = None,
    ) -> NeuralInferenceResult:
        """Tick di percezione del cervello."""
        self.model.eval()
        tokens = self.tokenizer.encode(text, max_length=32)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
        epi = self._prep_episodic(episodic_embeddings)

        with torch.no_grad():
            outputs, new_memory = self.model(input_tensor, self.user_memory, epi)
            if update_memory:
                self.user_memory = new_memory
            if "workspace" in outputs:
                self.workspace_state = outputs["workspace"].detach()

            logits = outputs["intent_logits"][0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            intent_dist = {IDX2INTENT[i]: float(probs[i]) for i in range(len(INTENTS))}
            top_idx = int(torch.argmax(logits).item())
            top_intent = IDX2INTENT[top_idx]
            top_conf = float(probs[top_idx])

            act_logits = outputs["action_logits"][0]
            act_probs = torch.softmax(act_logits, dim=-1).cpu().numpy()
            action_dist = {IDX2ACTION[i]: float(act_probs[i]) for i in range(len(ACTIONS))}
            act_idx = int(torch.argmax(act_logits).item())
            action = IDX2ACTION[act_idx]
            act_conf = float(act_probs[act_idx])

            target_emb = outputs["target_embedding"][0].cpu().tolist()
            latent_vec = outputs["latent_representation"][0].cpu().tolist()
            workspace = outputs["workspace"][0].cpu().tolist()
            polarity = float(outputs["polarity"][0].item())
            uncertainty = float(outputs["uncertainty"][0].item())
            alpha_p = max(1.0, float(outputs["alpha_prior"][0].item()))
            beta_p = max(1.0, float(outputs["beta_prior"][0].item()))
            lambda_r = max(0.0, float(outputs["lambda_rate"][0].item()))
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
            action=action,
            action_confidence=act_conf,
            action_distribution=action_dist,
            workspace=workspace,
            for_forecasting=False,
            priors_are_suggestions=True,
        )

    def learn_step(
        self,
        text: str,
        intent_label: str | None = None,
        polarity_label: float | None = None,
        action_label: str | None = None,
        episodic_embeddings: Optional[List[List[float]]] = None,
        auto_save: bool = True,
    ) -> float:
        """Online learning: intent / action / polarity."""
        self.model.train()
        tokens = self.tokenizer.encode(text, max_length=32)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
        epi = self._prep_episodic(episodic_embeddings)

        self.optimizer.zero_grad()
        outputs, new_memory = self.model(input_tensor, self.user_memory.detach(), epi)
        self.user_memory = new_memory.detach()
        if "workspace" in outputs:
            self.workspace_state = outputs["workspace"].detach()

        loss = torch.zeros((), device=self.device)

        if intent_label and intent_label in INTENT2IDX:
            target_idx = torch.tensor([INTENT2IDX[intent_label]], dtype=torch.long, device=self.device)
            loss = loss + nn.CrossEntropyLoss()(outputs["intent_logits"], target_idx)

        if action_label and action_label in ACTION2IDX:
            act_idx = torch.tensor([ACTION2IDX[action_label]], dtype=torch.long, device=self.device)
            loss = loss + nn.CrossEntropyLoss()(outputs["action_logits"], act_idx)

        if polarity_label is not None:
            target_pol = torch.tensor([[polarity_label]], dtype=torch.float, device=self.device)
            loss = loss + nn.MSELoss()(outputs["polarity"].unsqueeze(-1), target_pol)

        reg_prior = 0.01 * (
            torch.mean((outputs["alpha_prior"] - 2.0) ** 2)
            + torch.mean((outputs["beta_prior"] - 2.0) ** 2)
        )
        loss = loss + reg_prior

        if float(loss.item()) == 0.0 and not loss.requires_grad:
            # Nessun segnale supervisionato: skip
            return 0.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.total_learning_steps += 1
        loss_val = float(loss.item())
        if auto_save:
            self.save_brain()
        return loss_val

    def save_brain(self, path: str | None = None) -> str:
        save_path = path or self.default_checkpoint
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "user_memory": self.user_memory.cpu(),
                "workspace_state": self.workspace_state.cpu(),
                "total_learning_steps": self.total_learning_steps,
                "arch": "mindnet_v2_actions_workspace",
            },
            save_path,
        )
        return save_path

    def load_brain(self, path: str | None = None) -> bool:
        load_path = path or self.default_checkpoint
        if not os.path.exists(load_path):
            return False
        try:
            checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
            saved = checkpoint["model_state_dict"]
            saved_emb = saved["embedding.weight"].shape
            if saved_emb != self.model.embedding.weight.shape:
                return False
            # Compat: checkpoint vecchi senza action/workspace → partial load
            missing, unexpected = self.model.load_state_dict(saved, strict=False)
            if missing:
                # Nuove head random: ok per v2
                pass
            if "optimizer_state_dict" in checkpoint and not missing:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception:
                    pass
            if "user_memory" in checkpoint:
                um = checkpoint["user_memory"].to(self.device)
                if um.shape == self.user_memory.shape:
                    self.user_memory = um
            if "workspace_state" in checkpoint:
                ws = checkpoint["workspace_state"].to(self.device)
                if ws.shape == self.workspace_state.shape:
                    self.workspace_state = ws
            if "total_learning_steps" in checkpoint:
                self.total_learning_steps = checkpoint["total_learning_steps"]
            return True
        except Exception:
            return False

    def reset_memory(self) -> None:
        self.user_memory = torch.zeros(1, self.model.memory_dim, device=self.device)
        self.workspace_state = torch.zeros(1, self.model.workspace_dim, device=self.device)


_GLOBAL_ENGINE: Optional[MakimaNeuralEngine] = None


def get_neural_engine() -> MakimaNeuralEngine:
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = MakimaNeuralEngine()
    return _GLOBAL_ENGINE
