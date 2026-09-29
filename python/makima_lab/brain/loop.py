"""BrainLoop: un tick del cervello neurale operativo."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from makima_lab.mind.deliberation import MindDeliberationEngine
from makima_lab.mind.episodic_memory import EpisodicMemoryStore
from makima_lab.mind.schemas import CognitivePulse, EpistemicSelfState
from makima_lab.nlp.pipeline import SemanticForecastPipeline
from makima_lab.nlp.schemas.intent import Intent
from makima_lab.nlp.schemas.structured_intent import StructuredIntent


class BrainAction(str, Enum):
    SPEAK = "SPEAK"
    REQUEST_RUST_FORECAST = "REQUEST_RUST_FORECAST"
    REMEMBER = "REMEMBER"
    ASK_CLARIFY = "ASK_CLARIFY"
    IDLE = "IDLE"


@dataclass
class BrainTickResult:
    """Esito di un tick del cervello."""

    pulse: CognitivePulse
    action: BrainAction
    action_confidence: float
    structured_intent: StructuredIntent
    neural_intent: str
    workspace_norm: float
    rust_forecast_hint: Optional[str] = None
    hypotheses: List[str] = field(default_factory=list)

    def format_user_guide(self) -> str:
        lines = [
            f"Azione cervello: {self.action.value} ({self.action_confidence:.0%})",
            f"Intent NLP: {self.structured_intent.intent.value} → target={self.structured_intent.target}",
        ]
        if self.action == BrainAction.REQUEST_RUST_FORECAST and self.rust_forecast_hint:
            lines.append(f"Per i numeri calibrati esegui: {self.rust_forecast_hint}")
        return "\n".join(lines)


class BrainLoop:
    """Ciclo unico: NLP → rete neurale → deliberazione → azione."""

    FORECASTING_PATH = False

    def __init__(
        self,
        memory: Optional[EpisodicMemoryStore] = None,
        deliberation: Optional[MindDeliberationEngine] = None,
        nlp: Optional[SemanticForecastPipeline] = None,
        neural_engine=None,
    ) -> None:
        self.memory = memory or EpisodicMemoryStore()
        self.deliberation = deliberation or MindDeliberationEngine(memory_store=self.memory)
        self.nlp = nlp or SemanticForecastPipeline()
        self._neural = neural_engine  # lazy

    @property
    def neural(self):
        if self._neural is None:
            try:
                from makima_lab.neural import get_neural_engine

                self._neural = get_neural_engine()
            except Exception:
                self._neural = None
        return self._neural

    def _episodic_embeddings(self, query: str, top_k: int = 4) -> List[List[float]]:
        retrieved = self.memory.retrieve_relevant_memories(query, top_k=top_k)
        vectors: List[List[float]] = []
        embedder = getattr(self.memory, "_embedder", None)
        for mem, _score in retrieved:
            text = f"{mem.summary} {mem.content}"
            if embedder is not None:
                try:
                    vec = embedder.embed(text)
                    if hasattr(vec, "tolist"):
                        vec = vec.tolist()
                    vectors.append([float(x) for x in list(vec)[:64]])
                    continue
                except Exception:
                    pass
            # Fallback hash stabile a 64 dim
            import hashlib

            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=32).digest()
            vals = []
            for i in range(64):
                b = digest[i % len(digest)]
                vals.append((b / 255.0) * 2.0 - 1.0)
            vectors.append(vals)
        return vectors

    def _resolve_action(
        self,
        structured: StructuredIntent,
        neural_action: str,
        neural_conf: float,
    ) -> tuple[BrainAction, float]:
        """Policy: NLP di produzione ha priorità sul forecast; altrimenti rete."""
        if structured.admits_forecast() if hasattr(structured, "admits_forecast") else False:
            return BrainAction.REQUEST_RUST_FORECAST, max(neural_conf, structured.confidence)
        if structured.is_valid_for_core and structured.target and structured.intent in (
            Intent.QUERY,
            Intent.TEMPORAL_QUERY,
        ):
            return BrainAction.REQUEST_RUST_FORECAST, max(neural_conf, structured.confidence)

        try:
            action = BrainAction(neural_action)
        except ValueError:
            action = BrainAction.SPEAK

        if structured.intent == Intent.UNKNOWN:
            return BrainAction.ASK_CLARIFY, max(neural_conf, 0.7)
        if structured.intent == Intent.OBSERVATION:
            return BrainAction.REMEMBER, max(neural_conf, structured.confidence)
        if structured.intent == Intent.INFORMATION:
            return BrainAction.SPEAK, max(neural_conf, structured.confidence)
        return action, neural_conf

    def tick(self, stimulus: str, *, learn: bool = True) -> BrainTickResult:
        """Un ciclo completo del cervello operativo."""
        structured = self.nlp.process_intent(stimulus)
        epi = self._episodic_embeddings(stimulus)

        neural_intent = "query"
        neural_action = "SPEAK"
        neural_conf = 0.5
        workspace_norm = 0.0

        neural = self.neural
        if neural is not None:
            perception = neural.perceive(stimulus, update_memory=True, episodic_embeddings=epi)
            neural_intent = perception.intent
            neural_action = perception.action
            neural_conf = perception.action_confidence
            workspace_norm = float(sum(x * x for x in perception.workspace) ** 0.5)

            if learn:
                # Supervisiona azione con policy NLP (segnale stabile)
                action_for_learn, _ = self._resolve_action(structured, neural_action, neural_conf)
                intent_map = {
                    Intent.QUERY: "query",
                    Intent.TEMPORAL_QUERY: "query",
                    Intent.OBSERVATION: "outcome",
                    Intent.INFORMATION: "fact",
                    Intent.STATUS: "routine",
                    Intent.COMMAND: "routine",
                    Intent.UNKNOWN: "journal",
                }
                neural.learn_step(
                    stimulus,
                    intent_label=intent_map.get(structured.intent, "query"),
                    action_label=action_for_learn.value,
                    polarity_label=perception.polarity,
                    episodic_embeddings=epi,
                    auto_save=True,
                )

        action, action_conf = self._resolve_action(structured, neural_action, neural_conf)

        # Deliberazione narrativa (Mind) — stessi numeri Laplace onesti
        pulse = self.deliberation.deliberate(stimulus)

        # Se NLP era UNKNOWN ma la memoria ha risposto, non restare su ASK_CLARIFY
        if (
            action == BrainAction.ASK_CLARIFY
            and pulse.retrieved_memories
            and "Rifiuto Esplicito" not in pulse.inner_monologue
        ):
            action = BrainAction.SPEAK
            action_conf = max(action_conf, 0.65)

        # Arricchisci monologo con traccia neurale
        neural_trace = (
            f"0. [Cervello neurale]: intent_soft={neural_intent}, "
            f"azione={action.value} ({action_conf:.0%}), workspace_L2={workspace_norm:.3f}. "
            f"FORECASTING_PATH=False — probabilità solo via Rust."
        )
        pulse.inner_monologue = neural_trace + "\n" + pulse.inner_monologue

        rust_hint = None
        if action == BrainAction.REQUEST_RUST_FORECAST:
            q = stimulus.replace('"', '\\"')
            rust_hint = f'makima query "{q}"'
            # Append hint all'utterance senza inventare numeri
            pulse.conscious_utterance = (
                pulse.conscious_utterance.rstrip()
                + f"\n\n[Cervello] Per la probabilità calibrata esegui: {rust_hint}"
            )

        return BrainTickResult(
            pulse=pulse,
            action=action,
            action_confidence=action_conf,
            structured_intent=structured,
            neural_intent=neural_intent,
            workspace_norm=workspace_norm,
            rust_forecast_hint=rust_hint,
            hypotheses=list(pulse.hypotheses),
        )


_GLOBAL_BRAIN: Optional[BrainLoop] = None


def get_brain() -> BrainLoop:
    global _GLOBAL_BRAIN
    if _GLOBAL_BRAIN is None:
        _GLOBAL_BRAIN = BrainLoop()
    return _GLOBAL_BRAIN
