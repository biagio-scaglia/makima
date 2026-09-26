"""Schemi di stato cognitivo, monologo interiore e memoria episodica per la mente di Makima."""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CognitiveMood(str, Enum):
    """Stato affettivo ed epistemico interno di Makima."""
    CALM_ANALYTICAL = "CALM_ANALYTICAL"       # Stato nominale: lucida, ordinata, calcolatrice
    DEEP_REFLECTION = "DEEP_REFLECTION"       # Riflessione introspettiva su anomalie o errori passati
    VIGILANT_FOCUSED = "VIGILANT_FOCUSED"     # Allerta su regressioni, build a rischio o incertezze elevate
    INQUISITIVE = "INQUISITIVE"               # Desiderio di acquisire nuove evidenze e chiarire ambiguità
    CONFIDENT_HARMONY = "CONFIDENT_HARMONY"   # Alta calibrazione statistica e previsioni confermate


class MemoryCategory(str, Enum):
    """Categoria della memoria episodica."""
    DEVELOPER_FACT = "DEVELOPER_FACT"         # Confidenza o abitudine condivisa dall'utente
    PREDICTION_MOMENT = "PREDICTION_MOMENT"   # Previsione effettuata con relativo esito o incertezza
    ERROR_REFLECTION = "ERROR_REFLECTION"     # Analisi post-mortem di un errore di calibrazione
    REPO_MILESTONE = "REPO_MILESTONE"         # Evento saliente nel codice (release, refactoring maggiore)
    SPONTANEOUS_THOUGHT = "SPONTANEOUS_THOUGHT"# Pensiero nato autonomamente durante l'idle


@dataclass(frozen=True)
class EpistemicSelfState:
    """Rappresenta l'autoconsapevolezza matematica e contestuale di Makima nel momento presente."""
    epistemic_uncertainty: float              # Varianza media delle credenze attive [0.0, 0.25]
    brier_skill_score: float                  # Autostima statistica rispetto al caso [-1.0, 1.0]
    total_memories_count: int                 # Numero totale di memorie episodiche consolidate
    observed_commits_count: int               # Numero di commit osservati nel repository
    focus_target: Optional[str]               # Target o argomento attualmente al centro della sua attenzione
    mood: CognitiveMood                       # Stato affettivo/epistemico corrente
    uptime_seconds: float                     # Tempo di vita della sessione corrente

    def summary(self) -> str:
        return (
            f"Stato: {self.mood.value} | Incertezza Epistemica: {self.epistemic_uncertainty:.4f} | "
            f"BSS: {self.brier_skill_score:+.2f} | Focus: {self.focus_target or '[Globale]'} | "
            f"Memorie: {self.total_memories_count}"
        )


@dataclass
class CognitiveExperience:
    """Singola memoria episodica autobiografica persistente."""
    experience_id: str
    timestamp: float
    category: MemoryCategory
    summary: str
    content: str
    associated_target: Optional[str] = None
    epistemic_confidence: float = 0.5
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "timestamp": self.timestamp,
            "category": self.category.value,
            "summary": self.summary,
            "content": self.content,
            "associated_target": self.associated_target,
            "epistemic_confidence": round(self.epistemic_confidence, 4),
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CognitiveExperience:
        return cls(
            experience_id=data["experience_id"],
            timestamp=data["timestamp"],
            category=MemoryCategory(data["category"]),
            summary=data["summary"],
            content=data["content"],
            associated_target=data.get("associated_target"),
            epistemic_confidence=data.get("epistemic_confidence", 0.5),
            tags=data.get("tags", []),
        )


@dataclass
class CognitivePulse:
    """Risultato completo di una deliberazione o flusso di coscienza di Makima."""
    pulse_id: str
    timestamp: float
    prompt_or_trigger: str
    inner_monologue: str                      # Il ragionamento passo-passo che Makima formula tra sé prima di parlare
    conscious_utterance: str                  # La comunicazione finale all'utente
    self_state: EpistemicSelfState
    retrieved_memories: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pulse_id": self.pulse_id,
            "timestamp": self.timestamp,
            "prompt_or_trigger": self.prompt_or_trigger,
            "inner_monologue": self.inner_monologue,
            "conscious_utterance": self.conscious_utterance,
            "self_state": {
                "epistemic_uncertainty": round(self.self_state.epistemic_uncertainty, 4),
                "brier_skill_score": round(self.self_state.brier_skill_score, 4),
                "total_memories_count": self.self_state.total_memories_count,
                "focus_target": self.self_state.focus_target,
                "mood": self.self_state.mood.value,
            },
            "retrieved_memories": self.retrieved_memories,
            "hypotheses": self.hypotheses,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
