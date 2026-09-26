"""Rappresentazione finale strutturata dell'intento validato per il runtime Rust."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from makima_lab.nlp.schemas.intent import Intent
from makima_lab.nlp.schemas.temporal import TemporalWindow


@dataclass(frozen=True)
class StructuredIntent:
    """Oggetto immutabile e tipizzato che incapsula la comprensione semantica della query."""
    raw_query: str
    intent: Intent
    target: Optional[str]
    temporal_window: TemporalWindow
    entities: List[str] = field(default_factory=list)
    confidence: float = 0.0
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    is_valid_for_core: bool = False
    validation_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Restituisce la rappresentazione serializzabile conforme alle DTO Rust."""
        return {
            "raw_query": self.raw_query,
            "intent": self.intent.value,
            "target": self.target,
            "temporal_window": self.temporal_window.to_dict(),
            "entities": self.entities,
            "confidence": round(self.confidence, 4),
            "confidence_breakdown": {
                k: round(v, 4) if isinstance(v, (int, float)) else str(v)
                for k, v in self.confidence_breakdown.items()
            },
            "is_valid_for_core": self.is_valid_for_core,
            "validation_notes": self.validation_notes,
        }

    def to_json(self) -> str:
        """Serializza l'intento strutturato in una stringa JSON deterministica."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def summary(self) -> str:
        """Riepilogo formattato in stile diagnostico Makima."""
        lines = [
            f"Query Originale:  \"{self.raw_query}\"",
            f"Intent:           {self.intent.value}",
            f"Target:           {self.target or '[None]'}",
            f"Temporal Window:  {self.temporal_window.relation.value}" + (f" (days={self.temporal_window.days})" if self.temporal_window.days else ""),
            f"Entità Estratte:  {', '.join(self.entities) if self.entities else '[Nessuna]'}",
            f"Confidenza:       {self.confidence * 100:.1f}%",
            f"Valido per Core:  {'Sì' if self.is_valid_for_core else 'No'}",
        ]
        if self.validation_notes:
            lines.append(f"Note Validazione: {'; '.join(self.validation_notes)}")
        return "\n".join(lines)
