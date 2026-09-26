"""Rappresentazione tipizzata delle relazioni e delle finestre temporali."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class TemporalRelation(str, Enum):
    """Categorizzazione della relazione temporale della richiesta."""
    PAST = "PAST"                         # Evento passato ("quando ho pubblicato?", "ieri")
    PRESENT = "PRESENT"                   # Stato attuale ("cosa sto facendo?", "adesso")
    FUTURE = "FUTURE"                     # Evento futuro generico ("quando pubblicherò?", "prossimo")
    RELATIVE_INTERVAL = "RELATIVE_INTERVAL" # Finestra relativa ("questa settimana", "entro 7 giorni")
    SPECIFIC_DATE = "SPECIFIC_DATE"       # Data specifica ("entro il 2026-12-31", "a ottobre")
    UNKNOWN = "UNKNOWN"                   # Nessuna indicazione temporale rilevabile


@dataclass(frozen=True)
class TemporalWindow:
    """Finestra temporale strutturata per le inferenze previsionali."""
    relation: TemporalRelation
    raw_expression: str | None = None
    boundary: str | None = None
    days: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializzazione in dizionario JSON-compatibile."""
        return {
            "relation": self.relation.value,
            "raw_expression": self.raw_expression,
            "boundary": self.boundary,
            "days": self.days,
        }

    @classmethod
    def unspecified(cls) -> TemporalWindow:
        """Crea una finestra temporale priva di vincoli espliciti."""
        return cls(relation=TemporalRelation.UNKNOWN)
