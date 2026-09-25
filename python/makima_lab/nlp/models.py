"""Modelli di dominio semantico e rappresentazioni intermedie per le interrogazioni NLP."""

from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    """Intento semantico estratto dalla query in linguaggio naturale."""
    FORECAST = "FORECAST"
    RELEASE_PREDICTION = "RELEASE_PREDICTION"
    OBSERVATION_RECORD = "OBSERVATION_RECORD"
    STATUS_QUERY = "STATUS_QUERY"
    UNSUPPORTED = "UNSUPPORTED"


class TemporalRelation(str, Enum):
    """Relazione temporale associata alla previsione."""
    NEXT = "NEXT"                     # "il prossimo"
    BEFORE = "BEFORE"                 # "entro [data/mese]"
    WITHIN_DAYS = "WITHIN_DAYS"       # "entro N giorni"
    THIS_WEEK = "THIS_WEEK"           # "questa settimana"
    THIS_MONTH = "THIS_MONTH"         # "questo mese"
    UNSPECIFIED = "UNSPECIFIED"       # Orizzonte temporale non esplicitato


@dataclass(frozen=True)
class TemporalWindow:
    """Finestra temporale strutturata."""
    relation: TemporalRelation
    boundary: str | None = None
    days: int | None = None

    def __repr__(self) -> str:
        if self.relation == TemporalRelation.WITHIN_DAYS:
            return f"TemporalWindow(WITHIN_DAYS, days={self.days})"
        if self.boundary:
            return f"TemporalWindow({self.relation.value}, boundary='{self.boundary}')"
        return f"TemporalWindow({self.relation.value})"


@dataclass(frozen=True)
class ForecastQuery:
    """Rappresentazione formale intermedia della query tradotta per makima-core."""
    raw_query: str
    intent: Intent
    target: str | None
    temporal_window: TemporalWindow
    confidence: float

    @property
    def is_valid_forecast(self) -> bool:
        """Indica se la query e' valida ed eseguibile dal motore di previsione."""
        return (
            self.intent in (Intent.FORECAST, Intent.RELEASE_PREDICTION)
            and self.target is not None
            and len(self.target.strip()) > 0
        )

    def summary(self) -> str:
        """Restituisce un riepilogo testuale strutturato conforme allo stile Makima."""
        lines = [
            f"Query Originale:  \"{self.raw_query}\"",
            f"Intent:           {self.intent.value}",
            f"Target:           {self.target if self.target else '[None]'}",
            f"Temporal Window:  {self.temporal_window}",
            f"Parser Conf:      {self.confidence * 100:.1f}%",
            f"Valida per Core:  {'Si' if self.is_valid_forecast else 'No'}",
        ]
        return "\n".join(lines)
