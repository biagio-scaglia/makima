"""Package schemas per il modulo NLP di Makima."""

from makima_lab.nlp.schemas.intent import Intent
from makima_lab.nlp.schemas.temporal import TemporalRelation, TemporalWindow
from makima_lab.nlp.schemas.structured_intent import StructuredIntent

__all__ = [
    "Intent",
    "TemporalRelation",
    "TemporalWindow",
    "StructuredIntent",
]
