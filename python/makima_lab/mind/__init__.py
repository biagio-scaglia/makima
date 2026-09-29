"""Modulo della Mente Cognitiva, Coscienza ed Esperienza Episodica di Makima.

STATO: **laboratorio sperimentale / UX narrativa**.
Non partecipa al path di forecasting di produzione (CLI `makima query` / GUI forecast).
I numeri probabilistici restano esclusiva di `makima-core` (Rust).
"""

from makima_lab.mind.schemas import (
    CognitiveMood,
    CognitiveExperience,
    CognitivePulse,
    EpistemicSelfState,
    MemoryCategory,
)
from makima_lab.mind.episodic_memory import EpisodicMemoryStore
from makima_lab.mind.deliberation import MindDeliberationEngine
from makima_lab.mind.pulse import AutonomousMindPulse

from makima_lab.mind.knowledge_graph import (
    KnowledgeNode,
    KnowledgeEdge,
    BrainGraph,
    SecondBrainBuilder,
)

# Marker esplicito: fuori dal runtime di forecasting.
EXPERIMENTAL_LAB = True
FORECASTING_PATH = False

__all__ = [
    "EXPERIMENTAL_LAB",
    "FORECASTING_PATH",
    "CognitiveMood",
    "CognitiveExperience",
    "CognitivePulse",
    "EpistemicSelfState",
    "MemoryCategory",
    "EpisodicMemoryStore",
    "MindDeliberationEngine",
    "AutonomousMindPulse",
    "KnowledgeNode",
    "KnowledgeEdge",
    "BrainGraph",
    "SecondBrainBuilder",
]

