"""
Makima Neural Mind Module (LABORATORIO SPERIMENTALE).

Path parallelo di ispezione (`makima neural`). NON è il parser NLP di produzione
e NON calcola probabilità per il core Rust.
"""

from .vocab import MakimaTokenizer
from .models import MakimaMindNet, INTENTS, INTENT2IDX, IDX2INTENT, ACTIONS, ACTION2IDX, IDX2ACTION
from .engine import MakimaNeuralEngine, NeuralInferenceResult, get_neural_engine

EXPERIMENTAL_LAB = True
FORECASTING_PATH = False

__all__ = [
    "EXPERIMENTAL_LAB",
    "FORECASTING_PATH",
    "MakimaTokenizer",
    "MakimaMindNet",
    "MakimaNeuralEngine",
    "NeuralInferenceResult",
    "get_neural_engine",
    "INTENTS",
    "INTENT2IDX",
    "IDX2INTENT",
    "ACTIONS",
    "ACTION2IDX",
    "IDX2ACTION",
]
