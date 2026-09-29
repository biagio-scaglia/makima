"""Cervello operativo Makima: rete neurale + memoria + azioni (FORECASTING_PATH=False).

Il BrainLoop unifica NLP StructuredIntent, MakimaMindNet e deliberazione.
Le probabilità di produzione restano in Rust (`makima query`).
"""

from __future__ import annotations

from makima_lab.brain.loop import BrainAction, BrainLoop, BrainTickResult, get_brain

EXPERIMENTAL_LAB = True
FORECASTING_PATH = False

__all__ = [
    "EXPERIMENTAL_LAB",
    "FORECASTING_PATH",
    "BrainAction",
    "BrainLoop",
    "BrainTickResult",
    "get_brain",
]
