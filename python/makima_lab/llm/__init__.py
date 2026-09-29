"""Makima LLM Engine: Qwen 2.5 0.5B Instruct — solo explain/digest (sperimentale).

Non estrae StructuredIntent e non produce probabilità di forecasting.
"""

from makima_lab.llm.engine import QwenCognitiveEngine, get_llm_engine

EXPERIMENTAL_LAB = True
FORECASTING_PATH = False

__all__ = [
    "EXPERIMENTAL_LAB",
    "FORECASTING_PATH",
    "QwenCognitiveEngine",
    "get_llm_engine",
]
