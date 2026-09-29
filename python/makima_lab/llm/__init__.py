"""Makima LLM Engine: Qwen 2.5 0.5B Instruct — solo explain/digest grounded (sperimentale).

Non estrae StructuredIntent e non produce probabilità di forecasting.
I numeri devono arrivare dallo store / core Rust.
"""

from makima_lab.llm.engine import LlmResponse, QwenCognitiveEngine, get_llm_engine

EXPERIMENTAL_LAB = True
FORECASTING_PATH = False

__all__ = [
    "EXPERIMENTAL_LAB",
    "FORECASTING_PATH",
    "LlmResponse",
    "QwenCognitiveEngine",
    "get_llm_engine",
]
