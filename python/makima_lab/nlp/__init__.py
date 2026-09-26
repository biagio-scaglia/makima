"""Modulo NLP neurale/semantico di Makima Lab.

Fornisce la pipeline multi-livello per la comprensione del linguaggio,
classificazione di intenti, estrazione di target/entità, comprensione temporale,
stima della confidenza e validazione strutturata prima dell'interfaccia verso il Core Rust.
"""

from makima_lab.nlp.schemas.intent import Intent
from makima_lab.nlp.schemas.temporal import TemporalRelation, TemporalWindow
from makima_lab.nlp.schemas.structured_intent import StructuredIntent
from makima_lab.nlp.preprocessing.cleaner import TextCleaner
from makima_lab.nlp.preprocessing.tokenizer import SimpleTokenizer
from makima_lab.nlp.embeddings.representation import SemanticRepresentation
from makima_lab.nlp.intent.classifier import IntentClassifier
from makima_lab.nlp.entities.target_extractor import TargetExtractor
from makima_lab.nlp.temporal.analyzer import TemporalAnalyzer
from makima_lab.nlp.context.memory import ConversationContext
from makima_lab.nlp.confidence.estimator import ConfidenceEstimator
from makima_lab.nlp.validation.validator import IntentValidator
from makima_lab.nlp.parser import SemanticQueryParser
from makima_lab.nlp.pipeline import (
    MakimaNLPPipeline,
    SemanticForecastPipeline,
    SemanticForecastResult,
)
from makima_lab.nlp.models import ForecastQuery

__all__ = [
    "Intent",
    "TemporalRelation",
    "TemporalWindow",
    "StructuredIntent",
    "TextCleaner",
    "SimpleTokenizer",
    "SemanticRepresentation",
    "IntentClassifier",
    "TargetExtractor",
    "TemporalAnalyzer",
    "ConversationContext",
    "ConfidenceEstimator",
    "IntentValidator",
    "SemanticQueryParser",
    "MakimaNLPPipeline",
    "SemanticForecastPipeline",
    "SemanticForecastResult",
    "ForecastQuery",
]
