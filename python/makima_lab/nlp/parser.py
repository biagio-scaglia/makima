"""Parser semantico avanzato multi-livello per interrogazioni NLP in Makima."""

from __future__ import annotations
import time
from typing import List, Optional
from makima_lab.nlp.schemas.intent import Intent
from makima_lab.nlp.schemas.temporal import TemporalRelation, TemporalWindow
from makima_lab.nlp.schemas.structured_intent import StructuredIntent
from makima_lab.nlp.preprocessing.cleaner import TextCleaner
from makima_lab.nlp.preprocessing.tokenizer import SimpleTokenizer
from makima_lab.nlp.embeddings.representation import SemanticRepresentation
from makima_lab.nlp.intent.classifier import IntentClassifier
from makima_lab.nlp.entities.target_extractor import TargetExtractor
from makima_lab.nlp.temporal.analyzer import TemporalAnalyzer
from makima_lab.nlp.confidence.estimator import ConfidenceEstimator
from makima_lab.nlp.context.memory import ConversationContext, ConversationTurn
from makima_lab.nlp.validation.validator import IntentValidator
from makima_lab.nlp.models import ForecastQuery, Intent as LegacyIntent, TemporalRelation as LegacyTemporalRelation, TemporalWindow as LegacyTemporalWindow


class SemanticQueryParser:
    """Parser semantico multi-stadio basato su componenti specializzati e rappresentazione vettoriale."""

    def __init__(self, context: ConversationContext | None = None) -> None:
        self.cleaner = TextCleaner()
        self.tokenizer = SimpleTokenizer()
        self.representation = SemanticRepresentation()
        self.intent_classifier = IntentClassifier(self.representation)
        self.target_extractor = TargetExtractor(self.representation)
        self.temporal_analyzer = TemporalAnalyzer()
        self.confidence_estimator = ConfidenceEstimator()
        self.validator = IntentValidator()
        self.context = context or ConversationContext()

    @staticmethod
    def normalize_text(text: str) -> str:
        """Pulisce e normalizza il testo di input tramite TextCleaner."""
        return TextCleaner.clean(text)

    def parse_structured(
        self,
        text: str,
        available_targets: Optional[List[str]] = None,
        use_context: bool = True,
    ) -> StructuredIntent:
        """Esegue l'intera pipeline di parsing semantico strutturato."""
        # 1. Preprocessing
        cleaned = self.cleaner.clean(text)
        if not cleaned:
            return StructuredIntent(
                raw_query=text,
                intent=Intent.UNKNOWN,
                target=None,
                temporal_window=TemporalWindow.unspecified(),
                confidence=0.0,
                is_valid_for_core=False,
                validation_notes=["Input vuoto o non valido."],
            )

        # 2. Tokenizzazione
        tokens = self.tokenizer.tokenize(cleaned)

        # 3. Intent Classification
        intent, intent_score, breakdown = self.intent_classifier.classify(cleaned)

        # 4. Target & Entity Extraction
        target, entities, target_score, match_method = self.target_extractor.extract(
            cleaned, available_targets=available_targets
        )

        # 5. Risoluzione anaforica del contesto conversazionale (se target assente)
        if use_context and target is None:
            resolved_target = self.context.resolve_anaphora(cleaned, current_target=None)
            if resolved_target:
                target = resolved_target
                target_score = 0.75
                match_method = "context_anaphora"

        # 6. Temporal Understanding
        temporal_window = self.temporal_analyzer.analyze(cleaned)

        # 7. Confidence Estimation (composita su segnali reali)
        confidence, conf_breakdown = self.confidence_estimator.estimate(
            intent=intent,
            intent_score=intent_score,
            target=target,
            target_score=target_score,
            temporal_window=temporal_window,
        )
        conf_breakdown["target_match_method"] = match_method

        # 8. Validation & Guardrails
        is_valid, validation_notes = self.validator.validate(
            intent=intent,
            target=target,
            temporal_window=temporal_window,
            confidence=confidence,
        )

        structured = StructuredIntent(
            raw_query=text,
            intent=intent,
            target=target,
            temporal_window=temporal_window,
            entities=entities,
            confidence=confidence,
            confidence_breakdown=conf_breakdown,
            is_valid_for_core=is_valid,
            validation_notes=validation_notes,
        )

        # Registra nel contesto conversazionale per i turni successivi
        if use_context:
            self.context.append_turn(
                ConversationTurn(
                    raw_query=text,
                    structured_intent=structured,
                    timestamp_sec=time.time(),
                )
            )

        return structured

    def parse(
        self,
        text: str,
        available_targets: Optional[List[str]] = None,
    ) -> ForecastQuery:
        """Compatibilità legacy: restituisce un oggetto ForecastQuery conforme alle API storiche."""
        struct = self.parse_structured(text, available_targets=available_targets)

        # Mappatura verso i tipi legacy
        legacy_intent = LegacyIntent.UNSUPPORTED
        low = text.lower()
        if struct.intent in (Intent.QUERY, Intent.TEMPORAL_QUERY):
            if (struct.target and "release" in struct.target) or "rilasc" in low or "prossimo framework" in low:
                legacy_intent = LegacyIntent.RELEASE_PREDICTION
            elif struct.target is not None:
                legacy_intent = LegacyIntent.FORECAST
            else:
                legacy_intent = LegacyIntent.UNSUPPORTED
        elif struct.intent == Intent.OBSERVATION:
            legacy_intent = LegacyIntent.OBSERVATION_RECORD
        elif struct.intent == Intent.STATUS:
            legacy_intent = LegacyIntent.STATUS_QUERY

        # Mappatura finestra temporale legacy
        if "settimana" in low:
            legacy_rel = LegacyTemporalRelation.THIS_WEEK
        elif "mese" in low:
            legacy_rel = LegacyTemporalRelation.THIS_MONTH
        elif struct.temporal_window.relation == TemporalRelation.RELATIVE_INTERVAL and struct.temporal_window.days:
            legacy_rel = LegacyTemporalRelation.WITHIN_DAYS
        elif struct.temporal_window.relation in (TemporalRelation.FUTURE, TemporalRelation.PRESENT):
            legacy_rel = LegacyTemporalRelation.NEXT
        elif struct.temporal_window.relation == TemporalRelation.SPECIFIC_DATE or struct.temporal_window.boundary:
            legacy_rel = LegacyTemporalRelation.BEFORE
        else:
            legacy_rel = LegacyTemporalRelation.UNSPECIFIED

        legacy_window = LegacyTemporalWindow(
            relation=legacy_rel,
            boundary=struct.temporal_window.boundary or struct.temporal_window.raw_expression,
            days=struct.temporal_window.days,
        )

        return ForecastQuery(
            raw_query=text,
            intent=legacy_intent,
            target=struct.target,
            temporal_window=legacy_window,
            confidence=struct.confidence,
        )
