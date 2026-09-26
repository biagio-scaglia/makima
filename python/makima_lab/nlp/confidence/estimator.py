"""Modulo di stima composita della confidenza semantica basato su metriche di pipeline."""

from typing import Dict, Tuple
from makima_lab.nlp.schemas.intent import Intent
from makima_lab.nlp.schemas.temporal import TemporalRelation, TemporalWindow


class ConfidenceEstimator:
    """Calcola un punteggio di confidenza trasparente e non simulato aggregando segnali reali."""

    @classmethod
    def estimate(
        cls,
        intent: Intent,
        intent_score: float,
        target: str | None,
        target_score: float,
        temporal_window: TemporalWindow,
    ) -> Tuple[float, Dict[str, float]]:
        """Calcola la confidenza composita pesata.
        
        Formula:
            Confidence = w_intent * S_intent + w_target * S_target + w_temporal * S_temporal - P_coherence
        
        Returns:
            (composite_confidence, breakdown_dict)
        """
        # 1. Punteggio temporale
        if temporal_window.relation in (TemporalRelation.SPECIFIC_DATE, TemporalRelation.RELATIVE_INTERVAL):
            temporal_score = 1.0
        elif temporal_window.relation in (TemporalRelation.PAST, TemporalRelation.PRESENT, TemporalRelation.FUTURE):
            temporal_score = 0.85
        else:
            temporal_score = 0.50

        # 2. Punteggio del target
        t_score = target_score if target is not None else (0.40 if intent == Intent.STATUS else 0.10)

        # 3. Penalità di coerenza sintattica
        coherence_penalty = 0.0
        if intent in (Intent.QUERY, Intent.TEMPORAL_QUERY) and target is None:
            coherence_penalty = 0.25

        if intent == Intent.UNKNOWN:
            composite = max(0.10, intent_score * 0.70)
            breakdown = {
                "intent_signal": intent_score,
                "target_signal": 0.0,
                "temporal_signal": temporal_score,
                "coherence_penalty": 0.0,
            }
            return round(composite, 4), breakdown

        # Pesi della combinazione convessa
        w_intent = 0.40
        w_target = 0.40
        w_temporal = 0.20

        raw_score = (w_intent * intent_score) + (w_target * t_score) + (w_temporal * temporal_score) - coherence_penalty
        composite = min(0.99, max(0.05, raw_score))

        breakdown = {
            "intent_signal": round(intent_score, 4),
            "target_signal": round(t_score, 4),
            "temporal_signal": round(temporal_score, 4),
            "coherence_penalty": round(coherence_penalty, 4),
        }

        return round(composite, 4), breakdown
