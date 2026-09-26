"""Pipeline semantica end-to-end per la risoluzione e il calcolo probabilistico da NL."""

from __future__ import annotations
import math
import sys
from dataclasses import dataclass
from typing import Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from makima_lab.distributions import Bernoulli, BetaDistribution, PoissonDistribution
from makima_lab.nlp.models import ForecastQuery, Intent as LegacyIntent, TemporalRelation as LegacyTemporalRelation
from makima_lab.nlp.schemas.intent import Intent
from makima_lab.nlp.schemas.temporal import TemporalRelation, TemporalWindow
from makima_lab.nlp.schemas.structured_intent import StructuredIntent
from makima_lab.nlp.parser import SemanticQueryParser
from makima_lab.storage import compute_knowledge_base_from_store, load_store


# Dataset storico di riferimento predefinito per target base
DEFAULT_KNOWLEDGE_BASE = {
    "git:feature_ratio": {
        "successes": 15,
        "failures": 3,
        "historical_days": 14,
        "rate_per_day": 15.0 / 14.0,
    },
    "git:test_discipline": {
        "successes": 12,
        "failures": 2,
        "historical_days": 14,
        "rate_per_day": 12.0 / 14.0,
    },
    "framework_release": {
        "successes": 6,
        "failures": 2,
        "historical_days": 60,
        "rate_per_day": 6.0 / 60.0,
    },
    "daily_build": {
        "successes": 28,
        "failures": 2,
        "historical_days": 30,
        "rate_per_day": 28.0 / 30.0,
    },
    "deploy": {
        "successes": 15,
        "failures": 3,
        "historical_days": 30,
        "rate_per_day": 15.0 / 30.0,
    },
    "api_gateway": {
        "successes": 4,
        "failures": 1,
        "historical_days": 45,
        "rate_per_day": 4.0 / 45.0,
    },
    "git:commit_frequency": {
        "successes": 24,
        "failures": 1,
        "historical_days": 14,
        "rate_per_day": 24.0 / 14.0,
    },
    "git:bugfix_ratio": {
        "successes": 9,
        "failures": 2,
        "historical_days": 14,
        "rate_per_day": 9.0 / 14.0,
    },
    "git:chore_ratio": {
        "successes": 8,
        "failures": 1,
        "historical_days": 14,
        "rate_per_day": 8.0 / 14.0,
    },
}


@dataclass(frozen=True)
class SemanticForecastResult:
    """Risultato completo dell'elaborazione semantica e inferenziale."""
    query: ForecastQuery
    posterior: BetaDistribution | None
    temporal_probability: float | None
    entropy_bits: float | None
    explanation: str
    structured_intent: StructuredIntent | None = None

    def format_report(self) -> str:
        """Formatta il report completo per l'output su console."""
        lines = [
            "============================================================",
            "           MAKIMA SEMANTIC FORECAST PIPELINE (NLP)          ",
            "============================================================",
            f"Query Utente:         \"{self.query.raw_query}\"",
            f"Intento Riconosciuto: {self.query.intent.value}",
            f"Target Identificato:  {self.query.target if self.query.target else '[Nessuno]'}",
            f"Finestra Temporale:   {self.query.temporal_window}",
            f"Confidenza Parser:    {self.query.confidence * 100:.1f}%",
            "------------------------------------------------------------",
        ]

        if not self.query.is_valid_forecast or self.posterior is None:
            lines.append("Stato Previsione:     RIFIUTATA / NON ESEGUIBILE")
            lines.append(f"Motivazione:          {self.explanation}")
            lines.append("============================================================")
            return "\n".join(lines)

        mean_p = self.posterior.mean
        lines.append(f"Probabilità Evento:   {mean_p * 100:.2f}%  (E[P] = {mean_p:.4f})")
        lines.append(f"Densità ASCII:        {self.posterior.ascii_density(32)}")
        lines.append(f"Incertezza Epistemica (Var): {self.posterior.variance:.6f}")

        if self.entropy_bits is not None:
            lines.append(f"Entropia Informativa: {self.entropy_bits:.4f} bit")

        if self.temporal_probability is not None:
            lines.append(
                f"Probabilità Temporale P(T <= window): {self.temporal_probability * 100:.2f}%"
            )

        lines.append(
            f"Parametri Posterior:  Beta(alpha={self.posterior.alpha:.2f}, beta={self.posterior.beta:.2f})"
        )
        lines.append(f"Valutazione Trasparente: {self.explanation}")
        lines.append("============================================================")
        return "\n".join(lines)


class SemanticForecastPipeline:
    """Pipeline integrata: Linguaggio Naturale -> Semantic Parser -> Motore Probabilistico."""

    def __init__(self, parser: SemanticQueryParser | None = None, knowledge_base: dict | None = None):
        self.parser = parser or SemanticQueryParser()
        if knowledge_base is not None:
            self.knowledge_base = knowledge_base
        else:
            # Carica dallo storage persistente (.makima/makima.db SQLite o store.json)
            store_data = load_store()
            loaded_kb = compute_knowledge_base_from_store(store_data)
            merged_kb = dict(DEFAULT_KNOWLEDGE_BASE)
            merged_kb.update(loaded_kb)
            self.knowledge_base = merged_kb

    def execute_structured(self, text: str) -> tuple[StructuredIntent, SemanticForecastResult]:
        """Elabora la query producendo lo StructuredIntent validato e il calcolo probabilistico."""
        available_targets = list(self.knowledge_base.keys())
        struct = self.parser.parse_structured(text, available_targets=available_targets)
        legacy_query = self.parser.parse(text, available_targets=available_targets)

        if not struct.is_valid_for_core or struct.target is None:
            res = SemanticForecastResult(
                query=legacy_query,
                posterior=None,
                temporal_probability=None,
                entropy_bits=None,
                explanation="; ".join(struct.validation_notes) or "Richiesta non valida per il core.",
                structured_intent=struct,
            )
            return struct, res

        target = struct.target
        evidence = self.knowledge_base.get(
            target,
            {"successes": 1, "failures": 1, "historical_days": 30, "rate_per_day": 1.0 / 30.0},
        )

        prior = BetaDistribution.uniform()
        posterior = prior.bayesian_update(
            successes=evidence["successes"],
            failures=evidence["failures"],
        )

        bernoulli = Bernoulli(posterior.mean)
        entropy = bernoulli.entropy_bits

        # Calcolo probabilistico temporale analitico
        temporal_prob = None
        rate = evidence.get("rate_per_day", 0.1)
        rel = struct.temporal_window.relation

        if rel == TemporalRelation.RELATIVE_INTERVAL and struct.temporal_window.days:
            days = struct.temporal_window.days
            temporal_prob = 1.0 - math.exp(-rate * days)
        elif rel == TemporalRelation.RELATIVE_INTERVAL:
            temporal_prob = 1.0 - math.exp(-rate * 7.0)
        elif rel == TemporalRelation.SPECIFIC_DATE:
            temporal_prob = 1.0 - math.exp(-rate * 30.0)
        elif rel == TemporalRelation.FUTURE:
            temporal_prob = posterior.mean
        elif rel == TemporalRelation.PRESENT:
            temporal_prob = 1.0
        else:
            temporal_prob = posterior.mean

        source_desc = "telemetria Git reale" if target.startswith("git:") else "evidenze storiche"
        explanation = (
            f"Previsione basata su {source_desc} (target '{target}'): {evidence['successes']} successi, "
            f"{evidence['failures']} insuccessi registrati, rate stimato ~{rate:.2f} eventi/giorno."
        )

        res = SemanticForecastResult(
            query=legacy_query,
            posterior=posterior,
            temporal_probability=temporal_prob,
            entropy_bits=entropy,
            explanation=explanation,
            structured_intent=struct,
        )
        return struct, res

    def process_intent(self, text: str) -> StructuredIntent:
        """Elabora la query testuale e restituisce direttamente lo StructuredIntent validato."""
        available_targets = list(self.knowledge_base.keys())
        return self.parser.parse_structured(text, available_targets=available_targets)

    def execute(self, text: str) -> SemanticForecastResult:
        """Esegue l'elaborazione restituendo il SemanticForecastResult."""
        _, res = self.execute_structured(text)
        return res

    def run(self, text: str) -> SemanticForecastResult:
        """Alias compatibile per l'esecuzione della pipeline."""
        return self.execute(text)


MakimaNLPPipeline = SemanticForecastPipeline
