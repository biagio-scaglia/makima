"""Pipeline NLP: contratto produzione = StructuredIntent; Bayes Python solo lab demo.

Produzione: `process_intent` → JSON → `makima query` (core Rust).
`execute` / `execute_structured` restano demo Laplace locale (FORECASTING_PATH=False).
"""

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

# Numeri di forecast di produzione: solo Rust.
FORECASTING_PATH = False


# Catalogo target noti solo per risoluzione semantica (NESSUNA evidenza inventata).
# Le conteggi empirici devono provenire esclusivamente dallo store persistente.
KNOWN_TARGET_CATALOG: tuple[str, ...] = (
    "git:feature_ratio",
    "git:test_discipline",
    "git:commit_frequency",
    "git:bugfix_ratio",
    "git:chore_ratio",
    "framework_release",
    "daily_build",
    "deploy",
    "api_gateway",
)


@dataclass(frozen=True)
class SemanticForecastResult:
    """Demo Laplace in Python — non sostituisce il forecast Rust."""
    query: ForecastQuery
    posterior: BetaDistribution | None
    temporal_probability: float | None
    entropy_bits: float | None
    explanation: str
    structured_intent: StructuredIntent | None = None
    lab_demo: bool = True

    def format_report(self) -> str:
        """Report lab; per produzione usare `makima query`."""
        lines = [
            "============================================================",
            "     MAKIMA NLP LAB DEMO (FORECASTING_PATH=False)           ",
            "============================================================",
            f"Query Utente:         \"{self.query.raw_query}\"",
            f"Intento Riconosciuto: {self.query.intent.value}",
            f"Target Identificato:  {self.query.target if self.query.target else '[Nessuno]'}",
            f"Finestra Temporale:   {self.query.temporal_window}",
            f"Confidenza Parser:    {self.query.confidence * 100:.1f}%",
            "Nota:                 numeri demo Python — produzione = Rust core",
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
        else:
            lines.append("Probabilità Temporale: non calcolata (tasso empirico assente)")

        lines.append(
            f"Parametri Posterior:  Beta(alpha={self.posterior.alpha:.2f}, beta={self.posterior.beta:.2f})"
        )
        lines.append(f"Valutazione Trasparente: {self.explanation}")
        lines.append("============================================================")
        return "\n".join(lines)


class SemanticForecastPipeline:
    """NLP → StructuredIntent (produzione) + demo Laplace locale (lab)."""

    FORECASTING_PATH = False

    def __init__(self, parser: SemanticQueryParser | None = None, knowledge_base: dict | None = None):
        self.parser = parser or SemanticQueryParser()
        if knowledge_base is not None:
            self.knowledge_base = knowledge_base
        else:
            store_data = load_store()
            self.knowledge_base = compute_knowledge_base_from_store(store_data)

    def _available_targets(self) -> list[str]:
        """Target candidati: unione di quelli osservati e del catalogo lessicale."""
        observed = list(self.knowledge_base.keys())
        catalog = list(KNOWN_TARGET_CATALOG)
        return list(dict.fromkeys([*observed, *catalog]))

    def execute_structured(self, text: str) -> tuple[StructuredIntent, SemanticForecastResult]:
        """Demo lab: StructuredIntent + Laplace Python (non produzione)."""
        available_targets = self._available_targets()
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
                lab_demo=True,
            )
            return struct, res

        target = struct.target
        evidence = self.knowledge_base.get(
            target,
            {"successes": 0, "failures": 0, "historical_days": 0, "rate_per_day": 0.0},
        )

        prior = BetaDistribution.uniform()
        posterior = prior.bayesian_update(
            successes=int(evidence["successes"]),
            failures=int(evidence["failures"]),
        )

        bernoulli = Bernoulli(posterior.mean)
        entropy = bernoulli.entropy_bits

        # Mai inventare rate=0.1: senza tasso empirico → niente Poisson finto.
        rate = float(evidence.get("rate_per_day", 0.0) or 0.0)
        rel = struct.temporal_window.relation
        temporal_prob = None

        if rate > 0.0:
            if rel == TemporalRelation.RELATIVE_DURATION and struct.temporal_window.days:
                temporal_prob = 1.0 - math.exp(-rate * float(struct.temporal_window.days))
            elif rel == TemporalRelation.RELATIVE_DURATION:
                temporal_prob = 1.0 - math.exp(-rate * 7.0)
            elif rel == TemporalRelation.SPECIFIC_DATE:
                temporal_prob = 1.0 - math.exp(-rate * 30.0)
            elif rel == TemporalRelation.FUTURE:
                temporal_prob = posterior.mean
            elif rel == TemporalRelation.PRESENT:
                temporal_prob = 1.0
            else:
                temporal_prob = posterior.mean
        elif rel == TemporalRelation.PRESENT:
            temporal_prob = 1.0

        source_desc = "telemetria Git reale" if target.startswith("git:") else "evidenze storiche"
        total_ev = int(evidence["successes"]) + int(evidence["failures"])
        lab_note = " [LAB DEMO — produzione: makima query]"
        if total_ev == 0:
            explanation = (
                f"Target '{target}' senza evidenze empiriche: prior uniforme Beta(1,1). "
                f"Registra con 'makima observe {target} <0|1>' o sync-git.{lab_note}"
            )
        else:
            rate_note = (
                f"rate empirico ~{rate:.2f}/giorno"
                if rate > 0
                else "tasso giornaliero assente (nessun Poisson inventato)"
            )
            explanation = (
                f"Demo Laplace su {source_desc} ('{target}'): "
                f"{evidence['successes']} successi, {evidence['failures']} insuccessi, {rate_note}.{lab_note}"
            )

        res = SemanticForecastResult(
            query=legacy_query,
            posterior=posterior,
            temporal_probability=temporal_prob,
            entropy_bits=entropy,
            explanation=explanation,
            structured_intent=struct,
            lab_demo=True,
        )
        return struct, res

    def process_intent(self, text: str) -> StructuredIntent:
        """Contratto di produzione: StructuredIntent validato (nessun Bayes)."""
        return self.parser.parse_structured(text, available_targets=self._available_targets())

    def execute(self, text: str) -> SemanticForecastResult:
        """Demo Laplace Python (lab). Preferire `process_intent` + `makima query`."""
        _, res = self.execute_structured(text)
        return res

    def run(self, text: str) -> SemanticForecastResult:
        """Alias lab di `execute`."""
        return self.execute(text)


MakimaNLPPipeline = SemanticForecastPipeline
