"""Pipeline semantica end-to-end per la risoluzione e il calcolo probabilistico da NL."""

import math
import sys
from dataclasses import dataclass

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from makima_lab.distributions import Bernoulli, BetaDistribution, PoissonDistribution
from makima_lab.nlp.models import ForecastQuery, Intent, TemporalRelation
from makima_lab.nlp.parser import SemanticQueryParser


# Dataset storico di evidenze di riferimento per i target noti nel laboratorio
DEFAULT_KNOWLEDGE_BASE = {
    "framework_release": {
        "successes": 6,
        "failures": 2,
        "historical_days": 60,
        "rate_per_day": 6.0 / 60.0,  # ~0.10 rilasci/giorno (1 ogni ~10 giorni)
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
}


@dataclass(frozen=True)
class SemanticForecastResult:
    """Risultato completo dell'elaborazione semantica e inferenziale."""
    query: ForecastQuery
    posterior: BetaDistribution | None
    temporal_probability: float | None
    entropy_bits: float | None
    explanation: str

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
        self.knowledge_base = knowledge_base or DEFAULT_KNOWLEDGE_BASE

    def execute(self, text: str) -> SemanticForecastResult:
        """Esegue l'intero flusso di comprensione ed elaborazione matematica."""
        query = self.parser.parse(text)

        if not query.is_valid_forecast:
            return SemanticForecastResult(
                query=query,
                posterior=None,
                temporal_probability=None,
                entropy_bits=None,
                explanation="La richiesta non contiene un intento previsionale valido o un target riconoscibile.",
            )

        target = query.target or "framework_release"
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

        # Ragionamento temporale analitico se specificato nella query
        temporal_prob = None
        rate = evidence.get("rate_per_day", 0.1)

        rel = query.temporal_window.relation
        if rel == TemporalRelation.WITHIN_DAYS and query.temporal_window.days:
            # Modello Poisson / Processo di Poisson P(X >= 1 in T giorni) = 1 - exp(-rate * T)
            days = query.temporal_window.days
            temporal_prob = 1.0 - math.exp(-rate * days)
        elif rel == TemporalRelation.THIS_WEEK:
            temporal_prob = 1.0 - math.exp(-rate * 7.0)
        elif rel == TemporalRelation.THIS_MONTH or (rel == TemporalRelation.BEFORE and query.temporal_window.boundary in ("dicembre", "december")):
            temporal_prob = 1.0 - math.exp(-rate * 30.0)
        elif rel == TemporalRelation.NEXT:
            temporal_prob = posterior.mean

        explanation = (
            f"Previsione aggiornata con {evidence['successes']} successi e {evidence['failures']} insuccessi storici."
        )

        return SemanticForecastResult(
            query=query,
            posterior=posterior,
            temporal_probability=temporal_prob,
            entropy_bits=entropy,
            explanation=explanation,
        )
