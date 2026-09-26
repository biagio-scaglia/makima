"""Dataset di benchmark ed esempi di ground truth per la valutazione semantica/NLP."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
from makima_lab.nlp.schemas.intent import Intent
from makima_lab.nlp.schemas.temporal import TemporalRelation


@dataclass(frozen=True)
class NLPEvaluationSample:
    """Singolo campione annotato per la validazione della pipeline NLP."""
    text: str
    expected_intent: Intent
    expected_target: str | None
    expected_temporal_relation: TemporalRelation
    expected_valid_for_core: bool


BENCHMARK_DATASET: List[NLPEvaluationSample] = [
    # 1. Query di previsione e rilascio
    NLPEvaluationSample(
        text="Quando rilascerò il prossimo framework?",
        expected_intent=Intent.TEMPORAL_QUERY,
        expected_target="framework_release",
        expected_temporal_relation=TemporalRelation.FUTURE,
        expected_valid_for_core=True,
    ),
    NLPEvaluationSample(
        text="Qual è la probabilità che il deploy in produzione riesca?",
        expected_intent=Intent.QUERY,
        expected_target="deploy",
        expected_temporal_relation=TemporalRelation.UNKNOWN,
        expected_valid_for_core=True,
    ),
    NLPEvaluationSample(
        text="Riusciremo a completare le nuove feature entro 7 giorni?",
        expected_intent=Intent.QUERY,
        expected_target="git:feature_ratio",
        expected_temporal_relation=TemporalRelation.RELATIVE_INTERVAL,
        expected_valid_for_core=True,
    ),
    NLPEvaluationSample(
        text="Prevedi se la daily build passerà questa settimana",
        expected_intent=Intent.QUERY,
        expected_target="daily_build",
        expected_temporal_relation=TemporalRelation.RELATIVE_INTERVAL,
        expected_valid_for_core=True,
    ),
    NLPEvaluationSample(
        text="Qual è la probabilità per api_gateway entro il 2026-12-31?",
        expected_intent=Intent.QUERY,
        expected_target="api_gateway",
        expected_temporal_relation=TemporalRelation.SPECIFIC_DATE,
        expected_valid_for_core=True,
    ),

    # 2. Query temporali pure
    NLPEvaluationSample(
        text="Quando ho completato i test di regressione?",
        expected_intent=Intent.TEMPORAL_QUERY,
        expected_target="git:test_discipline",
        expected_temporal_relation=TemporalRelation.PAST,
        expected_valid_for_core=True,
    ),
    NLPEvaluationSample(
        text="Cosa sto facendo attualmente nel repository?",
        expected_intent=Intent.TEMPORAL_QUERY,
        expected_target=None,
        expected_temporal_relation=TemporalRelation.PRESENT,
        expected_valid_for_core=False,  # Nessun target per query di previsione
    ),

    # 3. Comandi operativi
    NLPEvaluationSample(
        text="Sincronizza i commit di git",
        expected_intent=Intent.COMMAND,
        expected_target="git:commit_frequency",
        expected_temporal_relation=TemporalRelation.UNKNOWN,
        expected_valid_for_core=True,
    ),
    NLPEvaluationSample(
        text="Aggiorna i modelli di calibrazione",
        expected_intent=Intent.COMMAND,
        expected_target=None,
        expected_temporal_relation=TemporalRelation.UNKNOWN,
        expected_valid_for_core=True,
    ),

    # 4. Informazioni e spiegazioni
    NLPEvaluationSample(
        text="Spiega come funziona il Brier Score",
        expected_intent=Intent.INFORMATION,
        expected_target=None,
        expected_temporal_relation=TemporalRelation.UNKNOWN,
        expected_valid_for_core=True,
    ),
    NLPEvaluationSample(
        text="Cosa significa la distribuzione Beta posteriore?",
        expected_intent=Intent.INFORMATION,
        expected_target=None,
        expected_temporal_relation=TemporalRelation.UNKNOWN,
        expected_valid_for_core=True,
    ),

    # 5. Registrazione osservazioni
    NLPEvaluationSample(
        text="Oggi ho completato la suite di unit test con successo",
        expected_intent=Intent.OBSERVATION,
        expected_target="git:test_discipline",
        expected_temporal_relation=TemporalRelation.PAST,
        expected_valid_for_core=True,
    ),

    # 6. Stato diagnostico
    NLPEvaluationSample(
        text="Qual è lo stato diagnostico del motore?",
        expected_intent=Intent.STATUS,
        expected_target=None,
        expected_temporal_relation=TemporalRelation.UNKNOWN,
        expected_valid_for_core=True,
    ),

    # 7. Casi sconosciuti, chitchat e fuori dominio (devono essere rifiutati esplicitamente come UNKNOWN)
    NLPEvaluationSample(
        text="Ciao, come stai oggi?",
        expected_intent=Intent.UNKNOWN,
        expected_target=None,
        expected_temporal_relation=TemporalRelation.UNKNOWN,
        expected_valid_for_core=False,
    ),
    NLPEvaluationSample(
        text="Raccontami una barzelletta simpatica",
        expected_intent=Intent.UNKNOWN,
        expected_target=None,
        expected_temporal_relation=TemporalRelation.UNKNOWN,
        expected_valid_for_core=False,
    ),
    NLPEvaluationSample(
        text="xyz123 nonsensetext qwerty",
        expected_intent=Intent.UNKNOWN,
        expected_target=None,
        expected_temporal_relation=TemporalRelation.UNKNOWN,
        expected_valid_for_core=False,
    ),
]
