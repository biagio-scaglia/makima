"""Classificatore modulare di intenti semantici per Makima."""

from __future__ import annotations
import re
from typing import Dict, Tuple
from makima_lab.nlp.schemas.intent import Intent
from makima_lab.nlp.embeddings.representation import SemanticRepresentation


class IntentClassifier:
    """Classifica l'intento della query combinando pattern lessicali, indizi sintattici e similarità semantica."""

    # Pattern specifici per ciascun intento
    INTENT_PATTERNS: Dict[Intent, list[str]] = {
        Intent.TEMPORAL_QUERY: [
            r"\bquando\b",
            r"\bin\s+che\s+data\b",
            r"\bquale\s+giorno\b",
            r"\ba\s+che\s+ora\b",
            r"\bentro\s+quando\b",
            r"\bcosa\s+(?:sto|stiamo)\s+(?:facendo|sviluppando)\b",
            r"\bcosa\s+ho\s+fatto\b",
            r"\bcosa\s+abbiamo\s+fatto\b",
        ],
        Intent.QUERY: [
            r"\bqual\s+[eè]\s+la\s+probabilit[aà]\b",
            r"\bquanto\s+[eè]\s+probabile\b",
            r"\bprevedi\b",
            r"\bprevisione\b",
            r"\briuscir[a-zà-öø-ÿ]*\b",
            r"\bprobabilit[aà]\b",
            r"\bforecast\b",
            r"\bce\s+la\s+faremo\b",
            r"\bfaremo\s+in\s+tempo\b",
            r"\bfinir[a-zà-öø-ÿ]*\b",
            r"\bcompleter[a-zà-öø-ÿ]*\b",
            r"\bsar[aà]\s+(?:pront[ao]|rilasciat[ao]|completat[ao])\b",
            r"\b[eè]\s+possibile\s+che\b",
            r"\bstima\b",
            r"\brilasc[a-zà-öø-ÿ]*\b",
        ],
        Intent.COMMAND: [
            r"\baggiorna\b",
            r"\bsincronizza\b",
            r"\besegui\b",
            r"\bricalcola\b",
            r"\bresetta\b",
            r"\bavvia\b",
            r"\bferma\b",
            r"\brisolvi\b",
            r"\bcalibra\b",
        ],
        Intent.INFORMATION: [
            r"\bcosa\s+significa\b",
            r"\bcome\s+funziona\b",
            r"\bspiega\b",
            r"\bperch[eé]\b",
            r"\bcos\s+[eè]\s+(?:il|la|un|una)\b",
            r"\bchi\s+sei\b",
            r"\bcosa\s+sei\b",
            r"\bdescrivi\b",
        ],
        Intent.OBSERVATION: [
            r"\boggi\s+ho\s+(?:fatto|completato|rilasciato|eseguito|fallito)\b",
            r"\bho\s+(?:completato|fatto|chiuso|risolto|registrato|committato)\b",
            r"\b[eè]\s+(?:andato|fallito|riuscito)\b",
            r"\bnuova\s+osservazione\b",
            r"\brilascio\s+effettuato\b",
            r"\bbuild\s+(?:completata|fallita)\b",
            r"\btest\s+(?:passati|falliti)\b",
        ],
        Intent.STATUS: [
            r"\bqual\s+[eè]\s+lo\s+stato\b",
            r"\bstato\s+del\s+sistema\b",
            r"\bstato\s+motore\b",
            r"\briepilogo\b",
            r"\bdiagnostica\b",
            r"\bsummary\b",
            r"\bstatus\b",
        ],
    }

    # Frasi tipiche o chitchat fuori dominio da respingere direttamente come UNKNOWN
    UNKNOWN_PATTERNS = [
        r"\bcome\s+stai\b",
        r"\braccontami\s+(?:una|un)\s+barzelletta\b",
        r"\bciao\b",
        r"\bbuongiorno\b",
        r"\bbuonasera\b",
        r"\bche\s+tempo\s+fa\b",
        r"\bsei\s+bello\b",
    ]

    def __init__(self, representation: SemanticRepresentation | None = None) -> None:
        self.repr = representation or SemanticRepresentation()

    def classify(self, text: str) -> Tuple[Intent, float, Dict[str, float]]:
        """Classifica il testo in un Intent con probabilità e punteggi associati.
        
        Returns:
            (Intent, confidence_score, score_breakdown)
        """
        normalized = text.lower().strip()

        # Controllo esplicito chitchat / fuori dominio
        for p in self.UNKNOWN_PATTERNS:
            if re.search(p, normalized):
                return Intent.UNKNOWN, 0.95, {"unknown_rejection": 0.95}

        # Calcola evidenze per ciascun intento
        scores: Dict[Intent, float] = {
            Intent.QUERY: 0.0,
            Intent.TEMPORAL_QUERY: 0.0,
            Intent.COMMAND: 0.0,
            Intent.INFORMATION: 0.0,
            Intent.OBSERVATION: 0.0,
            Intent.STATUS: 0.0,
            Intent.UNKNOWN: 0.1,  # Baseline di incertezza
        }

        # 1. Analisi dei pattern sintattici
        for intent, patterns in self.INTENT_PATTERNS.items():
            for p in patterns:
                if re.search(p, normalized):
                    scores[intent] += 0.45

        # Regole di correlazione semantica
        if re.search(r"\bquando\b", normalized):
            if any(k in normalized for k in ["rilasc", "release", "deploy", "finir", "completer", "arriver"]):
                scores[Intent.TEMPORAL_QUERY] += 0.35
                scores[Intent.QUERY] += 0.25
            else:
                scores[Intent.TEMPORAL_QUERY] += 0.50

        if re.search(r"\bprobabilit[aà]|preved|forecast\b", normalized):
            scores[Intent.QUERY] += 0.40

        # Normalizzazione dei punteggi in probabilità relative
        total = sum(scores.values())
        prob_breakdown = {k.value: round(v / total, 4) if total > 0 else 0.0 for k, v in scores.items()}

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        # Se il punteggio massimo è troppo debole o non supera la soglia minima, classifica come UNKNOWN
        if best_score < 0.35:
            return Intent.UNKNOWN, 0.50, prob_breakdown

        confidence = min(0.98, max(0.40, prob_breakdown.get(best_intent.value, 0.5) * 1.3))
        return best_intent, confidence, prob_breakdown
