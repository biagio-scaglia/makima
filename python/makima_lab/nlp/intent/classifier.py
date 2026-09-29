"""Classificatore di intenti: pattern lessicali + prototype matching semantico."""

from __future__ import annotations
import re
from typing import Dict, Tuple
from makima_lab.nlp.schemas.intent import Intent
from makima_lab.nlp.embeddings.representation import SemanticRepresentation


class IntentClassifier:
    """Combina regex tipizzati e similarità coseno verso prototipi di intento."""

    INTENT_PATTERNS: Dict[Intent, list[str]] = {
        Intent.TEMPORAL_QUERY: [
            r"\bquando\b",
            r"\bin\s+che\s+data\b",
            r"\bquale\s+giorno\b",
            r"\ba\s+che\s+ora\b",
            r"\bentro\s+quando\b",
            r"\bcosa\s+(?:sto|stiamo)\s+(?:facendo|sviluppendo)\b",
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
            r"\bdi\s+cosa\s+mi\s+piace\b",
            r"\bcosa\s+mi\s+piace\b",
            r"\bcose?\s+ti\s+ho\s+(?:detto|confidato)\b",
            r"\bricordi\b",
            r"\bcosa\s+(?:sai|ricordi)\s+(?:di|su)\b",
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

    # Frasi prototipo per matching semantico (complemento ai regex).
    INTENT_PROTOTYPES: Dict[Intent, list[str]] = {
        Intent.TEMPORAL_QUERY: [
            "quando rilascerò il prossimo framework",
            "in che data avverrà il deploy",
            "quando ho completato i test",
            "entro quando finiamo il rilascio",
            "quando uscirà la prossima versione",
        ],
        Intent.QUERY: [
            "qual è la probabilità di successo del deploy",
            "quanto è probabile che la build passi",
            "prevedi se riusciremo a rilasciare",
            "ce la faremo a chiudere la feature",
            "stima la probabilità del framework release",
        ],
        Intent.COMMAND: [
            "sincronizza i commit di git",
            "aggiorna e ricalcola i prior",
            "avvia la calibrazione del motore",
        ],
        Intent.INFORMATION: [
            "cosa significa il brier score",
            "come funziona l aggiornamento bayesiano",
            "spiega la distribuzione beta",
            "chi sei makima",
            "di cosa mi piace lavorare",
            "cosa ti ho detto sui miei gusti",
            "ricordi cosa mi piace",
        ],
        Intent.OBSERVATION: [
            "oggi ho completato il deploy con successo",
            "ho registrato un fallimento della build",
            "rilascio effettuato in produzione",
        ],
        Intent.STATUS: [
            "qual è lo stato del sistema",
            "mostra la diagnostica del motore",
            "riepilogo status makima",
        ],
    }

    UNKNOWN_PATTERNS = [
        r"\bcome\s+stai\b",
        r"\braccontami\s+(?:una|un)\s+barzelletta\b",
        r"\bciao\b",
        r"\bbuongiorno\b",
        r"\bbuonasera\b",
        r"\bche\s+tempo\s+fa\b",
        r"\bsei\s+bello\b",
    ]

    SEMANTIC_WEIGHT = 0.55
    SEMANTIC_MIN_SIM = 0.38

    def __init__(self, representation: SemanticRepresentation | None = None) -> None:
        self.repr = representation or SemanticRepresentation()

    def classify(self, text: str) -> Tuple[Intent, float, Dict[str, float]]:
        """Classifica il testo: regex + boost semantico sui prototipi."""
        normalized = text.lower().strip()

        for p in self.UNKNOWN_PATTERNS:
            if re.search(p, normalized):
                return Intent.UNKNOWN, 0.95, {"unknown_rejection": 0.95}

        scores: Dict[Intent, float] = {
            Intent.QUERY: 0.0,
            Intent.TEMPORAL_QUERY: 0.0,
            Intent.COMMAND: 0.0,
            Intent.INFORMATION: 0.0,
            Intent.OBSERVATION: 0.0,
            Intent.STATUS: 0.0,
            Intent.UNKNOWN: 0.1,
        }

        # 1. Pattern lessicali
        for intent, patterns in self.INTENT_PATTERNS.items():
            for p in patterns:
                if re.search(p, normalized):
                    scores[intent] += 0.45

        if re.search(r"\bquando\b", normalized):
            if any(
                k in normalized
                for k in ["rilasc", "release", "deploy", "finir", "completer", "arriver"]
            ):
                scores[Intent.TEMPORAL_QUERY] += 0.35
                scores[Intent.QUERY] += 0.25
            else:
                scores[Intent.TEMPORAL_QUERY] += 0.50

        if re.search(r"\bprobabilit[aà]|preved|forecast\b", normalized):
            scores[Intent.QUERY] += 0.40

        # 2. Prototype matching semantico: boost solo sul miglior intento
        semantic_hits: Dict[str, float] = {}
        best_sem_intent: Intent | None = None
        best_sem_score = 0.0
        for intent, prototypes in self.INTENT_PROTOTYPES.items():
            best_sim = 0.0
            for proto in prototypes:
                try:
                    sim = float(self.repr.similarity(normalized, proto))
                except Exception:
                    sim = 0.0
                if sim > best_sim:
                    best_sim = sim
            semantic_hits[intent.value] = round(best_sim, 4)
            if best_sim > best_sem_score:
                best_sem_score = best_sim
                best_sem_intent = intent

        if best_sem_intent is not None and best_sem_score >= self.SEMANTIC_MIN_SIM:
            scores[best_sem_intent] += self.SEMANTIC_WEIGHT * best_sem_score

        total = sum(scores.values())
        prob_breakdown = {
            k.value: round(v / total, 4) if total > 0 else 0.0 for k, v in scores.items()
        }
        for k, v in semantic_hits.items():
            prob_breakdown[f"semantic_{k}"] = v

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        if best_score < 0.35:
            return Intent.UNKNOWN, 0.50, prob_breakdown

        # Confidenza: mix tra quota relativa e score assoluto; floor se pattern forti
        relative = prob_breakdown.get(best_intent.value, 0.5)
        confidence = min(0.98, max(0.40, 0.55 * relative + 0.30 * min(1.0, best_score)))
        if best_score >= 0.70:
            confidence = max(confidence, 0.62)
        return best_intent, confidence, prob_breakdown
