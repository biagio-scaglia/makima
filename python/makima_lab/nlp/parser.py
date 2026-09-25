"""Parser semantico avanzato con supporto Sentence-Transformers ed estrazione intelligente per ForecastQuery."""

import re
from typing import List, Optional
from makima_lab.nlp.models import ForecastQuery, Intent
from makima_lab.nlp.temporal import extract_temporal_window
from makima_lab.embeddings import get_embedder

# Pattern per query puramente estetiche o chitchat da rifiutare esplicitamente
UNSUPPORTED_PATTERNS = [
    r"\bquanto\s+[eè]\s+(?:bello|bravo|forte|carino|buono|simpatico|brutto|intelligente)\b",
    r"\bcome\s+stai\b",
    r"\bchi\s+sei\b",
    r"\bcosa\s+fai\b",
    r"\braccontami\s+(?:una|un)\b",
    r"\bciao\b",
    r"\bbuongiorno\b",
    r"\bbuonasera\b",
]

# Pattern per intenti previsionali (singolari, plurali, verbi al futuro e interrogative)
FORECAST_PATTERNS = [
    r"\bquando\b",
    r"\bqual\s+[eè]\s+la\s+probabilit[aà]\b",
    r"\bquanto\s+[eè]\s+probabile\b",
    r"\bprevedi\b",
    r"\bprevisione\b",
    r"\briuscir[a-zà-öø-ÿ]*\b",  # riuscirò, riusciremo, riuscirà, riuscirai, riusciranno
    r"\bprobabilit[aà]\b",
    r"\bforecast\b",
    r"\bce\s+la\s+faremo\b",
    r"\bfaremo\s+in\s+tempo\b",
    r"\barriver[a-zà-öø-ÿ]*\b",
    r"\bfinir[a-zà-öø-ÿ]*\b",
    r"\bcompleter[a-zà-öø-ÿ]*\b",
    r"\bavremo\b",
    r"\bsar[aà]\s+(?:pront[ao]|rilasciat[ao]|completat[ao])\b",
    r"\b[eè]\s+possibile\b",
    r"\bstima\b",
    r"\btempo\s+stimato\b",
]

# Pattern di release e consegna software
RELEASE_PATTERNS = [
    r"\brilasc[a-zà-öø-ÿ]*\b",  # rilasciare, rilasceremo, rilascerò, rilascio, rilasciato, rilascerà, rilasceranno
    r"\brelease\b",
    r"\bdeploy[a-zà-öø-ÿ]*\b",
    r"\bpubblic[a-zà-öø-ÿ]*\b",
    r"\bspedir[a-zà-öø-ÿ]*\b",
    r"\blanci[a-zà-öø-ÿ]*\b",
    r"\bconsegn[a-zà-öø-ÿ]*\b",
    r"\bcomplet[a-zà-öø-ÿ]*\b",
]

# Mappatura semantica canonica di parole chiave e descrittori per target noti
TARGET_SEMANTIC_MAP: dict[str, list[str]] = {
    "git:feature_ratio": [
        "feature", "nuova feature", "nuove feature", "funzionalita", "funzionalità",
        "feature ratio", "sviluppo feature", "nuovo modulo", "shipping", "features"
    ],
    "git:test_discipline": [
        "test", "unit test", "test suite", "disciplina test", "coverage", "copertura",
        "regressione", "qualita", "qualità", "testing", "unit test", "unit test suite"
    ],
    "git:commit_frequency": [
        "commit", "frequenza commit", "attivita", "attività", "push", "ritmo", "cadenza commit"
    ],
    "git:bugfix_ratio": [
        "bugfix", "bug", "correzione", "fix", "riparazione", "patch", "errori", "risoluzione bug"
    ],
    "git:chore_ratio": [
        "chore", "manutenzione", "configurazione", "pulizia", "refactoring", "ci"
    ],
    "framework_release": [
        "framework", "framework release", "core release", "versione framework"
    ],
    "daily_build": [
        "daily build", "build giornaliera", "compilazione", "ci build", "build"
    ],
    "deploy": [
        "deploy", "deployment", "produzione", "rilascio produzione"
    ],
    "api_gateway": [
        "api gateway", "gateway", "endpoint", "api"
    ],
}


class SemanticQueryParser:
    """Parser semantico avanzato per query dirette al motore probabilistico Makima."""

    def __init__(self) -> None:
        self.embedder = get_embedder()

    @staticmethod
    def normalize_text(text: str) -> str:
        """Pulisce e normalizza il testo di input."""
        cleaned = text.lower().strip()
        cleaned = cleaned.replace("'", " ").replace("’", " ")
        cleaned = re.sub(r"[^\w\s\-_]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def parse(self, text: str, available_targets: Optional[List[str]] = None) -> ForecastQuery:
        """Elabora il testo in linguaggio naturale e genera una ForecastQuery strutturata."""
        normalized = self.normalize_text(text)

        # 1. Verifica se la frase appartiene a intenti non supportati (chitchat/estetica)
        for pattern in UNSUPPORTED_PATTERNS:
            if re.search(pattern, normalized):
                temporal = extract_temporal_window(normalized)
                return ForecastQuery(
                    raw_query=text,
                    intent=Intent.UNSUPPORTED,
                    target=None,
                    temporal_window=temporal,
                    confidence=0.95,
                )

        # 2. Identificazione dell'Intent
        intent = Intent.UNSUPPORTED
        confidence = 0.5

        is_forecast = any(re.search(p, normalized) for p in FORECAST_PATTERNS)
        is_release = any(re.search(p, normalized) for p in RELEASE_PATTERNS)

        if is_forecast and is_release:
            intent = Intent.RELEASE_PREDICTION
            confidence = 0.92
        elif is_release:
            intent = Intent.RELEASE_PREDICTION
            confidence = 0.88
        elif is_forecast:
            intent = Intent.FORECAST
            confidence = 0.85

        # 3. Estrazione Semantica del Target
        target = self._extract_target(normalized, available_targets=available_targets)

        # Se il target è stato trovato e la query ha un contesto di domanda/consegna/futuro, promuovi a forecast
        if target is not None and intent == Intent.UNSUPPORTED:
            if is_release or is_forecast or "?" in text or any(k in normalized for k in ["quando", "se", "entro", "settimana", "mese"]):
                intent = Intent.RELEASE_PREDICTION if is_release else Intent.FORECAST
                confidence = 0.80

        # Se non c'è intenzione previsionale e non c'è target valido, è Unsupported
        if intent == Intent.UNSUPPORTED or (target is None and not is_forecast and not is_release):
            intent = Intent.UNSUPPORTED
            confidence = 0.90
        elif target is None and (is_forecast or is_release):
            # Se l'intento è di release generica e nessun target è nominato, seleziona il target preferenziale di release
            if available_targets:
                if "git:feature_ratio" in available_targets:
                    target = "git:feature_ratio"
                elif "framework_release" in available_targets:
                    target = "framework_release"
                else:
                    target = available_targets[0]
            else:
                target = "git:feature_ratio"

        # 4. Estrazione della Finestra Temporale
        temporal_window = extract_temporal_window(normalized)

        return ForecastQuery(
            raw_query=text,
            intent=intent,
            target=target,
            temporal_window=temporal_window,
            confidence=confidence,
        )

    def _extract_target(self, normalized: str, available_targets: Optional[List[str]] = None) -> Optional[str]:
        """Estrae l'entità target dal testo tramite regole lessicali e similarità d'incorporamento vettoriale."""
        # 1. Corrispondenza diretta esatta con target disponibili o target noti
        all_candidate_keys = list(available_targets) if available_targets else []
        for target_id in all_candidate_keys:
            if target_id.lower() in normalized:
                return target_id

        # 2. Controllo mappa semantica diretta
        for target_id, keywords in TARGET_SEMANTIC_MAP.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", normalized):
                    # Se il target è presente nei target attivi o se non ci sono vincoli, restituiscilo
                    if available_targets is None or target_id in available_targets:
                        return target_id
                    # Se chiedeva ad es. framework e abbiamo solo git, o viceversa, restituisci comunque target_id
                    return target_id

        # 3. Controllo costrutti verbali: "prevedi se/che [target]", "rilasciare [target]"
        verb_match = re.search(
            r"\b(?:prevedi|previsione|rilasc[a-zà-öø-ÿ]*|svilupp[a-zà-öø-ÿ]*|target)\s+(?:se|che|il|la|un|una|del|della|di)?\s*([a-z0-9_\-]+)\b",
            normalized,
        )
        if verb_match:
            candidate = verb_match.group(1).strip()
            stop_words = {
                "entro", "quando", "se", "che", "prossimo", "prossima",
                "questa", "questo", "non", "il", "la", "un", "una",
                "verra", "verrà", "sia", "venga", "nuovo", "nuova"
            }
            if candidate not in stop_words:
                if candidate in ("framework", "core"):
                    return "framework_release"
                if candidate in ("feature", "features"):
                    return "git:feature_ratio"
                if candidate in ("test", "testing"):
                    return "git:test_discipline"
                return candidate

        # 4. Risoluzione Semantica con Embedding (Sentence-Transformers / MiniLM)
        targets_to_evaluate = list(TARGET_SEMANTIC_MAP.keys())
        if available_targets:
            for t in available_targets:
                if t not in targets_to_evaluate:
                    targets_to_evaluate.append(t)

        best_target = None
        best_similarity = -1.0

        for t in targets_to_evaluate:
            # Crea un testo di riferimento semantico per il target
            descriptors = TARGET_SEMANTIC_MAP.get(t, [t])
            target_context = f"{t} {' '.join(descriptors)}"
            sim = self.embedder.similarity(normalized, target_context)
            if sim > best_similarity:
                best_similarity = sim
                best_target = t

        # Soglia di confidenza semantica minima per l'assegnazione automatica del target
        if best_similarity >= 0.28 and best_target is not None:
            return best_target

        return None

