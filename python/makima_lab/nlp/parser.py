"""Parser semantico deterministico per trasformare linguaggio naturale in ForecastQuery."""

import re
from makima_lab.nlp.models import ForecastQuery, Intent
from makima_lab.nlp.temporal import extract_temporal_window

# Pattern per query puramente estetiche o chitchat da rifiutare esplicitamente
UNSUPPORTED_PATTERNS = [
    r"\bquanto\s+e\s+(?:bello|bravo|forte|carino|buono|simpatico)\b",
    r"\bcome\s+stai\b",
    r"\bchi\s+sei\b",
    r"\bcosa\s+fai\b",
    r"\braccontami\s+una\b",
    r"\bciao\b",
]

# Pattern per intenti previsionali
FORECAST_PATTERNS = [
    r"\bquando\b",
    r"\bqual\s+[eè]\s+la\s+probabilit[aà]\b",
    r"\bquanto\s+[eè]\s+probabile\b",
    r"\bprevedi\b",
    r"\briuscir[oò]\b",
    r"\bprobabilit[aà]\b",
    r"\bforecast\b",
]

# Pattern di release specifici (copre rilasciare, rilascerò, rilascio, rilasciato, rilascerà, release, deploy)
RELEASE_PATTERNS = [
    r"\brilasc[a-zà-öø-ÿ]*\b",
    r"\brelease\b",
    r"\bdeploy\b",
]

KNOWN_TARGET_KEYWORDS = [
    "framework_release",
    "framework",
    "daily_build",
    "deploy",
    "migration",
    "sistema",
    "modulo",
    "database",
    "api",
]


class SemanticQueryParser:
    """Parser semantico per query dirette al sistema di previsione Makima."""

    @staticmethod
    def normalize_text(text: str) -> str:
        """Pulisce e normalizza il testo di input."""
        cleaned = text.lower().strip()
        cleaned = cleaned.replace("'", " ").replace("’", " ")
        cleaned = re.sub(r"[^\w\s\-_]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def parse(self, text: str) -> ForecastQuery:
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
            confidence = 0.90
        elif is_forecast:
            intent = Intent.FORECAST
            confidence = 0.85
        elif is_release:
            intent = Intent.RELEASE_PREDICTION
            confidence = 0.70

        # 3. Estrazione del Target
        target = self._extract_target(normalized)

        # Se non c'è intenzione previsionale e non c'è target valido, è Unsupported
        if intent == Intent.UNSUPPORTED or (target is None and not is_forecast):
            intent = Intent.UNSUPPORTED
            confidence = 0.90

        # 4. Estrazione della Finestra Temporale
        temporal_window = extract_temporal_window(normalized)

        return ForecastQuery(
            raw_query=text,
            intent=intent,
            target=target,
            temporal_window=temporal_window,
            confidence=confidence,
        )

    def _extract_target(self, normalized: str) -> str | None:
        """Estrae l'entità target dal testo normalizzato."""
        # A. Estrazione tramite costrutti espliciti: "prevedi se/che [target]", "rilasciare [target]"
        verb_match = re.search(
            r"\b(?:prevedi|previsione|rilascer[oò]|rilasci|rilasciare|rilascio|target)\s+(?:se|che|il|la|un|una|del|della|di)?\s*([a-z0-9_\-]+)\b",
            normalized,
        )
        if verb_match:
            candidate = verb_match.group(1).strip()
            stop_words = {
                "entro", "quando", "se", "che", "prossimo", "prossima",
                "questa", "questo", "non", "il", "la", "un", "una",
                "verra", "verrà", "sia", "venga"
            }
            if candidate not in stop_words:
                if candidate == "framework":
                    return "framework_release"
                return candidate

        # B. Controllo parole chiave note
        for kw in KNOWN_TARGET_KEYWORDS:
            if re.search(rf"\b{kw}\b", normalized):
                if kw == "framework":
                    return "framework_release"
                return kw

        return None
