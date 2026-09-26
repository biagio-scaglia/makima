"""Modulo di pulizia e normalizzazione deterministica del testo per NLP."""

import re
import unicodedata


class TextCleaner:
    """Esegue la normalizzazione lessicale e ortografica del testo di input."""

    @staticmethod
    def clean(text: str) -> str:
        """Pulisce, normalizza accenti, apostrofi e spaziature multiple."""
        if not text:
            return ""

        # Normalizzazione Unicode (NFKC)
        normalized = unicodedata.normalize("NFKC", text.strip())

        # Uniformazione apostrofi e virgolette
        normalized = normalized.replace("’", "'").replace("`", "'").replace("“", '"').replace("”", '"')

        # Sostituzione apostrofo con spazio per separare articoli/preposizioni (es. "l'uscita" -> "l uscita", "qual è" -> "qual e")
        normalized = re.sub(r"(\w)'(\w)", r"\1 \2", normalized)

        # Traspone a minuscolo per confronto canonico
        normalized = normalized.lower()

        # Rimuove caratteri speciali preservando alfanumerici, trattini e underscore per identificatori (es. git:feature_ratio)
        normalized = re.sub(r"[^\w\s\-_:]", " ", normalized)

        # Collassa sequenze di spazi multipli
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized.strip()
