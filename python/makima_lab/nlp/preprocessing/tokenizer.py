"""Tokenizzatore linguistico modulare per analisi lessicale e vettoriale."""

from typing import List


class SimpleTokenizer:
    """Tokenizzatore di parole e n-grammi per la pipeline semantica."""

    STOPWORDS_IT = {
        "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
        "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
        "e", "ed", "o", "ma", "se", "perché", "come", "cosa",
        "mi", "ti", "si", "ci", "vi", "ne", "del", "dello", "della",
        "dei", "degli", "delle", "al", "allo", "alla", "ai", "agli", "alle"
    }

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Divide il testo in token di parole puliti."""
        if not text:
            return []
        return [tok for tok in text.split() if tok]

    @classmethod
    def filter_stopwords(cls, tokens: List[str]) -> List[str]:
        """Filtra le stopword comuni preservando token semanticamente densi."""
        return [tok for tok in tokens if tok not in cls.STOPWORDS_IT]

    @classmethod
    def extract_ngrams(cls, tokens: List[str], n: int = 2) -> List[str]:
        """Estrae n-grammi continui per il matching di entità composte (es. 'daily build', 'unit test')."""
        if len(tokens) < n:
            return []
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
