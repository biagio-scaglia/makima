"""Modulo di rappresentazione semantica vettoriale densa."""

from __future__ import annotations
from typing import List
import numpy as np
from makima_lab.embeddings import get_embedder


class SemanticRepresentation:
    """Fornisce vettori di embedding normalizzati e calcolo di similarità per la pipeline NLP.
    
    Specifiche:
    - Modello: all-MiniLM-L6-v2 (Sentence-Transformers) con fallback su hash subword
    - Dimensione: 384 float32
    - Normalizzazione: Euclidea (L2 norm = 1.0)
    - Metrica: Cosine Similarity ∈ [-1.0, 1.0]
    """

    def __init__(self) -> None:
        self._embedder = get_embedder()

    @property
    def dimension(self) -> int:
        """Restituisce la dimensionalità dello spazio vettoriale (384)."""
        return self._embedder.dim

    def encode(self, text_or_texts: str | List[str]) -> np.ndarray:
        """Codifica una o più stringhe in vettori densi normalizzati."""
        return self._embedder.encode(text_or_texts)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Calcola la similarità semantica coseno tra due espressioni testuali."""
        return self._embedder.similarity(text_a, text_b)

    def rank_candidates(self, query: str, candidates: List[str]) -> List[tuple[str, float]]:
        """Ordina una lista di candidati in base alla similarità semantica decrescente."""
        if not candidates:
            return []
        scores = []
        for cand in candidates:
            sim = self.similarity(query, cand)
            scores.append((cand, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
