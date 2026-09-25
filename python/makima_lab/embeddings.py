"""
Semantic Embedding Provider for Makima.
Integrates Sentence-Transformers / MiniLM with graceful offline hashing fallback.
Provides fast semantic vector extraction and cosine similarity.
"""

from __future__ import annotations
import math
from typing import List, Optional
import numpy as np

_EMBEDDER_INSTANCE = None


class SemanticEmbedder:
    """Provides dense semantic embeddings for sentences and queries."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = None
        self.dim = 384
        self.has_transformer = False

        try:
            import os
            import warnings
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
            warnings.filterwarnings("ignore", module=".*huggingface_hub.*")

            from transformers.utils import logging as tf_utils_logging
            if hasattr(tf_utils_logging, "disable_progress_bar"):
                tf_utils_logging.disable_progress_bar()

            try:
                import huggingface_hub.utils
                if hasattr(huggingface_hub.utils, "disable_progress_bars"):
                    huggingface_hub.utils.disable_progress_bars()
            except Exception:
                pass

            from sentence_transformers import SentenceTransformer
            try:
                self.model = SentenceTransformer(model_name, local_files_only=True)
            except Exception:
                self.model = SentenceTransformer(model_name, local_files_only=False)

            self.has_transformer = True
            if hasattr(self.model, "get_embedding_dimension"):
                self.dim = self.model.get_embedding_dimension() or 384
            elif hasattr(self.model, "get_sentence_embedding_dimension"):
                self.dim = self.model.get_sentence_embedding_dimension() or 384
            else:
                self.dim = 384
        except Exception:
            # Graceful offline fallback
            self.has_transformer = False
            self.dim = 384

    def encode(self, text_or_texts: str | List[str]) -> np.ndarray:
        """Encodes string or list of strings into normalized dense embedding vectors."""
        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
            single = True
        else:
            texts = text_or_texts
            single = False

        if self.has_transformer and self.model is not None:
            try:
                embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                return embeddings[0] if single else embeddings
            except Exception:
                pass

        # Offline fallback vector generation using subword hash projections
        vectors = []
        for t in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            words = t.lower().split()
            for i, w in enumerate(words):
                h = abs(hash(w))
                idx = h % self.dim
                val = math.sin(h + i)
                vec[idx] += val
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            vectors.append(vec)

        arr = np.stack(vectors, axis=0)
        return arr[0] if single else arr

    def similarity(self, text1: str, text2: str) -> float:
        """Computes semantic cosine similarity in [-1, 1] between two texts."""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
        return sim


def get_embedder() -> SemanticEmbedder:
    """Singleton getter for Makima's SemanticEmbedder."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        _EMBEDDER_INSTANCE = SemanticEmbedder()
    return _EMBEDDER_INSTANCE
