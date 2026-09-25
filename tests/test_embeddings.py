"""
Unit tests for SemanticEmbedder and semantic similarity in Makima.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from makima_lab.embeddings import SemanticEmbedder, get_embedder


class TestSemanticEmbedder(unittest.TestCase):

    def setUp(self):
        self.embedder = SemanticEmbedder()

    def test_encode_single_string(self):
        text = "Makima è un sistema di previsione probabilistica"
        vec = self.embedder.encode(text)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(len(vec.shape), 1)
        self.assertEqual(vec.shape[0], self.embedder.dim)
        # Check normalized vector
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_encode_batch(self):
        texts = [
            "Oggi pioverà a Milano",
            "Domani farà bel tempo e splenderà il sole",
            "Completato il commit di refactoring del codice",
        ]
        matrix = self.embedder.encode(texts)
        self.assertIsInstance(matrix, np.ndarray)
        self.assertEqual(matrix.shape, (3, self.embedder.dim))

    def test_semantic_similarity(self):
        # Similar sentences should have higher similarity than unrelated sentences
        s1 = "Oggi piove intensamente a Milano"
        s2 = "Precipitazioni e pioggia previste su Milano"
        s3 = "Ho eseguito il deployment del microservizio in cloud"

        sim_related = self.embedder.similarity(s1, s2)
        sim_unrelated = self.embedder.similarity(s1, s3)

        self.assertIsInstance(sim_related, float)
        self.assertIsInstance(sim_unrelated, float)
        self.assertGreaterEqual(sim_related, -1.0)
        self.assertLessEqual(sim_related, 1.0)


if __name__ == "__main__":
    unittest.main()
