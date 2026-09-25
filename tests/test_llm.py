"""Unit test per il modulo LLM / QwenCognitiveEngine di Makima."""

import sys
import unittest
from pathlib import Path

# Inclusione path del laboratorio
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from makima_lab.llm import QwenCognitiveEngine, get_llm_engine


class TestQwenCognitiveEngine(unittest.TestCase):
    """Test suite per il motore SLM Qwen 2.5."""

    def setUp(self):
        self.engine = get_llm_engine()

    def test_singleton_instance(self):
        """Verifica che get_llm_engine restituisca lo stesso singleton."""
        engine2 = get_llm_engine()
        self.assertIs(self.engine, engine2)

    def test_explain_target_contract(self):
        """Verifica che explain_target restituisca un testo esplicativo non vuoto."""
        explanation = self.engine.explain_target(
            target="deploy",
            probability=0.80,
            alpha=16.0,
            beta=4.0,
            evidence_count=18,
            poisson_rate=0.75,
            variance=0.0076,
        )
        self.assertIsInstance(explanation, str)
        self.assertTrue(len(explanation) > 10)

    def test_digest_contract(self):
        """Verifica che generate_digest produca una sintesi coerente."""
        targets = [
            {"name": "deploy", "prob": 0.80, "obs": 18},
            {"name": "git:feature_ratio", "prob": 0.73, "obs": 37},
        ]
        digest = self.engine.generate_digest(targets, brier_score=0.1429, ece=0.0492)
        self.assertIsInstance(digest, str)
        self.assertTrue(len(digest) > 10)

    def test_chat_contract(self):
        """Verifica che chat risponda a una richiesta."""
        res = self.engine.chat("Qual è la calibrazione attuale di Makima?")
        self.assertIsInstance(res, str)
        self.assertTrue(len(res) > 5)

    def test_fallback_generator(self):
        """Verifica che il fallback deterministico produca una risposta formale."""
        fallback = self.engine._fallback_generate("test prompt")
        self.assertIn("Makima Cognitive Engine", fallback)


if __name__ == "__main__":
    unittest.main()
