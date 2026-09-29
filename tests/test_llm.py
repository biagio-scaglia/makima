"""Unit test per il motore LLM grounded (Qwen / fallback)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from makima_lab.llm import QwenCognitiveEngine, get_llm_engine
from makima_lab.llm.engine import LlmResponse


class TestQwenCognitiveEngine(unittest.TestCase):
    def setUp(self):
        # Motore fresco senza caricare necessariamente i pesi
        self.engine = QwenCognitiveEngine()

    def test_singleton_instance(self):
        a = get_llm_engine()
        b = get_llm_engine()
        self.assertIs(a, b)

    def test_grounded_explain_fallback_contains_numbers(self):
        """Senza modello (o con fallback) i numeri input devono comparire nel testo."""
        text = self.engine._grounded_explain(
            target="deploy",
            probability=0.80,
            alpha=16.0,
            beta=4.0,
            evidence_count=18,
            poisson_rate=0.75,
            variance=0.0076,
        )
        self.assertIn("deploy", text)
        self.assertIn("18", text)
        self.assertIn("80.0%", text)
        self.assertIn("16.00", text)

    def test_explain_target_never_empty(self):
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
        self.assertGreater(len(explanation), 20)
        self.assertIn("deploy", explanation.lower())
        self.assertIn(self.engine.last_source, ("model", "fallback"))

    def test_digest_empty_store_is_honest(self):
        digest = self.engine.generate_digest([])
        self.assertIn("nessun target", digest.lower())

    def test_digest_with_targets_mentions_names(self):
        targets = [
            {"name": "deploy", "prob": 0.80, "obs": 18},
            {"name": "git:feature_ratio", "prob": 0.73, "obs": 37},
        ]
        digest = self.engine.generate_digest(targets, brier_score=0.1429, ece=0.0492)
        self.assertTrue("deploy" in digest.lower() or "feature" in digest.lower())

    def test_chat_without_context_refuses_invention(self):
        res = self.engine.chat("Qual è la probabilità segreta?")
        low = res.lower()
        self.assertTrue(
            "contesto" in low or "store" in low or "evidenz" in low or "senza" in low
        )

    def test_chat_with_context_uses_facts(self):
        res = self.engine.chat(
            "Come stiamo sul deploy?",
            context={"deploy_E[P]": "72.0%", "evidenze": 10},
        )
        self.assertIn("72", res)

    def test_anchor_validation(self):
        self.assertTrue(
            QwenCognitiveEngine._contains_anchors("E[P]=80.0% su deploy", ["80.0", "deploy"])
        )
        self.assertFalse(
            QwenCognitiveEngine._contains_anchors("tutto ok", ["80.0", "deploy", "16.00"])
        )

    def test_fallback_legacy_string(self):
        fallback = self.engine._fallback_generate("target: deploy explain")
        self.assertIn("Makima Cognitive Engine", fallback)

    def test_llm_response_str(self):
        r = LlmResponse(text="ciao", source="fallback", grounded=True)
        self.assertEqual(str(r), "ciao")


if __name__ == "__main__":
    unittest.main()
