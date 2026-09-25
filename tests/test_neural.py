"""
Unit tests for Makima Neural Engine and PyTorch architectures.
"""

import os
import unittest
import tempfile
import torch

from makima_lab.neural import (
    MakimaTokenizer,
    MakimaMindNet,
    MakimaNeuralEngine,
    NeuralInferenceResult,
)


class TestMakimaNeuralMind(unittest.TestCase):

    def setUp(self):
        self.tokenizer = MakimaTokenizer()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = os.path.join(self.temp_dir.name, "test_brain.pt")
        self.engine = MakimaNeuralEngine(checkpoint_path=self.checkpoint_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tokenizer_encoding_and_decoding(self):
        text = "Oggi pioverà a Milano oppure no?"
        tokens = self.tokenizer.encode(text, max_length=16)
        self.assertEqual(len(tokens), 16)
        decoded = self.tokenizer.decode(tokens)
        self.assertIn("oggi", decoded)

    def test_model_forward_pass(self):
        batch_size = 2
        seq_len = 16
        dummy_input = torch.randint(0, 100, (batch_size, seq_len))
        outputs, new_memory = self.engine.model(dummy_input)

        self.assertIn("intent_logits", outputs)
        self.assertIn("target_embedding", outputs)
        self.assertIn("polarity", outputs)
        self.assertIn("alpha_prior", outputs)
        self.assertIn("beta_prior", outputs)
        self.assertIn("lambda_rate", outputs)

        self.assertEqual(outputs["intent_logits"].shape, (batch_size, 5))
        self.assertEqual(outputs["target_embedding"].shape, (batch_size, 32))
        self.assertEqual(new_memory.shape, (batch_size, 64))

    def test_engine_perception(self):
        res = self.engine.perceive("Oggi ho completato la sessione di palestra e sto benissimo")
        self.assertIsInstance(res, NeuralInferenceResult)
        self.assertIn(res.intent, ["query", "journal", "routine", "outcome", "fact"])
        self.assertGreaterEqual(res.intent_confidence, 0.0)
        self.assertLessEqual(res.intent_confidence, 1.0)
        self.assertGreater(res.prior_alpha, 0.0)
        self.assertGreater(res.prior_beta, 0.0)
        self.assertGreater(res.lambda_rate, 0.0)
        self.assertGreaterEqual(res.memory_norm, 0.0)

        # Test report formatting
        report = res.format_report()
        self.assertIn("Makima Neural Perception", report)

    def test_online_learning_step_and_checkpoint(self):
        initial_steps = self.engine.total_learning_steps
        loss1 = self.engine.learn_step(
            text="domani pioverà sicuramente",
            intent_label="query",
            polarity_label=1.0,
            auto_save=True,
        )
        self.assertIsInstance(loss1, float)
        self.assertEqual(self.engine.total_learning_steps, initial_steps + 1)
        self.assertTrue(os.path.exists(self.checkpoint_path))

        # Test reloading checkpoint
        new_engine = MakimaNeuralEngine(checkpoint_path=self.checkpoint_path)
        self.assertEqual(new_engine.total_learning_steps, initial_steps + 1)

    def test_user_memory_evolution(self):
        self.engine.reset_memory()
        norm_initial = torch.norm(self.engine.user_memory).item()
        self.assertAlmostEqual(norm_initial, 0.0, places=5)

        self.engine.perceive("Prima interazione sul mio lavoro", update_memory=True)
        norm_after1 = torch.norm(self.engine.user_memory).item()
        self.assertGreater(norm_after1, 0.0)

        self.engine.perceive("Seconda interazione sui miei obiettivi", update_memory=True)
        norm_after2 = torch.norm(self.engine.user_memory).item()
        self.assertGreater(norm_after2, 0.0)


if __name__ == "__main__":
    unittest.main()
