"""Test del cervello neurale operativo (MakimaMindNet v2 + BrainLoop)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import torch
    from makima_lab.neural import (
        ACTIONS,
        MakimaMindNet,
        MakimaNeuralEngine,
        get_neural_engine,
    )
    from makima_lab.brain import BrainAction, BrainLoop, get_brain

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


@unittest.skipIf(not HAS_TORCH, "PyTorch non installato")
class TestMakimaMindNetV2(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ckpt = os.path.join(self.temp.name, "brain.pt")
        self.engine = MakimaNeuralEngine(checkpoint_path=self.ckpt)

    def tearDown(self):
        self.temp.cleanup()

    def test_forward_action_and_workspace(self):
        batch, seq = 2, 16
        x = torch.randint(1, 100, (batch, seq))
        epi = torch.randn(batch, 3, self.engine.model.episodic_dim)
        outputs, mem = self.engine.model(x, episodic_memory=epi)
        self.assertIn("action_logits", outputs)
        self.assertIn("workspace", outputs)
        self.assertEqual(outputs["action_logits"].shape, (batch, len(ACTIONS)))
        self.assertEqual(outputs["workspace"].shape, (batch, self.engine.model.workspace_dim))
        self.assertEqual(mem.shape, (batch, self.engine.model.memory_dim))

    def test_perceive_returns_action(self):
        res = self.engine.perceive("Quando rilascerò il prossimo framework?")
        self.assertIn(res.action, ACTIONS)
        self.assertFalse(res.for_forecasting)
        self.assertTrue(res.priors_are_suggestions)
        self.assertGreater(len(res.workspace), 0)
        self.assertIn("REQUEST_RUST_FORECAST", res.action_distribution)
        report = res.format_report()
        self.assertIn("Azione cervello", report)

    def test_learn_action_label(self):
        loss = self.engine.learn_step(
            "qual è la probabilità del deploy",
            intent_label="query",
            action_label="REQUEST_RUST_FORECAST",
            polarity_label=0.6,
            auto_save=True,
        )
        self.assertIsInstance(loss, float)
        self.assertTrue(os.path.exists(self.ckpt))


@unittest.skipIf(not HAS_TORCH, "PyTorch non installato")
class TestBrainLoop(unittest.TestCase):
    def test_tick_forecast_action(self):
        brain = BrainLoop()
        result = brain.tick("Quando rilascerò il prossimo framework?", learn=False)
        self.assertEqual(result.action, BrainAction.REQUEST_RUST_FORECAST)
        self.assertIsNotNone(result.rust_forecast_hint)
        self.assertIn("makima query", result.rust_forecast_hint)
        self.assertIn("Cervello neurale", result.pulse.inner_monologue)

    def test_tick_unknown_clarify(self):
        brain = BrainLoop()
        result = brain.tick("Raccontami una barzelletta", learn=False)
        self.assertEqual(result.action, BrainAction.ASK_CLARIFY)

    def test_get_brain_singleton(self):
        a = get_brain()
        b = get_brain()
        self.assertIs(a, b)


if __name__ == "__main__":
    unittest.main()
