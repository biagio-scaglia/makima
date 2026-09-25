"""Test suite per il laboratorio scientifico makima_lab."""

import sys
import unittest
from pathlib import Path

# Permette l'import del modulo makima_lab durante l'esecuzione dei test
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from makima_lab import __version__, lab_status
from makima_lab.distributions import Bernoulli, BetaDistribution, PoissonDistribution
from makima_lab.evaluation import Evaluator, brier_score, log_loss


class TestMakimaLab(unittest.TestCase):
    """Verifiche sui metadati, sulle distribuzioni e sulle metriche di calibrazione in Python."""

    def test_makima_lab_metadata(self):
        self.assertEqual(__version__, "0.1.0")

    def test_makima_lab_status(self):
        status = lab_status()
        self.assertEqual(status["lab"], "makima_lab")
        self.assertEqual(status["version"], "0.1.0")
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["role"], "Scientific Research & NLP Prototyping")

    def test_bernoulli_moments_and_entropy(self):
        b = Bernoulli(0.5)
        self.assertAlmostEqual(b.mean, 0.5)
        self.assertAlmostEqual(b.variance, 0.25)
        self.assertAlmostEqual(b.entropy_bits, 1.0)

    def test_beta_bayesian_update(self):
        prior = BetaDistribution(1.0, 1.0)
        post = prior.bayesian_update(successes=7, failures=3)
        self.assertEqual(post.alpha, 8.0)
        self.assertEqual(post.beta, 4.0)
        self.assertAlmostEqual(post.mean, 8.0 / 12.0)
        self.assertAlmostEqual(post.mode, 0.7)

    def test_poisson_pmf(self):
        p = PoissonDistribution(2.0)
        self.assertAlmostEqual(p.mean, 2.0)
        self.assertAlmostEqual(p.pmf(0), 0.135335, places=4)

    def test_brier_and_log_loss_scoring(self):
        # Previsione perfetta 1.0 con esito 1
        self.assertAlmostEqual(brier_score(1.0, True), 0.0)
        # Previsione 0.5 con esito 1 -> (0.5 - 1)^2 = 0.25
        self.assertAlmostEqual(brier_score(0.5, True), 0.25)
        # Log loss a 0.5 -> -ln(0.5) ~= 0.6931
        self.assertAlmostEqual(log_loss(0.5, True), 0.693147, places=4)

    def test_evaluator_skill_score(self):
        evaluator = Evaluator()
        evaluator.add(0.8, True)
        evaluator.add(0.9, True)
        evaluator.add(0.1, False)

        report = evaluator.evaluate(baseline_prob=0.5)
        self.assertEqual(report["total_evaluated"], 3)
        self.assertLess(report["mean_brier_score"], 0.25)
        self.assertGreater(report["brier_skill_score"], 0.0)
        self.assertIn("expected_calibration_error", report)
        self.assertIn("calibration_bins", report)
        self.assertEqual(len(report["calibration_bins"]), 5)


if __name__ == "__main__":
    unittest.main()
