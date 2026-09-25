"""Test suite per la validazione scientifica, baseline, calibrazione ed esperimenti di forecasting."""

import sys
import unittest
from pathlib import Path

# Inclusione path di python e della radice
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.forecasting.baselines import (
    BayesianConjugateModel,
    BayesianNeuralModel,
    ClimatologicalBaseline,
    ConstantFiftyBaseline,
    FullMakimaModel,
    StaticPriorBaseline,
)
from experiments.forecasting.calibration import CalibrationEvaluator, CalibrationMetrics
from experiments.forecasting.datasets import SyntheticDatasetGenerator
from experiments.forecasting.evaluate import (
    BenchmarkRunner,
    compute_brier_score,
    compute_brier_skill_score,
    compute_log_loss,
)
from experiments.forecasting.ablation import AblationRunner


class TestForecastingValidation(unittest.TestCase):
    """Test suite per la verifica di baseline, calibrazione ed ablation study."""

    def test_constant_fifty_baseline(self):
        model = ConstantFiftyBaseline()
        preds = model.fit_and_predict_stream([{"outcome": 1}, {"outcome": 0}])
        self.assertEqual(preds, [0.50, 0.50])
        brier = compute_brier_score(preds, [1, 0])
        self.assertAlmostEqual(brier, 0.25, places=4)

    def test_climatological_baseline(self):
        model = ClimatologicalBaseline()
        samples = [{"outcome": 1}, {"outcome": 1}, {"outcome": 0}]
        preds = model.fit_and_predict_stream(samples)
        # 1st: default 0.50, 2nd: 1/1=1.0 clipped, 3rd: 2/2=1.0 clipped
        self.assertEqual(len(preds), 3)
        self.assertAlmostEqual(preds[0], 0.50)

    def test_bayesian_conjugate_model_updating(self):
        model = BayesianConjugateModel(prior_alpha=1.0, prior_beta=1.0)
        samples = [{"outcome": 1}, {"outcome": 1}, {"outcome": 0}]
        preds = model.fit_and_predict_stream(samples)
        # 1st pred: 1/(1+1)=0.5, then obs=1
        # 2nd pred: 2/(2+1)=0.6667, then obs=1
        # 3rd pred: 3/(3+1)=0.75, then obs=0
        self.assertAlmostEqual(preds[0], 0.50, places=4)
        self.assertAlmostEqual(preds[1], 2.0 / 3.0, places=4)
        self.assertAlmostEqual(preds[2], 3.0 / 4.0, places=4)

    def test_proper_scoring_rules_properties(self):
        # Perfect prediction
        self.assertAlmostEqual(compute_brier_score([1.0, 0.0], [1, 0]), 0.0)
        # Random guess
        self.assertAlmostEqual(compute_brier_score([0.5, 0.5], [1, 0]), 0.25)
        # BSS
        self.assertAlmostEqual(compute_brier_skill_score(0.20, 0.25), 0.20)
        self.assertAlmostEqual(compute_brier_skill_score(0.25, 0.25), 0.00)

    def test_calibration_ece_and_diagram(self):
        preds = [0.1, 0.2, 0.8, 0.9]
        outcomes = [0, 0, 1, 1]
        metrics = CalibrationEvaluator.compute_calibration(preds, outcomes, num_bins=5)

        self.assertIsInstance(metrics, CalibrationMetrics)
        self.assertLess(metrics.expected_calibration_error, 0.20)
        diagram = CalibrationEvaluator.render_ascii_reliability_diagram(metrics, "TestModel")
        self.assertIn("RELIABILITY DIAGRAM", diagram)
        self.assertIn("ECE", diagram)

    def test_synthetic_dataset_generators(self):
        stat = SyntheticDatasetGenerator.generate_stationary_stream(num_samples=100, true_p=0.80)
        self.assertEqual(len(stat), 100)
        outcomes = [s["outcome"] for s in stat]
        mean_p = sum(outcomes) / len(outcomes)
        self.assertTrue(0.65 <= mean_p <= 0.95)

        suite = SyntheticDatasetGenerator.generate_full_benchmark_suite(seed=123)
        self.assertIn("stationary_stream_p72", suite)
        self.assertIn("regime_shifts", suite)
        self.assertIn("poisson_arrivals", suite)

    def test_benchmark_runner_execution(self):
        runner = BenchmarkRunner()
        samples = SyntheticDatasetGenerator.generate_stationary_stream(num_samples=200, true_p=0.75, seed=42)
        res = runner.run_benchmark_on_dataset("test_sample", samples)

        self.assertEqual(res["sample_count"], 200)
        self.assertIn("constant_50", res["models"])
        self.assertIn("full_makima", res["models"])
        # Full Makima should beat 50% baseline
        self.assertLess(res["models"]["full_makima"]["brier_score"], res["models"]["constant_50"]["brier_score"])

    def test_ablation_study_execution(self):
        samples = SyntheticDatasetGenerator.generate_stationary_stream(num_samples=200, true_p=0.80, seed=42)
        res = AblationRunner.run_variant("Full", samples, disabled_component=None)
        self.assertIn("brier_score", res)
        self.assertIn("ece", res)
        self.assertLess(res["brier_score"], 0.25)


if __name__ == "__main__":
    unittest.main()
