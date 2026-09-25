"""
Benchmark Evaluation Runner for Makima Forecasting Models.
Evaluates Proper Scoring Rules (Brier, Log Loss, BSS) and Calibration (ECE, MCE).
"""

from __future__ import annotations
import json
import math
from typing import Any, Dict, List, Tuple
import numpy as np

from experiments.forecasting.baselines import (
    BaseForecastingModel,
    BayesianConjugateModel,
    BayesianNeuralModel,
    BayesianNLPModel,
    ClimatologicalBaseline,
    ConstantFiftyBaseline,
    FullMakimaModel,
    StaticPriorBaseline,
)
from experiments.forecasting.calibration import CalibrationEvaluator, CalibrationMetrics
from experiments.forecasting.datasets import SyntheticDatasetGenerator


def compute_brier_score(predictions: List[float], outcomes: List[int | bool | float]) -> float:
    """Computes mean Brier Score: 1/N sum (p_i - y_i)^2."""
    p = np.array(predictions, dtype=np.float64)
    y = np.array(outcomes, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def compute_log_loss(predictions: List[float], outcomes: List[int | bool | float], eps: float = 1e-15) -> float:
    """Computes mean Logarithmic Loss (Negative Log-Likelihood / Cross Entropy)."""
    p = np.clip(np.array(predictions, dtype=np.float64), eps, 1.0 - eps)
    y = np.array(outcomes, dtype=np.float64)
    loss = - (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.mean(loss))


def compute_brier_skill_score(model_brier: float, baseline_brier: float) -> float:
    """Computes Brier Skill Score: 1 - BS_model / BS_baseline."""
    if baseline_brier < 1e-12:
        return 0.0
    return 1.0 - (model_brier / baseline_brier)


class BenchmarkRunner:
    """Runs rigorous multi-dataset, multi-model probabilistic forecasting benchmarks."""

    def __init__(self) -> None:
        self.models: List[BaseForecastingModel] = [
            ConstantFiftyBaseline(),
            ClimatologicalBaseline(),
            StaticPriorBaseline(),
            BayesianConjugateModel(),
            BayesianNLPModel(),
            BayesianNeuralModel(),
            FullMakimaModel(),
        ]

    def run_benchmark_on_dataset(
        self,
        dataset_name: str,
        samples: List[Dict[str, Any]],
        num_bins: int = 10,
    ) -> Dict[str, Any]:
        """Runs all models sequentially on a dataset stream."""
        outcomes = [s["outcome"] for s in samples]
        results: Dict[str, Any] = {
            "dataset": dataset_name,
            "sample_count": len(samples),
            "models": {},
        }

        # Compute reference 50% baseline Brier for BSS
        const_fifty = [0.5] * len(outcomes)
        ref_50_brier = compute_brier_score(const_fifty, outcomes)

        # Compute climatological base rate Brier for BSS
        base_rate = float(np.mean(outcomes))
        clim_preds = [base_rate] * len(outcomes)
        ref_clim_brier = compute_brier_score(clim_preds, outcomes)

        for model in self.models:
            model.reset()
            preds = model.fit_and_predict_stream(samples)

            brier = compute_brier_score(preds, outcomes)
            log_loss = compute_log_loss(preds, outcomes)
            bss_50 = compute_brier_skill_score(brier, ref_50_brier)
            bss_clim = compute_brier_skill_score(brier, ref_clim_brier)

            calib_metrics = CalibrationEvaluator.compute_calibration(preds, outcomes, num_bins=num_bins)

            results["models"][model.name] = {
                "brier_score": round(brier, 4),
                "log_loss": round(log_loss, 4),
                "bss_vs_50": round(bss_50 * 100.0, 2),
                "bss_vs_clim": round(bss_clim * 100.0, 2),
                "ece": round(calib_metrics.expected_calibration_error, 4),
                "mce": round(calib_metrics.max_calibration_error, 4),
                "calibration_rating": calib_metrics.rating,
                "improvement_vs_50_pct": round(((ref_50_brier - brier) / ref_50_brier) * 100.0, 2),
            }

        return results

    def run_full_suite(self, seed: int = 42) -> Dict[str, Any]:
        """Runs the entire multi-dataset benchmark suite."""
        suite = SyntheticDatasetGenerator.generate_full_benchmark_suite(seed=seed)
        full_results = {
            "title": "Makima Probabilistic Forecasting Scientific Benchmark",
            "seed": seed,
            "datasets": {},
            "aggregate_summary": {},
        }

        model_aggregates: Dict[str, Dict[str, List[float]]] = {
            m.name: {"brier": [], "log_loss": [], "ece": [], "bss_50": [], "bss_clim": []}
            for m in self.models
        }

        for ds_name, samples in suite.items():
            res = self.run_benchmark_on_dataset(ds_name, samples)
            full_results["datasets"][ds_name] = res

            for m_name, m_res in res["models"].items():
                model_aggregates[m_name]["brier"].append(m_res["brier_score"])
                model_aggregates[m_name]["log_loss"].append(m_res["log_loss"])
                model_aggregates[m_name]["ece"].append(m_res["ece"])
                model_aggregates[m_name]["bss_50"].append(m_res["bss_vs_50"])
                model_aggregates[m_name]["bss_clim"].append(m_res["bss_vs_clim"])

        for m_name, metrics in model_aggregates.items():
            mean_brier = float(np.mean(metrics["brier"]))
            mean_log_loss = float(np.mean(metrics["log_loss"]))
            mean_ece = float(np.mean(metrics["ece"]))
            mean_bss_50 = float(np.mean(metrics["bss_50"]))
            mean_bss_clim = float(np.mean(metrics["bss_clim"]))

            full_results["aggregate_summary"][m_name] = {
                "mean_brier_score": round(mean_brier, 4),
                "mean_log_loss": round(mean_log_loss, 4),
                "mean_ece": round(mean_ece, 4),
                "mean_bss_vs_50": round(mean_bss_50, 2),
                "mean_bss_vs_clim": round(mean_bss_clim, 2),
                "calibration_rating": CalibrationMetrics.get_rating_from_ece(mean_ece),
            }

        return full_results
