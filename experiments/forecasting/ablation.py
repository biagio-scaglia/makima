"""
Formal Ablation Study for Makima Forecasting Architecture.
Measures the empirical impact of removing each individual component from Full Makima.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List
import numpy as np

from experiments.forecasting.calibration import CalibrationEvaluator
from experiments.forecasting.datasets import SyntheticDatasetGenerator
from experiments.forecasting.evaluate import compute_brier_score, compute_log_loss, compute_brier_skill_score


class AblationRunner:
    """Runs systematic ablation experiments by removing individual subsystems."""

    @staticmethod
    def run_variant(
        variant_name: str,
        samples: List[Dict[str, Any]],
        disabled_component: str | None = None,
    ) -> Dict[str, Any]:
        """Runs a modified variant of Full Makima on the sample stream."""
        target_stats: Dict[str, Dict[str, Any]] = {}
        predictions = []
        outcomes = [s["outcome"] for s in samples]

        for s in samples:
            # 1. Target resolution
            if disabled_component == "nlp":
                target = "generic_default_target"
            else:
                target = s.get("target", "git:feature_ratio")

            if target not in target_stats:
                target_stats[target] = {
                    "s": 0, "f": 0, "rate_per_day": 1.0, "timestamps": []
                }

            stats = target_stats[target]

            # 2. Bayesian vs Non-Bayesian layer
            if disabled_component == "bayesian":
                # Raw point frequency without prior regularization
                tot = stats["s"] + stats["f"]
                bayesian_p = (stats["s"] / tot) if tot > 0 else 0.50
            elif disabled_component == "online_learning":
                # Static prior Beta(1, 1) without dynamic updates
                bayesian_p = 0.50
            else:
                a = 1.0 + stats["s"]
                b = 1.0 + stats["f"]
                bayesian_p = a / (a + b)

            # 3. Neural memory & attention
            if disabled_component == "neural_memory":
                neural_adj = 0.0
            else:
                neural_score = float(s.get("neural_polarity_score", 0.0))
                neural_adj = neural_score * 0.12

            # 4. Temporal Poisson CDF
            if disabled_component == "temporal_poisson":
                poisson_temporal_p = bayesian_p
            else:
                window_days = float(s.get("window_days", 7.0))
                rate = stats.get("rate_per_day", 1.0)
                poisson_temporal_p = 1.0 - math.exp(-max(0.05, rate) * (window_days / 7.0))

            # 5. Git telemetry
            if disabled_component == "git_telemetry":
                # Fallback to constant static rate
                poisson_temporal_p = 1.0 - math.exp(-0.1 * (s.get("window_days", 7.0) / 7.0))

            # Combined estimate
            if disabled_component == "temporal_poisson":
                combined = bayesian_p + neural_adj
            else:
                combined = 0.70 * (bayesian_p + neural_adj) + 0.30 * poisson_temporal_p

            pred = float(np.clip(combined, 0.01, 0.99))
            predictions.append(pred)

            # Online observation update
            if disabled_component != "online_learning":
                out = s.get("outcome", s.get("value", 0))
                ts = s.get("timestamp_sec", 0)

                if out >= 0.5:
                    stats["s"] += 1
                else:
                    stats["f"] += 1

                if ts > 0:
                    stats["timestamps"].append(ts)
                    ts_list = stats["timestamps"]
                    if len(ts_list) >= 2:
                        span = max(1.0, (max(ts_list) - min(ts_list)) / 86400.0)
                        stats["rate_per_day"] = len(ts_list) / span

        brier = compute_brier_score(predictions, outcomes)
        log_loss = compute_log_loss(predictions, outcomes)
        calib = CalibrationEvaluator.compute_calibration(predictions, outcomes, num_bins=10)

        # Baseline 50% for BSS
        const_50 = [0.5] * len(outcomes)
        ref_brier = compute_brier_score(const_50, outcomes)
        bss = compute_brier_skill_score(brier, ref_brier)

        return {
            "variant": variant_name,
            "disabled_component": disabled_component or "none (Full Makima)",
            "brier_score": round(brier, 4),
            "log_loss": round(log_loss, 4),
            "bss": round(bss * 100.0, 2),
            "ece": round(calib.expected_calibration_error, 4),
            "mce": round(calib.max_calibration_error, 4),
            "calibration_rating": calib.rating,
        }

    @staticmethod
    def run_ablation_study(seed: int = 42) -> Dict[str, Any]:
        """Runs the complete ablation matrix across synthetic and regime-shift datasets."""
        suite = SyntheticDatasetGenerator.generate_full_benchmark_suite(seed=seed)
        all_samples = []
        for samples in suite.values():
            all_samples.extend(samples)

        variants = [
            ("Full Makima", None),
            ("w/o Neural Memory", "neural_memory"),
            ("w/o NLP Resolution", "nlp"),
            ("w/o Git Telemetry", "git_telemetry"),
            ("w/o Bayesian Conjugate Layer", "bayesian"),
            ("w/o Temporal Poisson Features", "temporal_poisson"),
            ("w/o Online Learning (Static)", "online_learning"),
        ]

        ablation_results = {
            "title": "Makima Subsystem Ablation Study",
            "total_samples": len(all_samples),
            "variants": {},
        }

        full_brier = None
        for name, disabled in variants:
            res = AblationRunner.run_variant(name, all_samples, disabled_component=disabled)
            if disabled is None:
                full_brier = res["brier_score"]
                res["delta_brier_vs_full"] = 0.0
                res["degradation_pct"] = 0.0
            else:
                delta = res["brier_score"] - (full_brier or 0.0)
                deg_pct = (delta / (full_brier or 1.0)) * 100.0 if full_brier else 0.0
                res["delta_brier_vs_full"] = round(delta, 4)
                res["degradation_pct"] = round(deg_pct, 2)

            ablation_results["variants"][name] = res

        return ablation_results
