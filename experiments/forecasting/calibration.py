"""
Calibration and Reliability Assessment for Makima Forecasting Models.
Computes ECE, MCE, Calibration Bins, and ASCII Reliability Diagrams.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np


@dataclass(frozen=True)
class CalibrationBinResult:
    """Statistics for a single confidence interval bin."""
    lower_bound: float
    upper_bound: float
    count: int
    mean_predicted: float
    observed_frequency: float
    calibration_error: float


@dataclass(frozen=True)
class CalibrationMetrics:
    """Comprehensive calibration metrics and reliability table."""
    expected_calibration_error: float  # ECE
    max_calibration_error: float       # MCE
    bins: List[CalibrationBinResult]
    rating: str                        # EXCELLENT, GOOD, MODERATE, POOR

    @staticmethod
    def get_rating_from_ece(ece: float) -> str:
        if ece < 0.030:
            return "EXCELLENT"
        elif ece < 0.060:
            return "GOOD"
        elif ece < 0.120:
            return "MODERATE"
        else:
            return "POOR"


class CalibrationEvaluator:
    """Evaluates probability calibration quality against ground truth outcomes."""

    @staticmethod
    def compute_calibration(
        predictions: List[float],
        outcomes: List[int | bool | float],
        num_bins: int = 10,
    ) -> CalibrationMetrics:
        """Computes binned calibration, ECE, MCE, and rating."""
        if len(predictions) != len(outcomes) or len(predictions) == 0:
            return CalibrationMetrics(
                expected_calibration_error=0.0,
                max_calibration_error=0.0,
                bins=[],
                rating="POOR",
            )

        preds = np.array(predictions, dtype=np.float64)
        targets = np.array(outcomes, dtype=np.float64)
        n_total = len(preds)

        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        bins_result = []
        weighted_error_sum = 0.0
        max_error = 0.0

        for i in range(num_bins):
            lower = float(bin_boundaries[i])
            upper = float(bin_boundaries[i + 1])

            if i == num_bins - 1:
                mask = (preds >= lower) & (preds <= upper)
            else:
                mask = (preds >= lower) & (preds < upper)

            count = int(np.sum(mask))
            if count > 0:
                mean_p = float(np.mean(preds[mask]))
                obs_freq = float(np.mean(targets[mask]))
                err = abs(mean_p - obs_freq)

                weighted_error_sum += (count / n_total) * err
                if err > max_error:
                    max_error = err
            else:
                mean_p = 0.0
                obs_freq = 0.0
                err = 0.0

            bins_result.append(CalibrationBinResult(
                lower_bound=lower,
                upper_bound=upper,
                count=count,
                mean_predicted=mean_p,
                observed_frequency=obs_freq,
                calibration_error=err,
            ))

        rating = CalibrationMetrics.get_rating_from_ece(weighted_error_sum)

        return CalibrationMetrics(
            expected_calibration_error=weighted_error_sum,
            max_calibration_error=max_error,
            bins=bins_result,
            rating=rating,
        )

    @staticmethod
    def render_ascii_reliability_diagram(
        metrics: CalibrationMetrics,
        model_name: str = "Makima",
    ) -> str:
        """Renders an interpretable ASCII Reliability Diagram and 2D calibration curve."""
        lines = [
            f"┌────────────────────────────────────────────────────────────────────────┐",
            f"│  RELIABILITY DIAGRAM & CALIBRATION CURVE: {model_name:<28} │",
            f"├────────────────────────────────────────────────────────────────────────┤",
            f"│ ECE (Expected Calibration Error): {metrics.expected_calibration_error * 100:>5.2f}%   Status: {metrics.rating:<10}    │",
            f"│ MCE (Max Calibration Error):      {metrics.max_calibration_error * 100:>5.2f}%                             │",
            f"├────────────────────────────────────────────────────────────────────────┤",
            f"│ Bin Range  │ Count │ E[P] Pred │ Observed │   Error  │ Alignment (P vs R)  │",
            f"├────────────┼───────┼───────────┼──────────┼──────────┼─────────────────────┤",
        ]

        for b in metrics.bins:
            if b.count == 0:
                lines.append(
                    f"│ {b.lower_bound*100:>2.0f}%-{b.upper_bound*100:>3.0f}%  │     0 │       --  │       -- │       -- │ [-----------------] │"
                )
            else:
                # 17-width ASCII slider
                p_idx = min(16, max(0, int(round(b.mean_predicted * 16.0))))
                r_idx = min(16, max(0, int(round(b.observed_frequency * 16.0))))

                bar = list("-" * 17)
                if p_idx == r_idx:
                    bar[p_idx] = "="
                else:
                    bar[p_idx] = "P"
                    bar[r_idx] = "R"

                bar_str = "".join(bar)
                lines.append(
                    f"│ {b.lower_bound*100:>2.0f}%-{b.upper_bound*100:>3.0f}%  │ {b.count:>5} │    {b.mean_predicted*100:>5.1f}% │   {b.observed_frequency*100:>5.1f}% │   {b.calibration_error*100:>5.1f}% │ [{bar_str}] │"
                )

        lines.append(f"└────────────────────────────────────────────────────────────────────────┘")
        return "\n".join(lines)
