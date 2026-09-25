"""Modulo di valutazione statistica, Proper Scoring Rules e calibrazione predittiva."""

import math
from typing import Any


def brier_score(predicted_prob: float, actual_outcome: bool | int) -> float:
    """Calcola il Brier Score: (p - y)^2."""
    y = 1.0 if actual_outcome else 0.0
    return (predicted_prob - y) ** 2


def log_loss(predicted_prob: float, actual_outcome: bool | int, eps: float = 1e-15) -> float:
    """Calcola la Log Loss (Negative Log-Likelihood) con clipping protettivo."""
    p = max(eps, min(1.0 - eps, float(predicted_prob)))
    y = 1.0 if actual_outcome else 0.0
    return - (y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def brier_skill_score(model_brier: float, baseline_brier: float) -> float:
    """Calcola il Brier Skill Score: 1 - (BS_mod / BS_ref)."""
    if abs(baseline_brier) < 1e-12:
        return 0.0
    return 1.0 - (model_brier / baseline_brier)


def calibration_curve(
    records: list[tuple[float, bool]], num_bins: int = 5
) -> list[dict[str, Any]]:
    """Calcola i bin di calibrazione e le frequenze empiriche reali."""
    bins_count = max(1, num_bins)
    bin_width = 1.0 / bins_count
    bins: list[dict[str, Any]] = []

    for i in range(bins_count):
        lower = i * bin_width
        upper = (i + 1) * bin_width

        in_bin = [
            (p, y)
            for (p, y) in records
            if (lower <= p <= upper if i == bins_count - 1 else lower <= p < upper)
        ]

        if not in_bin:
            bins.append({
                "lower_bound": lower,
                "upper_bound": upper,
                "count": 0,
                "mean_predicted": 0.0,
                "observed_frequency": 0.0,
                "calibration_error": 0.0,
            })
        else:
            n = len(in_bin)
            mean_p = sum(p for p, _ in in_bin) / n
            obs_y = sum(1.0 if y else 0.0 for _, y in in_bin) / n
            error = abs(mean_p - obs_y)

            bins.append({
                "lower_bound": lower,
                "upper_bound": upper,
                "count": n,
                "mean_predicted": mean_p,
                "observed_frequency": obs_y,
                "calibration_error": error,
            })

    return bins


def expected_calibration_error(records: list[tuple[float, bool]], num_bins: int = 5) -> float:
    """Calcola l'Expected Calibration Error pesato (ECE)."""
    if not records:
        return 0.0
    bins = calibration_curve(records, num_bins)
    total = len(records)
    return sum((b["count"] / total) * b["calibration_error"] for b in bins if b["count"] > 0)


class Evaluator:
    """Valutatore empirico della qualità e calibrazione delle previsioni."""

    def __init__(self):
        self.records: list[tuple[float, bool]] = []

    def add(self, predicted_prob: float, actual_outcome: bool | int) -> None:
        """Aggiunge una coppia (previsione, esito reale)."""
        self.records.append((float(predicted_prob), bool(actual_outcome)))

    def evaluate(self, baseline_prob: float = 0.5, num_bins: int = 5) -> dict[str, Any]:
        """Calcola metriche aggregate di scoring e calibrazione."""
        if not self.records:
            return {"total_evaluated": 0}

        n = len(self.records)
        brier_sum = sum(brier_score(p, y) for p, y in self.records)
        log_loss_sum = sum(log_loss(p, y) for p, y in self.records)
        baseline_brier_sum = sum(brier_score(baseline_prob, y) for _, y in self.records)

        mean_brier = brier_sum / n
        mean_log_loss = log_loss_sum / n
        mean_baseline_brier = baseline_brier_sum / n
        bss = brier_skill_score(mean_brier, mean_baseline_brier)

        bins = calibration_curve(self.records, num_bins)
        ece = expected_calibration_error(self.records, num_bins)
        max_err = max((b["calibration_error"] for b in bins if b["count"] > 0), default=0.0)

        return {
            "total_evaluated": n,
            "mean_brier_score": mean_brier,
            "mean_log_loss": mean_log_loss,
            "baseline_brier_score": mean_baseline_brier,
            "brier_skill_score": bss,
            "expected_calibration_error": ece,
            "max_calibration_error": max_err,
            "calibration_bins": bins,
        }
