"""Modulo di valutazione statistica, Proper Scoring Rules e calibrazione predittiva."""

import math


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


class Evaluator:
    """Valutatore empirico della qualità e calibrazione delle previsioni."""

    def __init__(self):
        self.records: list[tuple[float, bool]] = []

    def add(self, predicted_prob: float, actual_outcome: bool | int) -> None:
        """Aggiunge una coppia (previsione, esito reale)."""
        self.records.append((float(predicted_prob), bool(actual_outcome)))

    def evaluate(self, baseline_prob: float = 0.5) -> dict[str, float]:
        """Calcola metriche aggregate di scoring e confronto contro baseline."""
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

        return {
            "total_evaluated": n,
            "mean_brier_score": mean_brier,
            "mean_log_loss": mean_log_loss,
            "baseline_brier_score": mean_baseline_brier,
            "brier_skill_score": bss,
        }
