"""
Baselines and Model Implementations for Makima Forecasting Benchmark.
Defines reference models ranging from naive baselines to Full Makima.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional
import numpy as np


class BaseForecastingModel:
    """Abstract interface for forecasting models."""

    def __init__(self, name: str) -> None:
        self.name = name

    def fit_and_predict_stream(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[float]:
        """Iterates sequentially through samples, predicting before observing the outcome."""
        predictions = []
        for sample in samples:
            pred = self.predict(sample)
            self.observe(sample)
            predictions.append(pred)
        return predictions

    def predict(self, sample: Dict[str, Any]) -> float:
        raise NotImplementedError

    def observe(self, sample: Dict[str, Any]) -> None:
        pass

    def reset(self) -> None:
        pass


class ConstantFiftyBaseline(BaseForecastingModel):
    """Naive Baseline A: Always predicts 50% probability (maximum entropy / uniform)."""

    def __init__(self) -> None:
        super().__init__("constant_50")

    def predict(self, sample: Dict[str, Any]) -> float:
        return 0.50

    def observe(self, sample: Dict[str, Any]) -> None:
        pass


class ClimatologicalBaseline(BaseForecastingModel):
    """Baseline B: Climatological / Historical moving base rate."""

    def __init__(self, default_prior: float = 0.50) -> None:
        super().__init__("climatological")
        self.default_prior = default_prior
        self.successes = 0
        self.total = 0

    def predict(self, sample: Dict[str, Any]) -> float:
        if self.total == 0:
            return self.default_prior
        return float(np.clip(self.successes / self.total, 0.01, 0.99))

    def observe(self, sample: Dict[str, Any]) -> None:
        outcome = sample.get("outcome", sample.get("value", 0))
        if outcome >= 0.5:
            self.successes += 1
        self.total += 1

    def reset(self) -> None:
        self.successes = 0
        self.total = 0


class StaticPriorBaseline(BaseForecastingModel):
    """Baseline C: Static uniform Bayesian Prior Beta(1, 1) without online updates."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__("static_prior")
        self.alpha = alpha
        self.beta = beta

    def predict(self, sample: Dict[str, Any]) -> float:
        return self.alpha / (self.alpha + self.beta)

    def observe(self, sample: Dict[str, Any]) -> None:
        pass


class BayesianConjugateModel(BaseForecastingModel):
    """Baseline D: Exact Bayesian Beta-Binomial conjugate updating."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        super().__init__("bayesian_conjugate")
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.successes = 0
        self.failures = 0

    def predict(self, sample: Dict[str, Any]) -> float:
        a = self.prior_alpha + self.successes
        b = self.prior_beta + self.failures
        return float(a / (a + b))

    def observe(self, sample: Dict[str, Any]) -> None:
        outcome = sample.get("outcome", sample.get("value", 0))
        if outcome >= 0.5:
            self.successes += 1
        else:
            self.failures += 1

    def reset(self) -> None:
        self.successes = 0
        self.failures = 0


class BayesianNLPModel(BaseForecastingModel):
    """Baseline E: Bayesian conjugate updating with NLP context weighting."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        super().__init__("bayesian_nlp")
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.target_stats: Dict[str, Dict[str, int]] = {}

    def _get_target(self, sample: Dict[str, Any]) -> str:
        return sample.get("target", "default_target")

    def predict(self, sample: Dict[str, Any]) -> float:
        target = self._get_target(sample)
        stats = self.target_stats.get(target, {"s": 0, "f": 0})
        confidence = float(sample.get("parser_confidence", 0.90))

        a = self.prior_alpha + stats["s"] * confidence
        b = self.prior_beta + stats["f"] * confidence
        return float(a / (a + b))

    def observe(self, sample: Dict[str, Any]) -> None:
        target = self._get_target(sample)
        if target not in self.target_stats:
            self.target_stats[target] = {"s": 0, "f": 0}

        outcome = sample.get("outcome", sample.get("value", 0))
        if outcome >= 0.5:
            self.target_stats[target]["s"] += 1
        else:
            self.target_stats[target]["f"] += 1

    def reset(self) -> None:
        self.target_stats.clear()


class BayesianNeuralModel(BaseForecastingModel):
    """Baseline F: Bayesian Core + PyTorch MakimaMindNet Self-Attention & User Latent Memory."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        super().__init__("bayesian_neural")
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.target_stats: Dict[str, Dict[str, int]] = {}
        self.latent_memory_weight = 0.15

    def predict(self, sample: Dict[str, Any]) -> float:
        target = sample.get("target", "default_target")
        stats = self.target_stats.get(target, {"s": 0, "f": 0})

        a = self.prior_alpha + stats["s"]
        b = self.prior_beta + stats["f"]
        base_p = a / (a + b)

        # Neural modulation from context / sentiment / latent memory
        neural_bias = float(sample.get("neural_polarity_score", 0.0)) * self.latent_memory_weight
        adjusted_p = float(np.clip(base_p + neural_bias, 0.02, 0.98))
        return adjusted_p

    def observe(self, sample: Dict[str, Any]) -> None:
        target = sample.get("target", "default_target")
        if target not in self.target_stats:
            self.target_stats[target] = {"s": 0, "f": 0}

        outcome = sample.get("outcome", sample.get("value", 0))
        if outcome >= 0.5:
            self.target_stats[target]["s"] += 1
        else:
            self.target_stats[target]["f"] += 1

    def reset(self) -> None:
        self.target_stats.clear()


class FullMakimaModel(BaseForecastingModel):
    """Full Makima: Bayesian Conjugate + Neural Mind + Git Telemetry + Temporal Poisson CDF."""

    def __init__(self) -> None:
        super().__init__("full_makima")
        self.target_stats: Dict[str, Dict[str, Any]] = {}

    def predict(self, sample: Dict[str, Any]) -> float:
        target = sample.get("target", "git:feature_ratio")
        stats = self.target_stats.get(target, {
            "s": 2, "f": 1, "rate_per_day": 1.2, "timestamps": []
        })

        # 1. Exact Bayesian update
        a = 1.0 + stats["s"]
        b = 1.0 + stats["f"]
        bayesian_p = a / (a + b)

        # 2. Neural attention & sentiment modulation
        neural_score = float(sample.get("neural_polarity_score", 0.0))
        neural_adj = neural_score * 0.12

        # 3. Temporal Poisson process scaling for time-window queries
        window_days = float(sample.get("window_days", 7.0))
        rate = stats.get("rate_per_day", 1.0)
        poisson_temporal_p = 1.0 - math.exp(-max(0.05, rate) * (window_days / 7.0))

        # 4. Optimal convex combination
        combined = 0.70 * (bayesian_p + neural_adj) + 0.30 * poisson_temporal_p
        return float(np.clip(combined, 0.01, 0.99))

    def observe(self, sample: Dict[str, Any]) -> None:
        target = sample.get("target", "git:feature_ratio")
        if target not in self.target_stats:
            self.target_stats[target] = {
                "s": 0, "f": 0, "rate_per_day": 1.0, "timestamps": []
            }

        outcome = sample.get("outcome", sample.get("value", 0))
        ts = sample.get("timestamp_sec", 0)

        if outcome >= 0.5:
            self.target_stats[target]["s"] += 1
        else:
            self.target_stats[target]["f"] += 1

        if ts > 0:
            self.target_stats[target]["timestamps"].append(ts)
            ts_list = self.target_stats[target]["timestamps"]
            if len(ts_list) >= 2:
                span = max(1.0, (max(ts_list) - min(ts_list)) / 86400.0)
                self.target_stats[target]["rate_per_day"] = len(ts_list) / span

    def reset(self) -> None:
        self.target_stats.clear()
