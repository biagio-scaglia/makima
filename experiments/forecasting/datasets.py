"""
Dataset Generators and Real Data Loaders for Makima Forecasting Benchmarks.
Generates reproducible synthetic streams (Stationary, Regime Shift, Poisson Arrivals) and Git replays.
"""

from __future__ import annotations
import math
import random
from typing import Any, Dict, List, Optional
import numpy as np


class SyntheticDatasetGenerator:
    """Generates synthetic probabilistic forecasting streams with known Ground Truth."""

    @staticmethod
    def generate_stationary_stream(
        num_samples: int = 1000,
        true_p: float = 0.72,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Generates an i.i.d. Bernoulli stream with true underlying probability true_p."""
        rng = np.random.RandomState(seed)
        samples = []
        base_ts = 1_700_000_000

        for i in range(num_samples):
            outcome = 1 if rng.rand() < true_p else 0
            # Noise in parser confidence and sentiment
            conf = float(rng.uniform(0.85, 0.98))
            polarity = float(rng.normal(0.2 if outcome == 1 else -0.2, 0.3))

            samples.append({
                "sample_id": i + 1,
                "target": "synthetic_stationary",
                "timestamp_sec": base_ts + i * 3600,
                "outcome": outcome,
                "true_probability": true_p,
                "parser_confidence": conf,
                "neural_polarity_score": np.clip(polarity, -1.0, 1.0),
                "window_days": 7.0,
            })
        return samples

    @staticmethod
    def generate_regime_shift_stream(
        num_samples: int = 1500,
        regimes: Optional[List[tuple[int, float]]] = None,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Generates a non-stationary stream with abrupt and gradual regime shifts.
        Simulates changing development cycles (e.g. feature sprint vs freeze period).
        """
        if regimes is None:
            regimes = [(500, 0.30), (500, 0.85), (500, 0.45)]

        rng = np.random.RandomState(seed)
        samples = []
        sample_idx = 1
        base_ts = 1_700_000_000

        for count, true_p in regimes:
            for _ in range(count):
                outcome = 1 if rng.rand() < true_p else 0
                conf = float(rng.uniform(0.80, 0.95))
                polarity = float(rng.normal(0.4 if true_p > 0.5 else -0.4, 0.35))

                samples.append({
                    "sample_id": sample_idx,
                    "target": "synthetic_regime_shift",
                    "timestamp_sec": base_ts + sample_idx * 3600,
                    "outcome": outcome,
                    "true_probability": true_p,
                    "parser_confidence": conf,
                    "neural_polarity_score": np.clip(polarity, -1.0, 1.0),
                    "window_days": 7.0,
                })
                sample_idx += 1

        return samples

    @staticmethod
    def generate_poisson_arrival_stream(
        num_samples: int = 1500,
        arrival_rate_per_day: float = 2.5,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Generates temporal arrival sequences with exponential inter-arrival times."""
        rng = np.random.RandomState(seed)
        samples = []
        current_ts = 1_700_000_000

        for i in range(num_samples):
            # Inter-arrival time Delta_t ~ Exp(lambda) in days
            delta_days = float(rng.exponential(1.0 / arrival_rate_per_day))
            current_ts += int(delta_days * 86400)

            # Query: will at least one event happen within 3 days?
            window_days = 3.0
            prob_window = 1.0 - math.exp(-arrival_rate_per_day * (window_days / 7.0))
            outcome = 1 if delta_days <= window_days else 0

            samples.append({
                "sample_id": i + 1,
                "target": "synthetic_poisson_stream",
                "timestamp_sec": current_ts,
                "outcome": outcome,
                "true_probability": prob_window,
                "parser_confidence": 0.92,
                "neural_polarity_score": float(rng.uniform(-0.3, 0.3)),
                "window_days": window_days,
            })

        return samples

    @staticmethod
    def generate_full_benchmark_suite(seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
        """Generates a comprehensive multi-dataset benchmark suite (5,000+ total samples)."""
        return {
            "stationary_stream_p72": SyntheticDatasetGenerator.generate_stationary_stream(
                num_samples=1500, true_p=0.72, seed=seed
            ),
            "stationary_stream_p25": SyntheticDatasetGenerator.generate_stationary_stream(
                num_samples=1000, true_p=0.25, seed=seed + 1
            ),
            "regime_shifts": SyntheticDatasetGenerator.generate_regime_shift_stream(
                num_samples=1500, seed=seed + 2
            ),
            "poisson_arrivals": SyntheticDatasetGenerator.generate_poisson_arrival_stream(
                num_samples=1000, arrival_rate_per_day=3.0, seed=seed + 3
            ),
        }
