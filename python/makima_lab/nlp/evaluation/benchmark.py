"""Runner di benchmark e valutazione quantitativa per la pipeline NLP di Makima."""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any
from makima_lab.nlp.pipeline import MakimaNLPPipeline
from makima_lab.nlp.evaluation.dataset import BENCHMARK_DATASET, NLPEvaluationSample
from makima_lab.nlp.schemas.structured_intent import StructuredIntent


@dataclass
class EvaluationMetrics:
    """Metriche aggregate della valutazione NLP."""
    total_samples: int = 0
    intent_correct: int = 0
    target_correct: int = 0
    temporal_correct: int = 0
    core_validity_correct: int = 0
    invalid_output_count: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def intent_accuracy(self) -> float:
        return (self.intent_correct / self.total_samples) if self.total_samples > 0 else 0.0

    @property
    def target_accuracy(self) -> float:
        return (self.target_correct / self.total_samples) if self.total_samples > 0 else 0.0

    @property
    def temporal_accuracy(self) -> float:
        return (self.temporal_correct / self.total_samples) if self.total_samples > 0 else 0.0

    @property
    def core_validity_agreement(self) -> float:
        return (self.core_validity_correct / self.total_samples) if self.total_samples > 0 else 0.0

    @property
    def invalid_output_rate(self) -> float:
        return (self.invalid_output_count / self.total_samples) if self.total_samples > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return (sum(self.latencies_ms) / len(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "intent_accuracy": round(self.intent_accuracy, 4),
            "target_accuracy": round(self.target_accuracy, 4),
            "temporal_accuracy": round(self.temporal_accuracy, 4),
            "core_validity_agreement": round(self.core_validity_agreement, 4),
            "invalid_output_rate": round(self.invalid_output_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "error_count": len(self.errors),
        }

    def summary_report(self) -> str:
        lines = [
            "=" * 60,
            "  MAKIMA NLP PIPELINE - VALUTAZIONE QUANTITATIVA",
            "=" * 60,
            f"Campioni Totali Valutati:     {self.total_samples}",
            f"Intent Accuracy:              {self.intent_accuracy * 100:.1f}% ({self.intent_correct}/{self.total_samples})",
            f"Target Accuracy:              {self.target_accuracy * 100:.1f}% ({self.target_correct}/{self.total_samples})",
            f"Temporal Accuracy:            {self.temporal_accuracy * 100:.1f}% ({self.temporal_correct}/{self.total_samples})",
            f"Core Validity Agreement:      {self.core_validity_agreement * 100:.1f}% ({self.core_validity_correct}/{self.total_samples})",
            f"Invalid Output Rate:          {self.invalid_output_rate * 100:.1f}% ({self.invalid_output_count}/{self.total_samples})",
            f"Latenza Media:                {self.avg_latency_ms:.2f} ms",
            f"Latenza P95:                  {self.p95_latency_ms:.2f} ms",
            "=" * 60,
        ]
        if self.errors:
            lines.append("DISCREPANZE RILEVATE:")
            for err in self.errors:
                lines.append(
                    f" - [{err['sample']}] "
                    f"Intent: got {err['got_intent']} vs exp {err['exp_intent']} | "
                    f"Target: got {err['got_target']} vs exp {err['exp_target']} | "
                    f"Temporal: got {err['got_temporal']} vs exp {err['exp_temporal']} | "
                    f"Valid: got {err['got_valid']} vs exp {err['exp_valid']}"
                )
            lines.append("=" * 60)
        return "\n".join(lines)


class NLPBenchmarkRunner:
    """Esecutore della suite di benchmark locale."""

    def __init__(self, pipeline: MakimaNLPPipeline | None = None) -> None:
        self.pipeline = pipeline or MakimaNLPPipeline()

    def run(self, dataset: List[NLPEvaluationSample] | None = None) -> EvaluationMetrics:
        samples = dataset or BENCHMARK_DATASET
        metrics = EvaluationMetrics(total_samples=len(samples))

        for sample in samples:
            t0 = time.perf_counter()
            structured = self.pipeline.process_intent(sample.text)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            metrics.latencies_ms.append(elapsed_ms)

            # 1. Intent check
            intent_ok = (structured.intent == sample.expected_intent)
            if intent_ok:
                metrics.intent_correct += 1

            # 2. Target check
            target_ok = (structured.target == sample.expected_target)
            if target_ok:
                metrics.target_correct += 1

            # 3. Temporal check
            temp_ok = (structured.temporal_window.relation == sample.expected_temporal_relation)
            if temp_ok:
                metrics.temporal_correct += 1

            # 4. Validity check
            valid_ok = (structured.is_valid_for_core == sample.expected_valid_for_core)
            if valid_ok:
                metrics.core_validity_correct += 1

            if not structured.is_valid_for_core:
                metrics.invalid_output_count += 1

            # Log mismatch if any
            if not (intent_ok and target_ok and temp_ok and valid_ok):
                metrics.errors.append({
                    "sample": sample.text,
                    "got_intent": structured.intent.value,
                    "exp_intent": sample.expected_intent.value,
                    "got_target": structured.target,
                    "exp_target": sample.expected_target,
                    "got_temporal": structured.temporal_window.relation.value,
                    "exp_temporal": sample.expected_temporal_relation.value,
                    "got_valid": structured.is_valid_for_core,
                    "exp_valid": sample.expected_valid_for_core,
                    "confidence": structured.confidence,
                })

        return metrics


if __name__ == "__main__":
    runner = NLPBenchmarkRunner()
    results = runner.run()
    print(results.summary_report())
