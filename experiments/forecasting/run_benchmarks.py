"""
Master Scientific Benchmark & Ablation Study Execution Script for Makima.
Executes the full forecasting test suite, generates ASCII diagrams, and saves benchmark artifacts.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add python and root path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir / "python"))
sys.path.insert(0, str(root_dir))

from experiments.forecasting.ablation import AblationRunner
from experiments.forecasting.calibration import CalibrationEvaluator
from experiments.forecasting.evaluate import BenchmarkRunner


def main():
    print("========================================================================")
    print("      M A K I M A   -   S C I E N T I F I C   B E N C H M A R K         ")
    print("        Probabilistic Forecasting, Proper Scoring & Calibration        ")
    print("========================================================================\n")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run Benchmark Suite across All Models and Datasets
    print("[1/3] Esecuzione Benchmark su 4 Dataset Probabilistici (5,000 campioni)...")
    runner = BenchmarkRunner()
    benchmark_results = runner.run_full_suite(seed=42)

    benchmark_file = results_dir / "benchmark.json"
    with open(benchmark_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"-> Benchmark completato. Risultati salvati in: {benchmark_file}\n")

    # Display Aggregate Summary Table
    print("┌────────────────────────────────────────────────────────────────────────────────────────┐")
    print("│                     TABELLA COMPARATIVA MODELLI & BASELINE                             │")
    print("├───────────────────────┬─────────────┬──────────┬──────────┬──────────┬─────────┬───────┤")
    print("│ Modello               │ Brier Score │ Log Loss │ BSS (50%)│ BSS(Clim)│   ECE   │ Rating│")
    print("├───────────────────────┼─────────────┼──────────┼──────────┼──────────┼─────────┼───────┤")

    for model_name, m in benchmark_results["aggregate_summary"].items():
        print(
            f"│ {model_name:<21} │   {m['mean_brier_score']:.4f}    │  {m['mean_log_loss']:.4f}  │  {m['mean_bss_vs_50']:>+6.2f}% │  {m['mean_bss_vs_clim']:>+6.2f}% │ {m['mean_ece']*100:>5.2f}%  │ {m['calibration_rating']:<5} │"
        )
    print("└───────────────────────┴─────────────┴──────────┴──────────┴──────────┴─────────┴───────┘\n")

    # 2. Run Formal Ablation Study
    print("[2/3] Esecuzione Ablation Study (Isolamento Sottosistemi Makima)...")
    ablation_results = AblationRunner.run_ablation_study(seed=42)

    ablation_file = results_dir / "ablation.json"
    with open(ablation_file, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"-> Ablation Study completato. Risultati salvati in: {ablation_file}\n")

    print("┌────────────────────────────────────────────────────────────────────────────────────────┐")
    print("│                           MAKIMA ABLATION MATRIX                                       │")
    print("├───────────────────────────────┬─────────────┬──────────┬──────────┬──────────┬─────────┤")
    print("│ Variante Architetturale       │ Brier Score │ Log Loss │   ECE    │ Δ Brier  │ Degrado │")
    print("├───────────────────────────────┼─────────────┼──────────┼──────────┼──────────┼─────────┤")

    for v_name, v in ablation_results["variants"].items():
        deg = f"+{v.get('degradation_pct', 0.0):.1f}%" if v.get('degradation_pct', 0.0) > 0 else "0.0%"
        delta = f"+{v.get('delta_brier_vs_full', 0.0):.4f}" if v.get('delta_brier_vs_full', 0.0) > 0 else "0.0000"
        print(
            f"│ {v_name:<29} │   {v['brier_score']:.4f}    │  {v['log_loss']:.4f}  │  {v['ece']*100:>5.2f}% │  {delta:<7} │ {deg:<7} │"
        )
    print("└───────────────────────────────┴─────────────┴──────────┴──────────┴──────────┴─────────┘\n")

    # 3. Generate Reliability Diagram for Full Makima
    print("[3/3] Generazione Reliability Diagram per Full Makima...")
    sample_dataset = benchmark_results["datasets"]["regime_shifts"]
    # Recompute predictions on regime shifts
    from experiments.forecasting.baselines import FullMakimaModel
    from experiments.forecasting.datasets import SyntheticDatasetGenerator
    samples = SyntheticDatasetGenerator.generate_regime_shift_stream(num_samples=1500, seed=42)
    model = FullMakimaModel()
    preds = model.fit_and_predict_stream(samples)
    outcomes = [s["outcome"] for s in samples]
    calib = CalibrationEvaluator.compute_calibration(preds, outcomes, num_bins=10)
    diagram = CalibrationEvaluator.render_ascii_reliability_diagram(calib, model_name="Full Makima (Regime Shifts)")
    print(diagram)
    print("\n[OK] Validazione scientifica eseguita con successo al 100%!")


if __name__ == "__main__":
    main()
