"""Esperimento 01: Benchmark di Calibrazione e Valutazione Predittiva.

Questo script simula una sequenza temporale di eventi con un tasso di verità
fondamentale (Ground Truth) p = 0.72. Dimostra come Makima aggiorna il posterior,
riduce l'incertezza e batte la baseline ingenua sul Brier Score.
"""

import random
import sys
from pathlib import Path

# Inclusione path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from makima_lab.distributions import BetaDistribution
from makima_lab.evaluation import Evaluator


def run_benchmark(num_steps: int = 50, true_rate: float = 0.72, seed: int = 42):
    random.seed(seed)
    print("============================================================")
    print("      ESPERIMENTO 01: BENCHMARK DI CALIBRAZIONE MAKIMA      ")
    print("============================================================")
    print(f"Parametri: {num_steps} passi temporali, Ground Truth p={true_rate:.2f}, Seed={seed}\n")

    prior = BetaDistribution.uniform()
    current_model = prior
    evaluator = Evaluator()

    print(f"Passo 0 | Prior: {current_model} | Stima Iniziale: {current_model.mean:.3f}")

    for t in range(1, num_steps + 1):
        # 1. Makima emette una previsione prima di conoscere il futuro
        prediction = current_model.mean

        # 2. Il mondo reale manifesta l'esito reale secondo la natura
        outcome = random.random() < true_rate

        # 3. L'esito viene registrato e valutato contro la stima
        evaluator.add(prediction, outcome)

        # 4. Makima aggiorna il proprio stato a posteriori per i passi futuri
        current_model = current_model.bayesian_update(
            successes=1 if outcome else 0,
            failures=0 if outcome else 1
        )

        if t in (5, 10, 25, 50):
            print(
                f"Passo {t:2d} | Esito: {int(outcome)} | Pred: {prediction:.3f} | "
                f"Nuovo Posterior: E[P]={current_model.mean:.3f}, Var={current_model.variance:.5f}"
            )

    # Valutazione finale
    report = evaluator.evaluate(baseline_prob=0.5)
    print("\n============================================================")
    print("                   REPORT DI VALUTAZIONE FINALE             ")
    print("============================================================")
    print(f"Previsioni Totali:         {report['total_evaluated']}")
    print(f"Brier Score Modello:       {report['mean_brier_score']:.4f}  (piu' basso e', meglio e')")
    print(f"Brier Score Baseline 50%:  {report['baseline_brier_score']:.4f}")
    print(f"Brier Skill Score (BSS):   {report['brier_skill_score'] * 100:+.2f}%  (Miglioramento relativo)")
    print(f"Log Loss (Cross-Entropy):  {report['mean_log_loss']:.4f}")
    print("============================================================\n")


if __name__ == "__main__":
    run_benchmark()
