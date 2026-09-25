"""CLI e Console Interattiva per Makima Python Lab."""

import sys
from pathlib import Path

# Assicura che makima_lab sia importabile
sys.path.insert(0, str(Path(__file__).parent.parent))

from makima_lab.distributions import Bernoulli, BetaDistribution, PoissonDistribution


def print_banner():
    print("===================================================")
    print("             MAKIMA PYTHON RESEARCH LAB            ")
    print("===================================================")
    print("Ambiente di Ricerca Statistica e Validazione")
    print("Distribuzioni caricate: Bernoulli, BetaDistribution, PoissonDistribution")
    print("Digita 'demo', 'update', 'test', o 'exit' per uscire.")
    print("===================================================\n")


def run_demo():
    print("--- Demo: Inferenza Bayesiana con Prior Coniugato Beta ---")
    prior = BetaDistribution(1.0, 1.0)
    print(f"1. Prior non-informativo (uniforme):")
    print(f"   {prior}")
    print(f"   {prior.ascii_density()}")
    print()

    print("2. Osservazione di 7 successi su 10 tentativi...")
    post1 = prior.bayesian_update(successes=7, failures=3)
    print(f"   Posterior: {post1}")
    print(f"   {post1.ascii_density()}")
    print()

    print("3. Arrivo di ulteriori 15 successi e 2 fallimenti...")
    post2 = post1.bayesian_update(successes=15, failures=2)
    print(f"   Posterior raffinato: {post2}")
    print(f"   {post2.ascii_density()}")
    print(f"   Varianza ridotta (incertezza epistemica calata): {post2.variance:.6f}")
    print()


def interactive_loop():
    print_banner()
    run_demo()

    while True:
        try:
            cmd = input("makima-lab> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nUscita dal laboratorio Python.")
            break

        if not cmd:
            continue
        if cmd in ("exit", "quit", "q"):
            print("Uscita dal laboratorio Python.")
            break
        elif cmd == "demo":
            run_demo()
        elif cmd.startswith("update"):
            parts = cmd.split()
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                s, f = int(parts[1]), int(parts[2])
                dist = BetaDistribution(1.0, 1.0).bayesian_update(s, f)
                print(f"Risultato: {dist}")
                print(dist.ascii_density())
            else:
                print("Uso: update <successi> <fallimenti>  (es: update 12 3)")
        elif cmd.startswith("bernoulli"):
            parts = cmd.split()
            if len(parts) == 2:
                try:
                    p = float(parts[1])
                    b = Bernoulli(p)
                    print(f"{b}")
                    print(f"Media: {b.mean:.4f}, Varianza: {b.variance:.4f}, Entropia: {b.entropy_bits:.4f} bit")
                except Exception as e:
                    print(f"Errore: {e}")
            else:
                print("Uso: bernoulli <p>  (es: bernoulli 0.73)")
        elif cmd.startswith("poisson"):
            parts = cmd.split()
            if len(parts) == 2:
                try:
                    lam = float(parts[1])
                    poi = PoissonDistribution(lam)
                    print(f"{poi}")
                    print(f"P(X=0): {poi.pmf(0):.4f}, P(X=1): {poi.pmf(1):.4f}, P(X=2): {poi.pmf(2):.4f}")
                except Exception as e:
                    print(f"Errore: {e}")
            else:
                print("Uso: poisson <lambda>  (es: poisson 3.5)")
        elif cmd in ("help", "h"):
            print("Comandi disponibili:")
            print("  demo                    Esegue la simulazione di aggiornamento bayesiano")
            print("  update <succ> <fail>    Calcola distribuzione Beta da successi/fallimenti")
            print("  bernoulli <p>           Analizza probabilità ed entropia di Bernoulli")
            print("  poisson <lambda>        Calcola probabilità di conteggio Poisson")
            print("  exit / quit             Torna alla console principale")
        else:
            print(f"Comando non riconosciuto: '{cmd}'. Digita 'help' per la lista comandi.")


if __name__ == "__main__":
    interactive_loop()
