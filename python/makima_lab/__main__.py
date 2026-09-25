"""CLI e Console Interattiva per Makima Python Lab & Neural Mind."""

import sys
from pathlib import Path

# Assicura la compatibilità con terminali Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Assicura che makima_lab sia importabile
sys.path.insert(0, str(Path(__file__).parent.parent))

from makima_lab.distributions import Bernoulli, BetaDistribution, PoissonDistribution
from makima_lab.nlp import SemanticForecastPipeline
from makima_lab.neural import get_neural_engine
from makima_lab.storage import record_journal_entry


def print_banner():
    print("===================================================")
    print("        MAKIMA PYTHON RESEARCH LAB & NEURAL MIND   ")
    print("===================================================")
    print("Ambiente di Ricerca: PyTorch Cognitive Net, NLP, Bayes")
    print("Comandi disponibili:")
    print("  - tell <pensiero/fatto>     : Confida un fatto a Makima (NLP + Neural + SQLite)")
    print("  - neural <frase>            : Ispezione della percezione neurale (Self-Attention)")
    print("  - memory                    : Visualizza lo stato di memoria latente utente")
    print("  - query <domanda>           : Risoluzione semantica e calcolo probabilistico")
    print("  - update <succ> <fail>      : Calcolo distribuzione coniugata Beta")
    print("  - bernoulli <p> / poisson <lambda>")
    print("  - exit                      : Torna al launcher principale")
    print("===================================================\n")


def handle_journal(text: str) -> None:
    """Registra una frase dell'utente, la percepisce con la rete neurale, aggiorna la memoria e salva su SQLite."""
    engine = get_neural_engine()
    perception = engine.perceive(text, update_memory=True)

    # Passo di apprendimento online automatico
    loss = engine.learn_step(
        text=text,
        intent_label=perception.intent,
        polarity_label=perception.polarity,
        auto_save=True,
    )

    tags = [perception.intent]
    entry_id = record_journal_entry(
        content=text,
        tags=tags,
        metadata={
            "neural_intent": perception.intent,
            "confidence": perception.intent_confidence,
            "polarity": perception.polarity,
            "memory_norm": perception.memory_norm,
            "loss": loss,
        },
    )

    print("\n[ Makima ha percepito e memorizzato ]")
    print(perception.format_report())
    print(f"\n-> Registrato in SQLite (.makima/makima.db) [Entry #{entry_id}]")
    print(f"-> Passo di apprendimento neurale completato (Loss AdamW: {loss:.4f})")
    print(f"-> Memoria latente utente aggiornata (Norma L2: {perception.memory_norm:.4f})\n")


def handle_neural_inspection(text: str) -> None:
    engine = get_neural_engine()
    res = engine.perceive(text, update_memory=False)
    print("\n" + res.format_report() + "\n")


def handle_memory_status() -> None:
    engine = get_neural_engine()
    import torch
    norm = float(torch.norm(engine.user_memory).item())
    vec_sample = engine.user_memory[0, :8].tolist()
    print("\n--- [ Stato Memoria Latente Utente (GRU Cell) ] ---")
    print(f"Norma L2 Totale:          {norm:.4f}")
    print(f"Passi di Apprendimento:   {engine.total_learning_steps}")
    print(f"Campione Vettore [0..7]:  {[round(x, 4) for x in vec_sample]}")
    print(f"Dispositivo PyTorch:      {engine.device}")
    print("--------------------------------------------------\n")


def interactive_loop():
    print_banner()
    pipeline = SemanticForecastPipeline()

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
        elif cmd.startswith("tell ") or cmd.startswith("journal "):
            text = cmd.split(maxsplit=1)[1].strip()
            handle_journal(text)
        elif cmd.startswith("neural "):
            text = cmd.split(maxsplit=1)[1].strip()
            handle_neural_inspection(text)
        elif cmd == "memory":
            handle_memory_status()
        elif cmd.startswith("query ") or cmd.startswith("nlp "):
            text = cmd.split(maxsplit=1)[1].strip()
            result = pipeline.execute(text)
            print()
            print(result.format_report())
            print()
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
            print("  tell <testo>            Confida un fatto, pensiero o abitudine a Makima")
            print("  neural <frase>          Analizza la rappresentazione neurale e attention")
            print("  memory                  Mostra la memoria latente e i pesi appresi")
            print("  query <frase>           Analizza una frase naturale con pipeline probabilistica")
            print("  update <succ> <fail>    Calcola distribuzione Beta da successi/fallimenti")
            print("  bernoulli <p>           Analizza probabilità ed entropia di Bernoulli")
            print("  poisson <lambda>        Calcola probabilità di conteggio Poisson")
            print("  exit / quit             Torna alla console principale")
        else:
            print(f"Comando non riconosciuto: '{cmd}'. Digita 'help' per la lista comandi.")


def main():
    if len(sys.argv) > 1:
        subcmd = sys.argv[1]
        if subcmd in ("tell", "journal") and len(sys.argv) > 2:
            text = " ".join(sys.argv[2:])
            handle_journal(text)
        elif subcmd == "neural" and len(sys.argv) > 2:
            text = " ".join(sys.argv[2:])
            handle_neural_inspection(text)
        elif subcmd == "memory":
            handle_memory_status()
        elif subcmd in ("query", "nlp") and len(sys.argv) > 2:
            text = " ".join(sys.argv[2:])
            pipeline = SemanticForecastPipeline()
            res = pipeline.execute(text)
            print("\n" + res.format_report() + "\n")
        else:
            interactive_loop()
    else:
        interactive_loop()


if __name__ == "__main__":
    main()
