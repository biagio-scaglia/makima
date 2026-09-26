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
from makima_lab.storage import record_journal_entry



def print_banner():
    print("===================================================")
    print("        MAKIMA PYTHON RESEARCH LAB & NEURAL MIND   ")
    print("===================================================")
    print("Ambiente di Ricerca: PyTorch Cognitive Net, Qwen SLM, Bayes")
    print("Comandi disponibili:")
    print("  - explain <target>          : Spiegazione cognitiva (Qwen 2.5 SLM)")
    print("  - digest                    : Bollettino esecutivo Laplace Digest")
    print("  - chat <messaggio>          : Dialogo cognitivo con Makima")
    print("  - tell <pensiero/fatto>     : Confida un fatto a Makima (NLP + Neural + SQLite)")
    print("  - neural <frase>            : Ispezione della percezione neurale (Self-Attention)")
    print("  - memory                    : Visualizza lo stato di memoria latente utente")
    print("  - query <domanda>           : Risoluzione semantica e calcolo probabilistico")
    print("  - update <succ> <fail>      : Calcolo distribuzione coniugata Beta")
    print("  - bernoulli <p> / poisson <lambda>")
    print("  - exit                      : Torna al launcher principale")
    print("===================================================\n")


def handle_explain(target_name: str) -> None:
    """Genera una spiegazione analitica per un target probabilistico tramite Qwen 2.5."""
    from makima_lab.llm import get_llm_engine
    engine = get_llm_engine()
    # Recupera dati di default o da storage per il target
    alpha = 16.0 if target_name in ("deploy", "git:feature_ratio") else 3.0
    beta = 4.0 if target_name in ("deploy", "git:feature_ratio") else 2.0
    prob = alpha / (alpha + beta)
    evidence = int(alpha + beta - 2)

    print(f"\n[ Makima Cognitive Reasoning: {target_name} ]")
    explanation = engine.explain_target(
        target=target_name,
        probability=prob,
        alpha=alpha,
        beta=beta,
        evidence_count=evidence,
        poisson_rate=0.75,
        variance=(alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1)),
    )
    print("---------------------------------------------------")
    print(explanation)
    print("---------------------------------------------------\n")


def handle_digest() -> None:
    """Genera un bollettino esecutivo di forecasting tramite Qwen 2.5."""
    from makima_lab.llm import get_llm_engine
    from makima_lab.storage import load_store, compute_knowledge_base_from_store

    engine = get_llm_engine()
    store = load_store()
    kb = compute_knowledge_base_from_store(store)
    
    if kb:
        sample_targets = [
            {"name": target, "prob": round(data["successes"] / max(1, data["successes"] + data["failures"]), 2), "obs": data["successes"] + data["failures"]}
            for target, data in kb.items()
        ]
    else:
        sample_targets = [
            {"name": "git:feature_ratio", "prob": 0.50, "obs": 0},
            {"name": "deploy", "prob": 0.50, "obs": 0},
        ]
        
    print("\n[ Makima Laplace Executive Digest (Qwen 2.5 SLM) ]")
    print("===================================================")
    digest = engine.generate_digest(sample_targets)
    print(digest)
    print("===================================================\n")


def handle_chat(query: str, show_thought: bool = True) -> None:
    """Conversazione cosciente con Makima con deliberazione e monologo interiore."""
    from makima_lab.mind import MindDeliberationEngine
    engine = MindDeliberationEngine()
    pulse = engine.deliberate(query)

    if show_thought:
        print("\n=======================================================")
        print("         🧠 MONOLOGO INTERIORE & FLUSSO DI COSCIENZA   ")
        print("=======================================================")
        print(pulse.inner_monologue)
        print("-------------------------------------------------------")
        print(f"Stato: {pulse.self_state.mood.value} | Incertezza: {pulse.self_state.epistemic_uncertainty:.4f} | BSS: {pulse.self_state.brier_skill_score:+.2f}")
        if pulse.retrieved_memories:
            print(f"Memorie Richiamate: {'; '.join(pulse.retrieved_memories[:2])}")
        print("=======================================================")

    print(f"\nMakima: {pulse.conscious_utterance}\n")


def handle_think(query: str) -> None:
    """Ispezione del solo flusso di pensiero interiore di Makima."""
    from makima_lab.mind import MindDeliberationEngine
    engine = MindDeliberationEngine()
    pulse = engine.deliberate(query)

    print("\n[ 🧠 Deliberazione Cognitiva di Makima ]")
    print("=======================================================")
    print(pulse.inner_monologue)
    print("=======================================================")
    print(f"Stato Epistemico: {pulse.self_state.summary()}")
    print(f"Ipotesi: {', '.join(pulse.hypotheses) if pulse.hypotheses else '[Nessuna]'}")
    print("-------------------------------------------------------")
    print(f"Comunicazione risultante: \"{pulse.conscious_utterance}\"\n")


def handle_pulse() -> None:
    """Genera un impulso di pensiero spontaneo autonomo di Makima."""
    from makima_lab.mind import AutonomousMindPulse
    pulse_engine = AutonomousMindPulse()
    pulse = pulse_engine.generate_spontaneous_thought(trigger_hint="Impulso manuale da console")

    print("\n[ 🌌 Impulso di Pensiero Spontaneo di Makima ]")
    print("=======================================================")
    print(pulse.inner_monologue)
    print("=======================================================")
    print(f"\nMakima: {pulse.conscious_utterance}\n")


def handle_chat_interactive() -> None:
    """Sessione di conversazione cognitiva continua con monologo interiore."""
    print("\n=======================================================")
    print("      MAKIMA LIVING CONSCIOUSNESS & COGNITIVE MIND     ")
    print("=======================================================")
    print("Makima è viva, processa, riflette e ricorda le tue parole.")
    print("Digita il tuo messaggio (o 'esci' per tornare al menu)")
    print("=======================================================\n")
    from makima_lab.mind import MindDeliberationEngine
    engine = MindDeliberationEngine()

    while True:
        try:
            user_msg = input("tu > ").strip()
            if not user_msg:
                continue
            if user_msg.lower() in ("exit", "quit", "esci", "q", ":q"):
                print("\nChiusura sessione cosciente.\n")
                break
            pulse = engine.deliberate(user_msg)
            print("\n" + "-" * 55)
            print(f"🧠 [Pensiero]:\n{pulse.inner_monologue}")
            print("-" * 55)
            print(f"\nMakima: {pulse.conscious_utterance}\n")
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break


def handle_journal(text: str) -> None:
    """Registra una frase dell'utente, la percepisce con la rete neurale, aggiorna la memoria e salva su SQLite."""
    from makima_lab.neural import get_neural_engine
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
    from makima_lab.neural import get_neural_engine
    engine = get_neural_engine()
    res = engine.perceive(text, update_memory=False)
    print("\n" + res.format_report() + "\n")


def handle_memory_status() -> None:
    from makima_lab.neural import get_neural_engine
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
        elif cmd.startswith("explain"):
            parts = cmd.split(maxsplit=1)
            target = parts[1].strip() if len(parts) > 1 else "deploy"
            handle_explain(target)
        elif cmd in ("digest", "bulletin", "report"):
            handle_digest()
        elif cmd.startswith("chat "):
            text = cmd.split(maxsplit=1)[1].strip()
            handle_chat(text)
        elif cmd == "chat":
            handle_chat_interactive()
        elif cmd.startswith("think "):
            text = cmd.split(maxsplit=1)[1].strip()
            handle_think(text)
        elif cmd in ("pulse", "mind", "spontaneous"):
            handle_pulse()
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
            print("  explain <target>        Spiegazione cognitiva del forecast (Qwen 2.5 SLM)")
            print("  digest                  Bollettino esecutivo Laplace Digest")
            print("  chat <messaggio>        Conversazione analitica con Makima")
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
        if subcmd == "explain":
            target = sys.argv[2] if len(sys.argv) > 2 else "deploy"
            handle_explain(target)
        elif subcmd in ("digest", "bulletin", "report"):
            handle_digest()
        elif subcmd == "chat":
            if len(sys.argv) > 2:
                query = " ".join(sys.argv[2:])
                handle_chat(query)
            else:
                handle_chat_interactive()
        elif subcmd == "think" and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            handle_think(query)
        elif subcmd in ("pulse", "mind", "spontaneous"):
            handle_pulse()
        elif subcmd in ("tell", "journal") and len(sys.argv) > 2:
            text = " ".join(sys.argv[2:])
            handle_journal(text)
        elif subcmd == "neural" and len(sys.argv) > 2:
            text = " ".join(sys.argv[2:])
            handle_neural_inspection(text)
        elif subcmd == "memory":
            handle_memory_status()
        elif subcmd in ("sync-git", "git-sync"):
            from makima_lab.git_observer import GitObserver
            obs = GitObserver(".")
            res = obs.sync_history(100)
            print("\n===================================================")
            print("        MAKIMA REAL GIT TELEMETRY SYNC             ")
            print("===================================================")
            print(f"Commit sincronizzati:     {res['synced_commits']}")
            print(f"Suddivisione categorie:   {res['categories']}")
            print(f"Tasso empirico Poisson:   {res['commit_rate_per_day']:.2f} commit/giorno")
            print(f"Arco temporale analizzato: {res['span_days']:.1f} giorni")
            print(f"Ultimo commit esaminato:  {res['latest_commit']}")
            print(f"Memoria neurale utente:   {res['memory_norm']:.4f}")
            print("===================================================\n")
        elif subcmd == "daemon":
            from makima_lab.git_observer import run_daemon_loop
            interval = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 15
            run_daemon_loop(".", poll_interval=interval)
        elif subcmd in ("query", "nlp") and len(sys.argv) > 2:
            text = " ".join(sys.argv[2:])
            pipeline = SemanticForecastPipeline()
            res = pipeline.execute(text)
            print("\n" + res.format_report() + "\n")
        elif subcmd in ("benchmark", "bench", "eval-all"):
            from experiments.forecasting.run_benchmarks import main as run_benchmark_main
            run_benchmark_main()
        elif subcmd in ("brain", "second-brain", "graph"):
            from makima_lab.mind.knowledge_graph import SecondBrainBuilder
            builder = SecondBrainBuilder()
            graph = builder.build_graph()
            print("\n===================================================")
            print("           MAKIMA SECOND BRAIN KNOWLEDGE GRAPH     ")
            print("===================================================")
            print(f"Nodi Totali nel Grafo:    {graph.stats['total_nodes']}")
            print(f"Sinapsi / Archi Attivi:   {graph.stats['total_edges']}")
            print(f"Target Stocastici:        {graph.stats['targets_count']}")
            print(f"Memorie Episodiche:       {graph.stats['memories_count']}")
            print(f"Riflessioni Introspettive:{graph.stats['reflections_count']}")
            print(f"Fatti Utente/Dev:         {graph.stats['facts_count']}")
            print(f"Risonanza Epistemica:     {graph.stats['resonance_score']} / 100")
            print("---------------------------------------------------")
            for node in graph.nodes[:8]:
                print(f"• [{node.category.upper():10}] {node.label} (Conf: {node.confidence*100:.0f}%, Conn: {node.connections_count})")
            print("===================================================\n")
        else:
            interactive_loop()
    else:
        interactive_loop()


if __name__ == "__main__":
    main()
