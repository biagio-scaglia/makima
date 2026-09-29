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


def handle_parse_intent(text: str) -> None:
    """Emette solo StructuredIntent JSON (contratto verso il core Rust)."""
    pipeline = SemanticForecastPipeline()
    structured = pipeline.process_intent(text)
    # JSON compatto su stdout (nessun testo decorativo) per il parsing dalla CLI Rust.
    print(structured.to_json_compact())


def _explain_payload(target_name: str) -> tuple[str, str, dict]:
    """Calcola explain grounded; restituisce (testo, provenance, meta)."""
    from makima_lab.llm import get_llm_engine
    from makima_lab.storage import load_store, compute_knowledge_base_from_store

    engine = get_llm_engine()
    kb = compute_knowledge_base_from_store(load_store())
    evidence = kb.get(target_name)
    if evidence is None:
        alpha, beta = 1.0, 1.0
        poisson_rate = 0.0
        evidence_count = 0
    else:
        alpha = 1.0 + float(evidence["successes"])
        beta = 1.0 + float(evidence["failures"])
        poisson_rate = float(evidence.get("rate_per_day", 0.0))
        evidence_count = int(evidence["successes"] + evidence["failures"])

    prob = alpha / (alpha + beta)
    variance = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))
    explanation = engine.explain_target(
        target=target_name,
        probability=prob,
        alpha=alpha,
        beta=beta,
        evidence_count=evidence_count,
        poisson_rate=poisson_rate,
        variance=variance,
    )
    meta = {
        "target": target_name,
        "n": evidence_count,
        "alpha": alpha,
        "beta": beta,
        "prob": prob,
    }
    return str(explanation), engine.last_source, meta


def handle_explain(target_name: str) -> None:
    """Genera una spiegazione ancorata ai dati empirici dello store (SLM o fallback)."""
    explanation, source, meta = _explain_payload(target_name)
    print(f"\n[ Makima Explain grounded: {meta['target']} ]")
    print(
        f"Dati store → N={meta['n']}, Beta({meta['alpha']:.2f},{meta['beta']:.2f}), "
        f"E[P]={meta['prob'] * 100:.1f}%"
    )
    print("---------------------------------------------------")
    print(explanation)
    print("---------------------------------------------------")
    print(f"Provenance SLM: {source}\n")


def handle_slm_explain_bridge(target_name: str) -> None:
    """Stdout pulito per IPC Tauri: testo + riga PROVENANCE."""
    explanation, source, meta = _explain_payload(target_name)
    header = (
        f"Target `{meta['target']}` — N={meta['n']}, "
        f"Beta({meta['alpha']:.2f},{meta['beta']:.2f}), E[P]={meta['prob'] * 100:.1f}%\n\n"
    )
    print(header + explanation)
    print(f"PROVENANCE:{source}")


def handle_slm_chat_bridge(message: str) -> None:
    """Chat Qwen grounded su contesto store (bridge GUI)."""
    from makima_lab.llm import get_llm_engine
    from makima_lab.storage import load_store, compute_knowledge_base_from_store

    engine = get_llm_engine()
    store = load_store()
    kb = compute_knowledge_base_from_store(store)
    context: dict = {}
    ranked = sorted(
        kb.items(),
        key=lambda kv: int(kv[1].get("successes", 0)) + int(kv[1].get("failures", 0)),
        reverse=True,
    )
    for name, data in ranked[:5]:
        s = int(data.get("successes", 0))
        f = int(data.get("failures", 0))
        prob = (s + 1) / (s + f + 2)
        context[f"target:{name}"] = f"E[P]={prob * 100:.1f}% N={s + f}"
    metrics = store.get("metrics") or {}
    if "brier_score" in metrics:
        context["brier_score"] = round(float(metrics["brier_score"]), 4)
    context["osservazioni_totali"] = len(store.get("observations") or [])
    text = engine.chat(message, context=context)
    print(str(text))
    print(f"PROVENANCE:{engine.last_source}")


def handle_remember_bridge(text: str) -> None:
    """Memorizza un fatto (tell) con stdout minimale per IPC."""
    from makima_lab.mind import EpisodicMemoryStore

    fact = EpisodicMemoryStore().record_developer_fact(text)
    try:
        from makima_lab.neural import get_neural_engine

        engine = get_neural_engine()
        engine.perceive(text, update_memory=True)
        engine.learn_step(
            text=text,
            intent_label="fact",
            action_label="REMEMBER",
            auto_save=True,
        )
    except Exception:
        pass
    entry_id = record_journal_entry(
        content=text,
        tags=["tell", "gui"],
        metadata={"episodic_id": fact.experience_id},
    )
    print(
        f"Memorizzato: «{text}»\n"
        f"Diario: {fact.experience_id} | SQLite entry #{entry_id}\n"
        f"Ora puoi chiedere in chat (modalità Previsione/Qwen) di richiamarlo."
    )
    print("PROVENANCE:memory")


def handle_digest() -> None:
    """Bollettino esecutivo ancorato allo store (nessun target inventato)."""
    from makima_lab.llm import get_llm_engine
    from makima_lab.storage import load_store, compute_knowledge_base_from_store

    engine = get_llm_engine()
    store = load_store()
    kb = compute_knowledge_base_from_store(store)

    sample_targets = []
    for target, data in kb.items():
        s = int(data["successes"])
        f = int(data["failures"])
        # Allineato a Laplace Beta(1+s,1+f)
        prob = (s + 1) / (s + f + 2)
        sample_targets.append({"name": target, "prob": prob, "obs": s + f})

    print("\n[ Makima Laplace Digest (grounded) ]")
    print("===================================================")
    digest = engine.generate_digest(sample_targets)
    print(digest)
    print(f"Provenance SLM: {engine.last_source}")
    print("===================================================\n")


def handle_chat(query: str, show_thought: bool = True) -> None:
    """Conversazione con cervello neurale (BrainLoop)."""
    from makima_lab.brain import get_brain

    result = get_brain().tick(query)
    pulse = result.pulse

    if show_thought:
        print("\n=======================================================")
        print("         CERVELLO NEURALE — MONOLOGO INTERIORE         ")
        print("=======================================================")
        print(pulse.inner_monologue)
        print("-------------------------------------------------------")
        print(result.format_user_guide())
        print(
            f"Stato: {pulse.self_state.mood.value} | "
            f"Incertezza: {pulse.self_state.epistemic_uncertainty:.4f} | "
            f"BSS: {pulse.self_state.brier_skill_score:+.2f}"
        )
        if pulse.retrieved_memories:
            print(f"Memorie Richiamate: {'; '.join(pulse.retrieved_memories[:2])}")
        print("=======================================================")

    print(f"\nMakima: {pulse.conscious_utterance}\n")


def handle_think(query: str) -> None:
    """Ispezione del tick completo del cervello neurale."""
    from makima_lab.brain import get_brain

    result = get_brain().tick(query)
    pulse = result.pulse

    print("\n[ Cervello neurale Makima — BrainLoop.tick ]")
    print("=======================================================")
    print(pulse.inner_monologue)
    print("=======================================================")
    print(result.format_user_guide())
    print(f"Stato Epistemico: {pulse.self_state.summary()}")
    print(f"Workspace L2: {result.workspace_norm:.4f}")
    print(f"Ipotesi: {', '.join(pulse.hypotheses) if pulse.hypotheses else '[Nessuna]'}")
    print("-------------------------------------------------------")
    print(f"Comunicazione: \"{pulse.conscious_utterance}\"\n")


def handle_pulse() -> None:
    """Impulso spontaneo via BrainLoop (stimulus idle)."""
    from makima_lab.brain import get_brain

    result = get_brain().tick("Impulso autonomo: cosa merita attenzione nello store?", learn=False)
    pulse = result.pulse

    print("\n[ Impulso di pensiero — cervello neurale ]")
    print("=======================================================")
    print(pulse.inner_monologue)
    print("=======================================================")
    print(result.format_user_guide())
    print(f"\nMakima: {pulse.conscious_utterance}\n")


def handle_chat_interactive() -> None:
    """Sessione continua sul BrainLoop neurale."""
    print("\n=======================================================")
    print("      MAKIMA NEURAL BRAIN (BrainLoop)                  ")
    print("=======================================================")
    print("Cervello operativo: percezione → memoria → azione.")
    print("I numeri di forecast restano in `makima query` (Rust).")
    print("Digita un messaggio (o 'esci')")
    print("=======================================================\n")
    from makima_lab.brain import get_brain

    brain = get_brain()

    while True:
        try:
            user_msg = input("tu > ").strip()
            if not user_msg:
                continue
            if user_msg.lower() in ("exit", "quit", "esci", "q", ":q"):
                print("\nChiusura sessione cervello.\n")
                break
            result = brain.tick(user_msg)
            print("\n" + "-" * 55)
            print(f"[Pensiero]:\n{result.pulse.inner_monologue}")
            print("-" * 55)
            print(result.format_user_guide())
            print(f"\nMakima: {result.pulse.conscious_utterance}\n")
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break


def handle_journal(text: str) -> None:
    """Registra un fatto: diario episodico + SQLite + apprendimento neurale."""
    from makima_lab.neural import get_neural_engine
    from makima_lab.mind import EpisodicMemoryStore

    engine = get_neural_engine()
    perception = engine.perceive(text, update_memory=True)

    loss = engine.learn_step(
        text=text,
        intent_label=perception.intent,
        action_label="REMEMBER",
        polarity_label=perception.polarity,
        auto_save=True,
    )

    # Diario autobiografico usato da chat/think (BrainLoop)
    fact = EpisodicMemoryStore().record_developer_fact(text)

    tags = [perception.intent, "tell"]
    entry_id = record_journal_entry(
        content=text,
        tags=tags,
        metadata={
            "neural_intent": perception.intent,
            "confidence": perception.intent_confidence,
            "polarity": perception.polarity,
            "memory_norm": perception.memory_norm,
            "loss": loss,
            "episodic_id": fact.experience_id,
        },
    )

    print("\n[ Makima ha percepito e memorizzato ]")
    print(perception.format_report())
    print(f"\n-> Diario episodico: .makima/mind_journal.jsonl [{fact.experience_id}]")
    print(f"-> SQLite journal:   .makima/makima.db [Entry #{entry_id}]")
    print(f"-> Apprendimento neurale (Loss: {loss:.4f}, Mem L2: {perception.memory_norm:.4f})")
    print("-> Ora puoi richiamarlo con: chat / think\n")


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
    print("FORECASTING_PATH:          False (lab — non usa questi prior per il core)")
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
            struct = pipeline.process_intent(text)
            print("\n[ StructuredIntent — nessun Bayes Python ]")
            print(struct.summary())
            print(f"\nJSON: {struct.to_json_compact()}")
            print(f'\nPer il forecast: makima query "{text}"\n')
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
        elif subcmd in ("slm-explain", "slm_explain") and len(sys.argv) > 2:
            handle_slm_explain_bridge(" ".join(sys.argv[2:]))
        elif subcmd in ("slm-chat", "slm_chat") and len(sys.argv) > 2:
            handle_slm_chat_bridge(" ".join(sys.argv[2:]))
        elif subcmd in ("remember",) and len(sys.argv) > 2:
            handle_remember_bridge(" ".join(sys.argv[2:]))
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
        elif subcmd in ("parse-intent", "parse", "intent-json") and len(sys.argv) > 2:
            text = " ".join(sys.argv[2:])
            handle_parse_intent(text)
        elif subcmd in ("query", "nlp") and len(sys.argv) > 2:
            text = " ".join(sys.argv[2:])
            # Solo StructuredIntent: il Bayes di produzione è in Rust (`makima query`).
            pipeline = SemanticForecastPipeline()
            struct = pipeline.process_intent(text)
            print("\n[ StructuredIntent — contratto verso Rust Core ]")
            print(struct.summary())
            print("\nJSON compatto:")
            print(struct.to_json_compact())
            print(
                "\nNessun calcolo Bayes in Python. Per la previsione usa:\n"
                f'  makima query "{text}"\n'
            )
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
