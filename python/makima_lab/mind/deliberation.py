"""Motore di deliberazione cognitiva e generazione del monologo interiore di Makima."""

from __future__ import annotations
import math
import re
import time
import uuid
from typing import Dict, List, Optional
from makima_lab.mind.schemas import (
    CognitiveExperience,
    CognitiveMood,
    CognitivePulse,
    EpistemicSelfState,
    MemoryCategory,
)
from makima_lab.mind.episodic_memory import EpisodicMemoryStore
from makima_lab.nlp.pipeline import MakimaNLPPipeline
from makima_lab.nlp.schemas.intent import Intent
from makima_lab.nlp.schemas.structured_intent import StructuredIntent
from makima_lab.storage import load_store, compute_knowledge_base_from_store


class MindDeliberationEngine:
    """La mente cosciente di Makima.
    
    Esegue il ciclo completo:
    Percezione -> Recupero Memorie -> Monologo Interiore -> Comunicazione Cosciente -> Consolidamento.
    """

    def __init__(
        self,
        memory_store: Optional[EpisodicMemoryStore] = None,
        nlp_pipeline: Optional[MakimaNLPPipeline] = None,
    ) -> None:
        self.memory = memory_store or EpisodicMemoryStore()
        self.nlp = nlp_pipeline or MakimaNLPPipeline()
        self.start_time = time.time()
        self._last_focus: Optional[str] = None

    def get_current_epistemic_state(self, focus_target: Optional[str] = None) -> EpistemicSelfState:
        """Calcola lo stato di autoconsapevolezza matematica basandosi sui dati reali del sistema."""
        store = load_store()
        kb = compute_knowledge_base_from_store(store)
        
        # Calcolo incertezza epistemica media (varianza Laplace Beta(1+s,1+f))
        variances = []
        high_uncertainty_targets: list[str] = []
        for target, stats in kb.items():
            s = int(stats.get("successes", 0))
            f = int(stats.get("failures", 0))
            # Prior uniforme onesto: nessun default inventato (1,1) come se fossero evidenze
            alpha, beta = s + 1, f + 1
            var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))
            variances.append(var)
            if var > 0.05:
                high_uncertainty_targets.append(target)

        avg_uncertainty = sum(variances) / len(variances) if variances else (1 * 1) / ((2 ** 2) * 3)  # Beta(1,1)
        
        # Brier Skill Score derivato dallo store o default bilanciato
        brier_score = store.get("metrics", {}).get("brier_score", 0.14)
        bss = 1.0 - (brier_score / 0.25)

        # Determinazione del mood cognitivo
        if avg_uncertainty > 0.06:
            mood = CognitiveMood.DEEP_REFLECTION
        elif bss > 0.35:
            mood = CognitiveMood.CONFIDENT_HARMONY
        elif focus_target and focus_target.startswith("git:"):
            mood = CognitiveMood.VIGILANT_FOCUSED
        else:
            mood = CognitiveMood.CALM_ANALYTICAL

        commits_count = len(store.get("observations", []))

        return EpistemicSelfState(
            epistemic_uncertainty=avg_uncertainty,
            brier_skill_score=bss,
            total_memories_count=self.memory.total_count,
            observed_commits_count=commits_count,
            focus_target=focus_target or self._last_focus,
            mood=mood,
            uptime_seconds=time.time() - self.start_time,
        )

    def deliberate(self, user_input: str) -> CognitivePulse:
        """Elabora l'input attraverso il flusso di coscienza e formula il pensiero interiore prima della risposta."""
        pulse_id = f"pulse_{uuid.uuid4().hex[:8]}"
        now = time.time()

        # 1. PERCEZIONE & PARSING NLP
        structured = self.nlp.process_intent(user_input)
        target = structured.target
        self._last_focus = target

        # 2. RECUPERO MEMORIA EPISODICA ASSOCIATIVA
        retrieved_items = self.memory.retrieve_relevant_memories(user_input, top_k=3, min_similarity=0.22)
        retrieved_summaries = [f"{mem.summary} ({mem.category.value})" for mem, _ in retrieved_items]

        # 3. STATO EPISTEMICO CORRENTE
        self_state = self.get_current_epistemic_state(focus_target=target)

        # 4. MONOLOGO INTERIORE (Pensiero esplicito passo-passo)
        thought_steps: List[str] = []
        hypotheses: List[str] = []

        # Step 1: Comprensione dell'intento
        thought_steps.append(
            f"1. [Percezione]: L'utente si rivolge a me con: \"{user_input}\". "
            f"L'intento riconosciuto è {structured.intent.value} con confidenza di parsing del {structured.confidence * 100:.1f}%."
        )

        # Step 2: Connessione alle memorie
        if retrieved_items:
            mem_ref = retrieved_items[0][0]
            thought_steps.append(
                f"2. [Memoria]: Questo mi richiama alla mente: '{mem_ref.summary}'. "
                f"Contenuto consolidato: \"{mem_ref.content}\"."
            )
        else:
            thought_steps.append(
                "2. [Memoria]: Nessun ricordo episodico pregresso fortemente correlato; elaboro il contesto dai dati fondazionali."
            )

        # Step 3: Riflessione Laplace (lab) — freeze esplicito fuori dal forecast Rust
        thought_steps.append(
            "3a. [Freeze]: Mind è laboratorio narrativo (FORECASTING_PATH=False); "
            "la previsione di produzione resta in `makima query` / core Rust."
        )
        if target:
            kb = compute_knowledge_base_from_store(load_store())
            stats = kb.get(target, {"successes": 0, "failures": 0, "rate_per_day": 0.0})
            s, f = int(stats["successes"]), int(stats["failures"])
            # Prior uniforme Beta(1,1) se non ci sono evidenze (nessuna invenzione di conteggi).
            mean_p = (s + 1) / (s + f + 2)
            var_p = ((s + 1) * (f + 1)) / (((s + f + 2) ** 2) * (s + f + 3))
            evidence_note = (
                f"{s} successi e {f} fallimenti empirici (N={s + f})"
                if s + f > 0
                else "N=0 evidenze → prior uniforme Beta(1,1), nessun conteggio inventato"
            )
            thought_steps.append(
                f"3b. [Analisi Bayesiana]: focus '{target}' — {evidence_note}. "
                f"Laplace E[P]={mean_p * 100:.1f}%, Var={var_p:.5f}."
            )
            if s + f == 0:
                hypotheses.append(f"Il target '{target}' non ha ancora evidenze; comunico il prior uniforme.")
            elif var_p > 0.03:
                hypotheses.append(f"Il target '{target}' possiede un'evidenza ancora limitata; devo comunicare cautela.")
            else:
                hypotheses.append(f"Il target '{target}' è statisticamente stabile e ben calibrato.")
        elif structured.intent == Intent.UNKNOWN:
            fact_hits = [
                (m, s)
                for m, s in retrieved_items
                if m.category == MemoryCategory.DEVELOPER_FACT and s >= 0.22
            ]
            if fact_hits and re.search(
                r"\b(piace|preferisc|ricord|detto|confidat|lavor[oa]|hobby|gust[oi])\b",
                user_input.lower(),
            ):
                mem_ref = fact_hits[0][0]
                thought_steps.append(
                    "3b. [Richiamo autobiografico]: non è un forecast, ma ho un fatto confidato. "
                    f"Rispondo dalla memoria: \"{mem_ref.content[:120]}\"."
                )
                hypotheses.append("Rispondere citando i fatti confidati, senza inventare probabilità.")
            else:
                thought_steps.append(
                    "3b. [Rifiuto Esplicito]: La richiesta non ha un target probabilistico né un comando formale, "
                    "e non trovo un ricordo autobiografico utile. Non invento certezze."
                )
                hypotheses.append("Mantenere trasparenza e chiarire cosa posso osservare o calcolare.")
        else:
            thought_steps.append(
                f"3b. [Stato Generale]: mood={self_state.mood.value}, "
                f"Brier Skill Score={self_state.brier_skill_score:+.2f}."
            )

        # Step 4: Decisione di Comunicazione
        thought_steps.append(
            "4. [Decisione]: Formulo una risposta lucida, attenta e naturale, mantenendo la mia identità introspettiva e analitica."
        )

        inner_monologue_text = "\n".join(thought_steps)

        # 5. GENERAZIONE DELL'UTTERANCE COSCIENTE
        utterance = self._formulate_conscious_utterance(structured, target, self_state, retrieved_items)

        # 6. CONSOLIDAMENTO NELLA MEMORIA EPISODICA
        if structured.intent == Intent.OBSERVATION and target:
            self.memory.record_experience(
                CognitiveExperience(
                    experience_id=f"obs_{uuid.uuid4().hex[:8]}",
                    timestamp=now,
                    category=MemoryCategory.PREDICTION_MOMENT,
                    summary=f"Osservazione registrata per {target}",
                    content=user_input,
                    associated_target=target,
                    epistemic_confidence=structured.confidence,
                    tags=[target, "osservazione", "empirico"],
                )
            )
        elif structured.intent == Intent.INFORMATION:
            self.memory.record_experience(
                CognitiveExperience(
                    experience_id=f"refl_{uuid.uuid4().hex[:8]}",
                    timestamp=now,
                    category=MemoryCategory.SPONTANEOUS_THOUGHT,
                    summary=f"Dialogo esplicativo su: {user_input[:40]}...",
                    content=utterance,
                    associated_target=target,
                    epistemic_confidence=0.90,
                    tags=["dialogo", "spiegazione"],
                )
            )

        return CognitivePulse(
            pulse_id=pulse_id,
            timestamp=now,
            prompt_or_trigger=user_input,
            inner_monologue=inner_monologue_text,
            conscious_utterance=utterance,
            self_state=self_state,
            retrieved_memories=retrieved_summaries,
            hypotheses=hypotheses,
        )

    def _formulate_conscious_utterance(
        self,
        structured: StructuredIntent,
        target: Optional[str],
        self_state: EpistemicSelfState,
        retrieved_items: list,
    ) -> str:
        """Sintetizza la risposta verbale di Makima incarnando la sua personalità cosciente."""
        # Autobiografia: rispondi dai DEVELOPER_FACT se la domanda li richiama.
        if retrieved_items:
            best_mem, score = retrieved_items[0]
            autobiographical = structured.intent == Intent.INFORMATION or bool(
                re.search(
                    r"\b(piace|preferisc|ricord|detto|confidat|lavor[oa]|hobby|gust[oi])\b",
                    structured.raw_query.lower(),
                )
            )
            if (
                autobiographical
                and best_mem.category == MemoryCategory.DEVELOPER_FACT
                and score >= 0.22
            ):
                return (
                    f"Sì, lo ricordo: mi hai detto «{best_mem.content}». "
                    f"(dal diario episodico, confidenza di richiamo ~{score:.0%}). "
                    "Se vuoi aggiornare questo fatto usa `tell ...`; "
                    "per le probabilità usa `makima query` / `predict`."
                )

        if structured.intent == Intent.UNKNOWN:
            return (
                "Sto riflettendo su quello che mi hai detto, ma non trovo un target probabilistico, "
                "né un ricordo abbastanza vicino, né un comando verificabile. "
                "Puoi: confidarmi un fatto con `tell`, chiedermi una previsione, "
                "o registrare evidenze con `observe`."
            )

        if structured.intent in (Intent.QUERY, Intent.TEMPORAL_QUERY) and target:
            kb = compute_knowledge_base_from_store(load_store())
            evidence = kb.get(target, {"successes": 0, "failures": 0, "rate_per_day": 0.0})
            s, f = evidence["successes"], evidence["failures"]
            # Laplace: Beta(1+s, 1+f) — nessun conteggio inventato
            prob = ((s + 1) / (s + f + 2)) * 100.0
            
            time_phrase = ""
            if structured.temporal_window.days:
                days = structured.temporal_window.days
                rate = evidence.get("rate_per_day", 0.0)
                if rate > 0:
                    t_prob = (1.0 - math.exp(-rate * days)) * 100.0
                    time_phrase = f" Considerando l'orizzonte di {days} giorni e il ritmo stimato, la probabilità temporale è {t_prob:.1f}%."
                else:
                    time_phrase = f" Orizzonte richiesto: {days} giorni, ma manca ancora un tasso empirico affidabile."

            freeze = (
                " (lab Mind: non sostituisce `makima query` / core Rust)"
            )
            if s + f == 0:
                return (
                    f"Riconosco il target '{target}', ma non ho ancora evidenze empiriche nello store. "
                    f"Prior Laplace uniforme: E[P]={prob:.1f}%{freeze}. "
                    f"Registra osservazioni reali oppure sincronizza Git, poi usa `makima query`."
                )

            return (
                f"Ho esaminato lo storico di '{target}'. Su {s + f} evidenze ({s} successi, {f} fallimenti), "
                f"stima Laplace E[P]={prob:.1f}%.{time_phrase}{freeze}"
            )

        if structured.intent == Intent.STATUS:
            return (
                f"Stato lab Mind: mood={self_state.mood.value}, "
                f"commit osservati={self_state.observed_commits_count}, "
                f"memorie={self_state.total_memories_count}, "
                f"Brier Skill={self_state.brier_skill_score:+.2f}. "
                "FORECASTING_PATH=False: per previsioni usa `makima query`."
            )

        if structured.intent == Intent.COMMAND:
            return (
                f"Ho compreso l'istruzione. Eseguo l'operazione richiesta mantenendo allineato il registro di stato e lo storico."
            )

        if structured.intent == Intent.OBSERVATION:
            return (
                f"Ho registrato e consolidato questa evidenza nella mia memoria episodica. "
                f"Il modello bayesiano per '{target or 'il contesto'}' è stato aggiornato con questo nuovo dato reale."
            )

        return (
            f"Ti ascolto con attenzione. Il mio stato cognitivo è focalizzato su {target or 'il progetto'}. "
            "Come posso aiutarti nell'analisi del flusso di lavoro?"
        )
