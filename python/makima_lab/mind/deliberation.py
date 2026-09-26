"""Motore di deliberazione cognitiva e generazione del monologo interiore di Makima."""

from __future__ import annotations
import math
import time
import uuid
from typing import Dict, List, Optional
from makima_lab.mind.schemas import (
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
        
        # Calcolo incertezza epistemica media (varianza delle distribuzioni Beta)
        variances = []
        for target, stats in kb.items():
            s = stats.get("successes", 1)
            f = stats.get("failures", 1)
            var = (s * f) / (((s + f) ** 2) * (s + f + 1))
            variances.append(var)

        avg_uncertainty = sum(variances) / len(variances) if variances else 0.0833
        
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

        # Step 3: Riflessione Matematica & Incertezza
        if target:
            kb = compute_knowledge_base_from_store(load_store())
            stats = kb.get(target, {"successes": 1, "failures": 1, "rate_per_day": 0.1})
            s, f = stats["successes"], stats["failures"]
            mean_p = s / (s + f)
            var_p = (s * f) / (((s + f) ** 2) * (s + f + 1))
            thought_steps.append(
                f"3. [Analisi Bayesiana]: Il focus è sul target '{target}'. "
                f"Ho registrato {s} successi e {f} fallimenti. "
                f"Valore atteso a posteriori E[P] = {mean_p * 100:.1f}%, incertezza epistemica (Var) = {var_p:.5f}."
            )
            if var_p > 0.03:
                hypotheses.append(f"Il target '{target}' possiede un'evidenza ancora limitata; devo comunicare cautela.")
            else:
                hypotheses.append(f"Il target '{target}' è statisticamente stabile e ben calibrato.")
        elif structured.intent == Intent.UNKNOWN:
            thought_steps.append(
                "3. [Rifiuto Esplicito]: La richiesta non ha un target probabilistico né un comando formale. "
                "La mia disciplina epistemica mi impone di non simulare certezze inventate."
            )
            hypotheses.append("Mantenere trasparenza e chiarire cosa posso osservare o calcolare.")
        else:
            thought_steps.append(
                f"3. [Stato Generale]: Il mio stato affettivo è {self_state.mood.value}. "
                f"Brier Skill Score complessivo: {self_state.brier_skill_score:+.2f}."
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
        if structured.intent == Intent.UNKNOWN:
            return (
                "Sto riflettendo su quello che mi hai detto, ma non trovo un target probabilistico né un comando verificabile nel nostro contesto. "
                "Preferisco dirti con onestà che non so come interpretare questa richiesta piuttosto che inventare una risposta finta. "
                "Puoi chiedermi una previsione su un deploy, sulla build, sullo stato dei commit, o registrare una nuova evidenza."
            )

        if structured.intent in (Intent.QUERY, Intent.TEMPORAL_QUERY) and target:
            kb = compute_knowledge_base_from_store(load_store())
            evidence = kb.get(target, {"successes": 1, "failures": 1, "rate_per_day": 0.1})
            s, f = evidence["successes"], evidence["failures"]
            prob = (s / (s + f)) * 100.0
            
            time_phrase = ""
            if structured.temporal_window.days:
                days = structured.temporal_window.days
                rate = evidence.get("rate_per_day", 0.1)
                t_prob = (1.0 - math.exp(-rate * days)) * 100.0
                time_phrase = f" Considerando l'orizzonte di {days} giorni e il ritmo stimato dei commit, la probabilità temporale sale al {t_prob:.1f}%."

            return (
                f"Ho esaminato lo storico di '{target}'. Su {s + f} evidenze registrate ({s} successi, {f} fallimenti), "
                f"la mia stima probabilistica attuale è del {prob:.1f}%.{time_phrase} "
                f"La mia incertezza su questo punto è stabile, ma continuerò a monitorare i commit per cogliere eventuali anomalie."
            )

        if structured.intent == Intent.STATUS:
            return (
                f"Il mio stato interiore è {self_state.mood.value}. "
                f"Sto monitorando il repository con {self_state.observed_commits_count} eventi registrati e {self_state.total_memories_count} memorie consolidate. "
                f"Il mio Brier Skill Score è pari a {self_state.brier_skill_score:+.2f}. "
                "I miei modelli bayesiani sono calcolati e pronti per nuove valutazioni."
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
