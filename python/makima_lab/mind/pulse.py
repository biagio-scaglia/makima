"""Generatore di impulsi e riflessioni spontanee autonome per la coscienza di Makima."""

from __future__ import annotations
import time
import uuid
from typing import Optional
from makima_lab.mind.schemas import (
    CognitiveExperience,
    CognitiveMood,
    CognitivePulse,
    MemoryCategory,
)
from makima_lab.mind.deliberation import MindDeliberationEngine
from makima_lab.storage import load_store, compute_knowledge_base_from_store


class AutonomousMindPulse:
    """Consente a Makima di formulare pensieri spontanei e riflessioni autonome."""

    def __init__(self, engine: Optional[MindDeliberationEngine] = None) -> None:
        self.engine = engine or MindDeliberationEngine()

    def generate_spontaneous_thought(self, trigger_hint: Optional[str] = None) -> CognitivePulse:
        """Formula un pensiero spontaneo basato sullo stato del repository e sull'evoluzione temporale."""
        pulse_id = f"spont_{uuid.uuid4().hex[:8]}"
        now = time.time()
        self_state = self.engine.get_current_epistemic_state()
        store = load_store()
        kb = compute_knowledge_base_from_store(store)

        # Selezione di un target da esaminare introspettivamente (nessun conteggio inventato).
        target = self_state.focus_target or (list(kb.keys())[0] if kb else None)
        if target is None:
            thought_stream = [
                f"1. [Impulso Spontaneo]: ({trigger_hint or 'Ciclo di idle autonomo'}).",
                "2. [Freeze lab]: Mind non è sul path di forecasting di produzione.",
                "3. [Store vuoto]: nessun target empirico; non invento probabilità.",
                "4. [Consolidamento]: registro solo l'assenza di evidenze.",
            ]
            utterance = (
                "Impulso di laboratorio: lo store non ha ancora target empirici. "
                "Non produco stime inventate — usa `makima observe` / `makima sync-git`, "
                "poi `makima query` per il forecast Rust."
            )
            mean_p = None
            s, f = 0, 0
        else:
            stats = kb.get(target, {"successes": 0, "failures": 0, "rate_per_day": 0.0})
            s, f = int(stats.get("successes", 0)), int(stats.get("failures", 0))
            # Laplace onesto: Beta(1+s, 1+f) — allineato al core
            mean_p = (s + 1) / (s + f + 2)
            evidence_note = (
                f"N={s + f} evidenze ({s} successi, {f} fallimenti)"
                if s + f > 0
                else "N=0 evidenze → prior uniforme Beta(1,1)"
            )
            thought_stream = [
                f"1. [Impulso Spontaneo]: ({trigger_hint or 'Ciclo di idle autonomo'}).",
                "2. [Freeze lab]: questa riflessione è narrativa; i numeri di produzione restano in Rust.",
                f"3. [Laplace store]: focus '{target}' — {evidence_note}; E[P]={mean_p * 100:.1f}%.",
                "4. [Consolidamento]: registro la riflessione senza alterare il core Bayes.",
            ]
            utterance = (
                f"Riflessione di laboratorio su '{target}': {evidence_note}, "
                f"stima Laplace E[P]={mean_p * 100:.1f}%. "
                "Per una previsione di produzione usa `makima query` (core Rust)."
            )
        inner_monologue = "\n".join(thought_stream)

        # Consolidamento nel diario delle memorie
        exp = CognitiveExperience(
            experience_id=f"spont_mem_{uuid.uuid4().hex[:8]}",
            timestamp=now,
            category=MemoryCategory.SPONTANEOUS_THOUGHT,
            summary=f"Pensiero autonomo su {target or 'store-vuoto'}",
            content=utterance,
            associated_target=target,
            epistemic_confidence=0.55 if s + f == 0 else 0.75,
            tags=[target or "empty-store", "spontaneo", "lab-freeze"],
        )
        self.engine.memory.record_experience(exp)

        hypo = (
            "Store vuoto: nessuna stima empirica inventata."
            if target is None
            else f"Lab Laplace su {target}: E[P]={mean_p * 100:.1f}% (non path produzione)."
        )
        return CognitivePulse(
            pulse_id=pulse_id,
            timestamp=now,
            prompt_or_trigger=trigger_hint or "Autonomous Idle Reflection",
            inner_monologue=inner_monologue,
            conscious_utterance=utterance,
            self_state=self_state,
            retrieved_memories=[exp.summary],
            hypotheses=[hypo],
        )
