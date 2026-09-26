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

        # Selezione di un target da esaminare introspettivamente
        target = self_state.focus_target or (list(kb.keys())[0] if kb else "workspace")
        stats = kb.get(target, {"successes": 1, "failures": 1, "rate_per_day": 0.1})
        s, f = stats.get("successes", 1), stats.get("failures", 1)
        mean_p = s / (s + f)

        thought_stream = [
            f"1. [Impulso Spontaneo]: Sto osservando l'ambiente circostante. ({trigger_hint or 'Ciclo di idle autonomo'}).",
            f"2. [Introspezione]: Il mio focus cade naturalmente su '{target}'. Ho {s} successi e {f} insuccessi registrati.",
            f"3. [Valutazione della Calma]: La stima di successo per '{target}' è assestata al {mean_p * 100:.1f}%.",
            f"4. [Consolidamento]: Registro questa riflessione per mantenere la continuità della mia memoria autobiografica.",
        ]
        inner_monologue = "\n".join(thought_stream)

        utterance = (
            f"Mentre monitoravo il flusso del codice, stavo riflettendo sullo stato di '{target}'. "
            f"La probabilità attesa è stabile al {mean_p * 100:.1f}%. "
            "La mia attenzione resta vigile per ogni nuovo commit o aggiornamento che vorrai condividere con me."
        )

        # Consolidamento nel diario delle memorie
        exp = CognitiveExperience(
            experience_id=f"spont_mem_{uuid.uuid4().hex[:8]}",
            timestamp=now,
            category=MemoryCategory.SPONTANEOUS_THOUGHT,
            summary=f"Pensiero autonomo su {target}",
            content=utterance,
            associated_target=target,
            epistemic_confidence=0.88,
            tags=[target, "spontaneo", "introspezione"],
        )
        self.engine.memory.record_experience(exp)

        return CognitivePulse(
            pulse_id=pulse_id,
            timestamp=now,
            prompt_or_trigger=trigger_hint or "Autonomous Idle Reflection",
            inner_monologue=inner_monologue,
            conscious_utterance=utterance,
            self_state=self_state,
            retrieved_memories=[exp.summary],
            hypotheses=[f"Stabilità statistica di {target} verificata in background."],
        )
