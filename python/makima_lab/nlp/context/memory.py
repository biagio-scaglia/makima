"""Modulo di memoria conversazionale delimitata per la risoluzione anaforica del contesto."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional
from makima_lab.nlp.schemas.structured_intent import StructuredIntent


@dataclass
class ConversationTurn:
    """Rappresentazione di un singolo turno conversazionale registrato."""
    raw_query: str
    structured_intent: StructuredIntent
    timestamp_sec: float


class ConversationContext:
    """Buffer FIFO di contesto conversazionale a dimensione limitata (K turni).
    
    Permette la risoluzione esplicita di riferimenti anaforici (es. "e quello precedente?", "quando lo rilascio?").
    """

    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max_turns
        self._turns: deque[ConversationTurn] = deque(maxlen=max_turns)

    def append_turn(self, turn: ConversationTurn) -> None:
        """Aggiunge un turno strutturato nel buffer FIFO."""
        self._turns.append(turn)

    @property
    def last_target(self) -> Optional[str]:
        """Restituisce l'ultimo target discusso nei turni recenti."""
        for turn in reversed(self._turns):
            if turn.structured_intent.target:
                return turn.structured_intent.target
        return None

    @property
    def previous_targets(self) -> List[str]:
        """Restituisce la cronologia ordinata degli ultimi target unici."""
        targets = []
        for turn in reversed(self._turns):
            tgt = turn.structured_intent.target
            if tgt and tgt not in targets:
                targets.append(tgt)
        return targets

    def resolve_anaphora(self, text: str, current_target: Optional[str]) -> Optional[str]:
        """Risolve riferimenti pronominali o anaforici se il target attuale è assente."""
        if current_target:
            return current_target

        normalized = text.lower()

        # Riconoscimento "quello precedente / la versione precedente / prima"
        if "precedent" in normalized or "prima" in normalized:
            targets = self.previous_targets
            if len(targets) >= 2:
                return targets[1]
            if len(targets) == 1:
                return targets[0]

        # Riconoscimento pronomi "lo", "la", "questo", "questa", "dello stesso"
        if any(p in normalized for p in ["quando lo", "quando la", "di questo", "per questo", "dello stesso", "e per"]):
            return self.last_target

        return None

    def clear(self) -> None:
        """Pulisce il buffer di contesto."""
        self._turns.clear()
