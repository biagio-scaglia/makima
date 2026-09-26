"""Modulo di validazione dello schema e integrità semantica per il confine Rust Core."""

from typing import List, Tuple
from makima_lab.nlp.schemas.intent import Intent
from makima_lab.nlp.schemas.temporal import TemporalWindow


class IntentValidator:
    """Valida formalmente la coerenza dell'intento strutturato prima dell'invio al core Rust."""

    MIN_CONFIDENCE_THRESHOLD = 0.25

    @classmethod
    def validate(
        cls,
        intent: Intent,
        target: str | None,
        temporal_window: TemporalWindow,
        confidence: float,
    ) -> Tuple[bool, List[str]]:
        """Valida l'intento estratto e restituisce (is_valid, validation_notes)."""
        notes: List[str] = []

        # 1. Rifiuto esplicito intent sconosciuti o fuori dominio
        if intent == Intent.UNKNOWN:
            notes.append("Intento classificato come UNKNOWN o fuori dominio.")
            return False, notes

        # 2. Controllo soglia minima di confidenza
        if confidence < cls.MIN_CONFIDENCE_THRESHOLD:
            notes.append(f"Confidenza ({confidence * 100:.1f}%) inferiore alla soglia minima ({cls.MIN_CONFIDENCE_THRESHOLD * 100:.1f}%).")
            return False, notes

        # 3. Validazione requisiti target per query di previsione
        if intent in (Intent.QUERY, Intent.TEMPORAL_QUERY):
            if target is None or not target.strip():
                notes.append("Nessun target probabilistico identificato per la query.")
                return False, notes

        # 4. Validazione requisiti per osservazioni empiriche
        if intent == Intent.OBSERVATION:
            if target is None or not target.strip():
                notes.append("Target mancante per la registrazione dell'osservazione.")
                return False, notes

        # 5. Intent di stato e informativi sono sempre validi se la confidenza è sufficiente
        if intent in (Intent.STATUS, Intent.INFORMATION, Intent.COMMAND):
            notes.append("Struttura valida per esecuzione.")
            return True, notes

        notes.append("Intento e target conformi ai vincoli di dominio.")
        return True, notes
