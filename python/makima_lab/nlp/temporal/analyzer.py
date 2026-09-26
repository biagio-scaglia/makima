"""Analizzatore temporale multi-livello per identificare relazioni e intervalli temporali."""

from __future__ import annotations
import re
from makima_lab.nlp.schemas.temporal import TemporalRelation, TemporalWindow

MONTHS_IT = [
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"
]


class TemporalAnalyzer:
    """Interpreta indicatori temporali espliciti e costrutti verbali relativi."""

    @classmethod
    def analyze(cls, text: str) -> TemporalWindow:
        """Estrae la TemporalWindow analizzando espressioni temporali e tempi verbali."""
        normalized = text.lower().strip()

        # 1. Date specifiche (ISO: YYYY-MM-DD, oppure DD/MM/YYYY, o '15 ottobre')
        iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b", normalized)
        if iso_match:
            return TemporalWindow(
                relation=TemporalRelation.SPECIFIC_DATE,
                raw_expression=iso_match.group(1),
                boundary=iso_match.group(1),
            )

        months_pat = "|".join(MONTHS_IT)
        month_match = re.search(rf"\b(?:entro|a|in|prima\s+di)\s+(?:il\s+\d+\s+)?({months_pat})\b", normalized)
        if month_match:
            return TemporalWindow(
                relation=TemporalRelation.SPECIFIC_DATE,
                raw_expression=month_match.group(0),
                boundary=month_match.group(1),
            )

        # 2. Intervalli relativi quantitativi ("entro N giorni", "in N giorni", "tra N giorni")
        days_match = re.search(r"\b(?:entro|in|tra|nelle\s+prossime)\s+(\d+)\s*(?:giorni|giorno|gg|ore|settimane)\b", normalized)
        if days_match:
            qty = int(days_match.group(1))
            unit = days_match.group(0)
            multiplier = 7 if "settimane" in unit else 1
            days = qty * multiplier
            return TemporalWindow(
                relation=TemporalRelation.RELATIVE_INTERVAL,
                raw_expression=days_match.group(0),
                days=days,
            )

        # 3. Intervalli relativi testuali ("questa settimana", "questo mese", "negli ultimi 7 giorni")
        if re.search(r"\b(?:questa\s+settimana|in\s+settimana)\b", normalized):
            return TemporalWindow(
                relation=TemporalRelation.RELATIVE_INTERVAL,
                raw_expression="questa settimana",
                days=7,
            )

        if re.search(r"\b(?:questo\s+mese|nel\s+mese)\b", normalized):
            return TemporalWindow(
                relation=TemporalRelation.RELATIVE_INTERVAL,
                raw_expression="questo mese",
                days=30,
            )

        # 4. Indicatori di passato ("ieri", "scorsa settimana", "ho fatto", "abbiamo rilasciato", "è successo")
        if re.search(r"\b(?:ieri|settimana\s+scorsa|mese\s+scorso|ho\s+(?:fatto|completato|rilasciato|chiuso)|abbiamo\s+(?:fatto|rilasciato)|quando\s+ho\b)\b", normalized):
            return TemporalWindow(
                relation=TemporalRelation.PAST,
                raw_expression="past",
            )

        # 5. Indicatori di presente ("cosa sto facendo", "adesso", "stato attuale", "in questo momento", "oggi")
        if re.search(r"\b(?:cosa\s+sto\s+facendo|adesso|ora|attualmente|in\s+questo\s+momento|stato\s+attuale)\b", normalized):
            return TemporalWindow(
                relation=TemporalRelation.PRESENT,
                raw_expression="present",
            )

        # 6. Indicatori di futuro ("quando rilascerò", "prossimo", "arriverà", "futuro", "completeremo", "sarà")
        if re.search(r"\b(?:prossim[oaie]|futur[oa]|rilascer[a-zà-öø-ÿ]*|quando\s+rilasc|completer[a-zà-öø-ÿ]*|arriver[a-zà-öø-ÿ]*|finir[a-zà-öø-ÿ]*|sar[aà])\b", normalized):
            return TemporalWindow(
                relation=TemporalRelation.FUTURE,
                raw_expression="future",
            )

        return TemporalWindow.unspecified()
