"""Estrazione e parsing deterministico delle relazioni temporali."""

import re
from makima_lab.nlp.models import TemporalRelation, TemporalWindow

MONTHS = [
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]


def extract_temporal_window(text: str) -> TemporalWindow:
    """Estrae una finestra temporale strutturata da una stringa normalizzata."""
    normalized = text.lower().strip()

    # 1. Riconoscimento "entro N giorni / in N giorni"
    days_match = re.search(r"\b(?:entro|in|tra)\s+(\d+)\s*(?:giorni|giorno|gg|days?)\b", normalized)
    if days_match:
        days = int(days_match.group(1))
        return TemporalWindow(relation=TemporalRelation.WITHIN_DAYS, days=days)

    # 2. Riconoscimento "questa settimana / in settimana"
    if re.search(r"\b(?:questa\s+settimana|in\s+settimana|this\s+week)\b", normalized):
        return TemporalWindow(relation=TemporalRelation.THIS_WEEK)

    # 3. Riconoscimento "questo mese / in questo mese"
    if re.search(r"\b(?:questo\s+mese|this\s+month)\b", normalized):
        return TemporalWindow(relation=TemporalRelation.THIS_MONTH)

    # 4. Riconoscimento "prossimo / next"
    if re.search(r"\b(?:prossim[oaie]|next)\b", normalized):
        return TemporalWindow(relation=TemporalRelation.NEXT)

    # 5. Riconoscimento "entro [mese / data ISO]"
    # Esempio: "entro dicembre", "entro 2026-12-31"
    months_pattern = "|".join(MONTHS)
    before_month_match = re.search(rf"\b(?:entro|prima\s+di|before)\s+({months_pattern})\b", normalized)
    if before_month_match:
        boundary = before_month_match.group(1)
        return TemporalWindow(relation=TemporalRelation.BEFORE, boundary=boundary)

    # Data ISO (YYYY-MM-DD o DD/MM/YYYY)
    date_match = re.search(r"\b(?:entro|prima\s+di|before)\s+(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b", normalized)
    if date_match:
        return TemporalWindow(relation=TemporalRelation.BEFORE, boundary=date_match.group(1))

    return TemporalWindow(relation=TemporalRelation.UNSPECIFIED)
