"""Gestione della persistenza e caricamento dati condivisi (.makima/store.json)."""

import json
from pathlib import Path
from typing import Any


DEFAULT_STORE_RELATIVE_PATH = Path(".makima") / "store.json"


def get_default_store_path() -> Path:
    """Restituisce il percorso assoluto o relativo al workspace del file di storage."""
    # Controlla la cartella corrente o sale di un livello se eseguito da python/ o tests/
    cwd = Path.cwd()
    candidate = cwd / DEFAULT_STORE_RELATIVE_PATH
    if candidate.exists():
        return candidate
    
    parent_candidate = cwd.parent / DEFAULT_STORE_RELATIVE_PATH
    if parent_candidate.exists():
        return parent_candidate
    
    return candidate


def load_store(path: Path | str | None = None) -> dict[str, Any]:
    """Carica lo store JSON da disco, gestendo i fallimenti in modo resiliente."""
    store_path = Path(path) if path else get_default_store_path()
    if not store_path.exists():
        return {
            "version": "0.1.0",
            "observations": [],
            "outcomes": [],
        }

    try:
        with open(store_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "version": "0.1.0",
            "observations": [],
            "outcomes": [],
        }


def compute_knowledge_base_from_store(store_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Estrae un dizionario di evidenze aggregate per ciascun target presente nello store."""
    observations = store_data.get("observations", [])
    kb: dict[str, dict[str, Any]] = {}

    for obs in observations:
        target = obs.get("target")
        val = float(obs.get("value", 0.0))
        ts = int(obs.get("timestamp_sec", 0))

        if not target:
            continue

        if target not in kb:
            kb[target] = {
                "successes": 0,
                "failures": 0,
                "timestamps": [],
                "historical_days": 30,
                "rate_per_day": 0.1,
            }

        if val >= 0.5:
            kb[target]["successes"] += 1
        else:
            kb[target]["failures"] += 1

        if ts > 0:
            kb[target]["timestamps"].append(ts)

    # Calcola il rate temporale empirico basato sull'intervallo temporale
    for target, stats in kb.items():
        ts_list = stats["timestamps"]
        if len(ts_list) >= 2:
            min_ts = min(ts_list)
            max_ts = max(ts_list)
            span_days = max(1.0, (max_ts - min_ts) / 86400.0)
            stats["historical_days"] = span_days
            stats["rate_per_day"] = len(ts_list) / span_days
        else:
            total_obs = stats["successes"] + stats["failures"]
            stats["rate_per_day"] = max(0.05, total_obs / 30.0)

    return kb
