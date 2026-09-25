"""Gestione della persistenza e caricamento dati condivisi (.makima/makima.db SQLite e store.json)."""

import json
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_DB_RELATIVE_PATH = Path(".makima") / "makima.db"
DEFAULT_STORE_RELATIVE_PATH = Path(".makima") / "store.json"


def get_default_db_path() -> Path:
    """Restituisce il percorso del database SQLite."""
    cwd = Path.cwd()
    candidate = cwd / DEFAULT_DB_RELATIVE_PATH
    if candidate.exists():
        return candidate
    parent_candidate = cwd.parent / DEFAULT_DB_RELATIVE_PATH
    if parent_candidate.exists():
        return parent_candidate
    return candidate


def get_default_store_path() -> Path:
    """Restituisce il percorso del file JSON di fallback."""
    cwd = Path.cwd()
    candidate = cwd / DEFAULT_STORE_RELATIVE_PATH
    if candidate.exists():
        return candidate
    parent_candidate = cwd.parent / DEFAULT_STORE_RELATIVE_PATH
    if parent_candidate.exists():
        return parent_candidate
    return candidate


def load_store(path: Path | str | None = None) -> dict[str, Any]:
    """Carica i dati dallo storage SQLite (.makima/makima.db) o JSON con fallback trasparente."""
    if path is not None:
        p = Path(path)
        if not p.exists():
            return {
                "version": "0.1.0",
                "observations": [],
                "outcomes": [],
            }
        if str(path).endswith(".db"):
            try:
                conn = sqlite3.connect(p)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("SELECT id, target, value, timestamp_sec FROM observations ORDER BY id ASC")
                obs_rows = cursor.fetchall()
                observations = [
                    {
                        "id": row["id"],
                        "target": row["target"],
                        "value": float(row["value"]),
                        "timestamp_sec": int(row["timestamp_sec"]),
                    }
                    for row in obs_rows
                ]

                cursor.execute("SELECT target, occurred, timestamp_sec FROM outcomes ORDER BY id ASC")
                out_rows = cursor.fetchall()
                outcomes = [
                    {
                        "target": row["target"],
                        "occurred": bool(row["occurred"]),
                        "timestamp_sec": int(row["timestamp_sec"]),
                    }
                    for row in out_rows
                ]

                conn.close()
                return {
                    "version": "0.1.0",
                    "observations": observations,
                    "outcomes": outcomes,
                }
            except Exception:
                return {"version": "0.1.0", "observations": [], "outcomes": []}
        else:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"version": "0.1.0", "observations": [], "outcomes": []}

    # Se path è None, prova prima il database di default SQLite
    db_path = get_default_db_path()
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT id, target, value, timestamp_sec FROM observations ORDER BY id ASC")
            obs_rows = cursor.fetchall()
            observations = [
                {
                    "id": row["id"],
                    "target": row["target"],
                    "value": float(row["value"]),
                    "timestamp_sec": int(row["timestamp_sec"]),
                }
                for row in obs_rows
            ]

            cursor.execute("SELECT target, occurred, timestamp_sec FROM outcomes ORDER BY id ASC")
            out_rows = cursor.fetchall()
            outcomes = [
                {
                    "target": row["target"],
                    "occurred": bool(row["occurred"]),
                    "timestamp_sec": int(row["timestamp_sec"]),
                }
                for row in out_rows
            ]

            conn.close()
            return {
                "version": "0.1.0",
                "observations": observations,
                "outcomes": outcomes,
            }
        except Exception:
            pass

    # Fallback finale sul file JSON di default
    store_path = get_default_store_path()
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


def record_journal_entry(
    content: str,
    target: str | None = None,
    intent: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Salva una riflessione o fatto dell'utente nel database SQLite."""
    import time
    target_db = Path(db_path) if db_path is not None else get_default_db_path()
    target_db.parent.mkdir(parents=True, exist_ok=True)
    extracted_target = target or (tags[0] if tags else None)
    extracted_intent = intent or "journal"
    ts = int(time.time())

    conn = sqlite3.connect(target_db)
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_text TEXT NOT NULL,
                extracted_target TEXT,
                extracted_intent TEXT,
                timestamp_sec INTEGER NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS event_log (
                offset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                target TEXT,
                payload_json TEXT NOT NULL,
                timestamp_sec INTEGER NOT NULL
            );
            """
        )
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO journal_entries (raw_text, extracted_target, extracted_intent, timestamp_sec) VALUES (?, ?, ?, ?)",
            (content, extracted_target, extracted_intent, ts),
        )
        entry_id = cur.lastrowid or 0
        payload = json.dumps({"raw_text": content, "target": extracted_target, "intent": extracted_intent, "metadata": metadata or {}})
        cur.execute(
            "INSERT INTO event_log (topic, target, payload_json, timestamp_sec) VALUES (?, ?, ?, ?)",
            ("journal.entry_added", extracted_target, payload, ts),
        )
        conn.commit()
        return entry_id
    finally:
        conn.close()


