"""
Git Observer and Telemetry Module for Makima.
Extracts real commit telemetry, updates Bayesian targets in SQLite, and drives neural continuous learning.
"""

from __future__ import annotations
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from makima_lab.storage import get_default_db_path, record_journal_entry
from makima_lab.neural import get_neural_engine


@dataclass
class GitCommit:
    commit_hash: str
    author: str
    email: str
    timestamp_sec: int
    subject: str
    category: str  # feature, bugfix, refactor, docs, test, chore


def categorize_commit(subject: str) -> str:
    s = subject.lower()
    if any(k in s for k in ("fix", "bug", "corregge", "risolve", "patch")):
        return "bugfix"
    elif any(k in s for k in ("feat", "implementa", "aggiunge", "integra", "nuov", "add")):
        return "feature"
    elif any(k in s for k in ("test", "verifica", "check", "assert")):
        return "test"
    elif any(k in s for k in ("doc", "docs", "readme", "architecture", "guida")):
        return "docs"
    elif any(k in s for k in ("refactor", "ottimizza", "clean", "migliora")):
        return "refactor"
    return "chore"


class GitObserver:
    """Monitors real Git repository activity and translates it into probabilistic events."""

    def __init__(self, repo_path: str = ".") -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.last_seen_hash: Optional[str] = None

    def fetch_recent_commits(self, max_count: int = 100) -> List[GitCommit]:
        """Fetches commit history from git log."""
        cmd = [
            "git", "log", f"-n{max_count}",
            "--pretty=format:%H|%an|%ae|%at|%s",
        ]
        try:
            res = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
        except Exception:
            return []

        commits: List[GitCommit] = []
        for line in res.stdout.strip().splitlines():
            if not line or "|" not in line:
                continue
            parts = line.split("|", maxsplit=4)
            if len(parts) == 5:
                chash, author, email, ts_str, subject = parts
                try:
                    ts = int(ts_str)
                except ValueError:
                    ts = int(time.time())
                cat = categorize_commit(subject)
                commits.append(
                    GitCommit(
                        commit_hash=chash,
                        author=author,
                        email=email,
                        timestamp_sec=ts,
                        subject=subject,
                        category=cat,
                    )
                )
        return commits

    def sync_history(self, max_count: int = 100) -> Dict[str, Any]:
        """Scans the git repository history and ingests real evidence into Makima's SQLite database and neural net."""
        import sqlite3
        commits = self.fetch_recent_commits(max_count)
        if not commits:
            return {"status": "empty", "synced_commits": 0}

        db_path = get_default_db_path()
        conn = sqlite3.connect(db_path)
        engine = get_neural_engine()

        synced_count = 0
        categories_count: Dict[str, int] = {}

        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp_sec INTEGER NOT NULL,
                    notes TEXT
                );
                """
            )

            # Ingest oldest first for chronological evolution
            for c in reversed(commits):
                categories_count[c.category] = categories_count.get(c.category, 0) + 1

                # 1. Target: git_feature_ratio (1.0 = feature, 0.0 = bugfix)
                if c.category in ("feature", "bugfix"):
                    val = 1.0 if c.category == "feature" else 0.0
                    conn.execute(
                        "INSERT INTO observations (target, value, timestamp_sec, notes) VALUES (?, ?, ?, ?)",
                        ("git:feature_ratio", val, c.timestamp_sec, f"{c.commit_hash[:7]}: {c.subject}"),
                    )

                # 2. Target: git_test_discipline (1.0 = test commit, 0.0 = other)
                test_val = 1.0 if c.category == "test" else 0.0
                conn.execute(
                    "INSERT INTO observations (target, value, timestamp_sec, notes) VALUES (?, ?, ?, ?)",
                    ("git:test_discipline", test_val, c.timestamp_sec, f"{c.commit_hash[:7]}: {c.subject}"),
                )

                # 3. Feed neural mind with real commit message
                engine.learn_step(
                    text=c.subject,
                    intent_label="outcome" if c.category in ("feature", "bugfix") else "routine",
                    polarity_label=0.8 if c.category == "feature" else 0.5,
                    auto_save=False,
                )

                synced_count += 1

            conn.commit()
            engine.save_brain()
            self.last_seen_hash = commits[0].commit_hash if commits else None

        finally:
            conn.close()

        # Compute arrival rate lambda for commits per day
        timestamps = [c.timestamp_sec for c in commits]
        span_days = max(1.0, (max(timestamps) - min(timestamps)) / 86400.0) if len(timestamps) > 1 else 1.0
        rate_per_day = len(commits) / span_days

        return {
            "status": "success",
            "synced_commits": synced_count,
            "categories": categories_count,
            "span_days": span_days,
            "commit_rate_per_day": rate_per_day,
            "latest_commit": commits[0].commit_hash[:7] if commits else None,
            "memory_norm": float(engine.user_memory.norm().item()),
        }

    def poll_new_commits(self) -> List[GitCommit]:
        """Returns only new commits created since last poll."""
        commits = self.fetch_recent_commits(20)
        if not commits:
            return []

        if self.last_seen_hash is None:
            self.last_seen_hash = commits[0].commit_hash
            return []

        new_commits: List[GitCommit] = []
        for c in commits:
            if c.commit_hash == self.last_seen_hash:
                break
            new_commits.append(c)

        if commits:
            self.last_seen_hash = commits[0].commit_hash

        return new_commits


def run_daemon_loop(repo_path: str = ".", poll_interval: int = 15) -> None:
    """Runs Makima's live background daemon listening to Git repository telemetry."""
    print("===================================================")
    print("           MAKIMA REAL TELEMETRY DAEMON            ")
    print("===================================================")
    print(f"Directory monitorata: {os.path.abspath(repo_path)}")
    print(f"Intervallo polling:   {poll_interval} secondi")
    print("In ascolto di commit Git reali & aggiornamenti stato...")
    print("Premi Ctrl+C per arrestare il daemon.")
    print("===================================================\n")

    observer = GitObserver(repo_path)
    initial_sync = observer.sync_history(50)
    print(f"[OK] Sincronizzazione iniziale completata: {initial_sync['synced_commits']} commit elaborati.")
    print(f"-> Frequenza empirica Poisson: {initial_sync['commit_rate_per_day']:.2f} commit/giorno")
    print(f"-> Suddivisione categorie:     {initial_sync['categories']}")
    print(f"-> Memoria latente neurale:    {initial_sync['memory_norm']:.4f}\n")

    engine = get_neural_engine()

    try:
        while True:
            time.sleep(poll_interval)
            new_commits = observer.poll_new_commits()
            if new_commits:
                print(f"\n[!] Rilevati {len(new_commits)} nuovi commit Git in tempo reale:")
                for c in reversed(new_commits):
                    perception = engine.perceive(c.subject, update_memory=True)
                    loss = engine.learn_step(
                        text=c.subject,
                        intent_label="outcome",
                        polarity_label=0.8 if c.category == "feature" else 0.4,
                        auto_save=True,
                    )
                    record_journal_entry(
                        content=f"Git Commit [{c.commit_hash[:7]}]: {c.subject}",
                        target=f"git:{c.category}",
                        intent="outcome",
                        tags=["git", c.category],
                        metadata={"commit_hash": c.commit_hash, "author": c.author},
                    )
                    print(f"   * [{c.commit_hash[:7]}] ({c.category.upper()}) {c.subject}")
                    print(f"     -> Neural Intent: {perception.intent.upper()} (conf {perception.intent_confidence:.1%}) | Loss: {loss:.4f}")
                print(f"-> Stato memoria utente aggiornato (L2: {engine.user_memory.norm().item():.4f})\n")
    except KeyboardInterrupt:
        print("\n[Arresto Daemon] Makima daemon terminato con successo.")
