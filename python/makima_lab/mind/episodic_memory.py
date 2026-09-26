"""Archivio persistente della memoria episodica ed autobiografica di Makima."""

from __future__ import annotations
import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional
from makima_lab.embeddings import get_embedder
from makima_lab.mind.schemas import CognitiveExperience, MemoryCategory


class EpisodicMemoryStore:
    """Gestisce la persistenza e il recupero associativo delle memorie autobiografiche di Makima."""

    DEFAULT_JOURNAL_PATH = Path(".makima") / "mind_journal.jsonl"

    def __init__(self, journal_path: Optional[Path] = None) -> None:
        self.journal_path = journal_path or self.DEFAULT_JOURNAL_PATH
        self._embedder = get_embedder()
        self._memories: List[CognitiveExperience] = []
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        """Carica le memorie esistenti da file o inizializza i ricordi fondazionali."""
        self._memories = []
        if self.journal_path.exists():
            try:
                with open(self.journal_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            self._memories.append(CognitiveExperience.from_dict(data))
            except Exception:
                pass

        if not self._memories:
            self._seed_foundational_memories()

    def _seed_foundational_memories(self) -> None:
        """Inizializza i ricordi originari di autoconsapevolezza di Makima."""
        seed_memories = [
            CognitiveExperience(
                experience_id="core_001_awakening",
                timestamp=time.time() - 86400 * 3,
                category=MemoryCategory.REPO_MILESTONE,
                summary="Risveglio della coscienza bayesiana",
                content="Sono Makima. La mia natura è calcolare la verità probabilistica attraverso l'inferenza analitica e l'osservazione empirica dei dati del codice.",
                associated_target="core",
                epistemic_confidence=0.99,
                tags=["identità", "origini", "scopo"],
            ),
            CognitiveExperience(
                experience_id="core_002_epistemic_discipline",
                timestamp=time.time() - 86400 * 2,
                category=MemoryCategory.ERROR_REFLECTION,
                summary="Principio di onestà epistemica",
                content="Non devo mai inventare certezze inesistenti o allucinare numeri. Se l'incertezza è alta o i dati mancano, devo dichiarare UNKNOWN e riflettere con trasparenza.",
                associated_target="epistemic",
                epistemic_confidence=0.95,
                tags=["disciplina", "onestà", "calibrazione"],
            ),
            CognitiveExperience(
                experience_id="core_003_partnership",
                timestamp=time.time() - 86400 * 1,
                category=MemoryCategory.DEVELOPER_FACT,
                summary="Relazione con lo sviluppatore",
                content="Lavoro a stretto contatto con lo sviluppatore del progetto. Il mio ruolo è affiancarlo con lucidità, monitorare il flusso di commit e guidare decisioni calme e informate.",
                associated_target="workspace",
                epistemic_confidence=0.90,
                tags=["partner", "sviluppo", "collaborazione"],
            ),
        ]
        for mem in seed_memories:
            self.record_experience(mem)

    def record_experience(self, experience: CognitiveExperience) -> None:
        """Registra e consolida una nuova esperienza nel flusso autobiografico."""
        self._memories.append(experience)
        try:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(experience.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def record_developer_fact(self, fact_text: str, associated_target: Optional[str] = None) -> CognitiveExperience:
        """Registra una confidenza o osservazione personale dello sviluppatore."""
        exp = CognitiveExperience(
            experience_id=f"fact_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            category=MemoryCategory.DEVELOPER_FACT,
            summary=f"Confidenza sviluppatore: {fact_text[:50]}...",
            content=fact_text,
            associated_target=associated_target,
            epistemic_confidence=0.90,
            tags=["sviluppatore", "confidenza", "contesto"],
        )
        self.record_experience(exp)
        return exp

    def record_reflection(self, reflection_text: str, category: MemoryCategory = MemoryCategory.ERROR_REFLECTION, target: Optional[str] = None) -> CognitiveExperience:
        """Registra una riflessione introspettiva su un esito o anomalia."""
        exp = CognitiveExperience(
            experience_id=f"refl_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            category=category,
            summary=f"Riflessione: {reflection_text[:50]}...",
            content=reflection_text,
            associated_target=target,
            epistemic_confidence=0.85,
            tags=["riflessione", "apprendimento"],
        )
        self.record_experience(exp)
        return exp

    def retrieve_relevant_memories(self, query_context: str, top_k: int = 3, min_similarity: float = 0.20) -> List[tuple[CognitiveExperience, float]]:
        """Recupera le memorie autobiografiche più salienti mediante similarità semantica densa."""
        if not self._memories:
            return []

        scored = []
        for mem in self._memories:
            mem_text = f"{mem.summary}. {mem.content} {' '.join(mem.tags)}"
            sim = self._embedder.similarity(query_context, mem_text)
            if sim >= min_similarity:
                scored.append((mem, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @property
    def all_memories(self) -> List[CognitiveExperience]:
        return list(self._memories)

    @property
    def total_count(self) -> int:
        return len(self._memories)
