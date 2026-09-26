"""Costruzione ed esportazione del grafo di conoscenza (Second Brain) di Makima."""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

from makima_lab.mind.episodic_memory import EpisodicMemoryStore
from makima_lab.mind.schemas import MemoryCategory


@dataclass
class KnowledgeNode:
    """Nodo del Grafo di Conoscenza."""
    id: str
    label: str
    category: str  # "target", "memory", "reflection", "fact", "concept"
    confidence: float
    connections_count: int
    summary: str
    content: str
    tags: List[str]
    weight: float


@dataclass
class KnowledgeEdge:
    """Arco sinaptico di connessione nel Second Brain."""
    source: str
    target: str
    relation: str
    weight: float


@dataclass
class BrainGraph:
    """Rappresentazione globale del Second Brain di Makima."""
    nodes: List[KnowledgeNode]
    edges: List[KnowledgeEdge]
    stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "stats": self.stats,
        }


class SecondBrainBuilder:
    """Costruisce il grafo sinaptico aggregando memorie autobiografiche, target stocastici e concetti ontologici."""

    def __init__(self, journal_path: Optional[Path] = None) -> None:
        self.memory_store = EpisodicMemoryStore(journal_path=journal_path)

    def build_graph(self, target_summaries: Optional[List[Dict[str, Any]]] = None) -> BrainGraph:
        """Costruisce e restituisce il grafo di conoscenza collegando tutte le entità neurali ed empiriche."""
        nodes: Dict[str, KnowledgeNode] = {}
        edges: List[KnowledgeEdge] = []

        # 1. Nodi Concettuali Fondazionali di Ancoraggio
        core_concepts = [
            KnowledgeNode(
                id="concept_bayesian_core",
                label="Inferenza Bayesiana",
                category="concept",
                confidence=1.0,
                connections_count=0,
                summary="Motore probabilistico esatto basato su distribuzioni Beta coniugate e regola di Laplace.",
                content="L'inferenza bayesiana permette a Makima di aggiornare la propria credenza epistemica $P(p)$ all'arrivo di ogni evidenza empirica senza ricorrere a euristiche arbitrarie.",
                tags=["fondamenta", "probabilità", "matematica"],
                weight=1.5,
            ),
            KnowledgeNode(
                id="concept_epistemic_calibration",
                label="Calibrazione Epistemica",
                category="concept",
                confidence=0.98,
                connections_count=0,
                summary="Tracciamento dell'incertezza e calcolo del Brier Score sui Ground Truth.",
                content="L'onestà epistemica impone la quantificazione esatta della varianza e dell'entropia di Shannon su ogni affermazione emessa.",
                tags=["calibrazione", "brier", "verità"],
                weight=1.3,
            ),
            KnowledgeNode(
                id="concept_git_telemetry",
                label="Telemetria Git & Repository",
                category="concept",
                confidence=0.95,
                connections_count=0,
                summary="Flusso continuo di eventi dal version control e dal ciclo di sviluppo.",
                content="I commit, le build e i test forniscono evidenze empiriche per stimare frequenze e tassi di successo operativi.",
                tags=["git", "telemetria", "codice"],
                weight=1.2,
            ),
            KnowledgeNode(
                id="concept_nlp_lab",
                label="Lab NLP Multilivello",
                category="concept",
                confidence=0.96,
                connections_count=0,
                summary="Pipeline neurale a 9 stadi per l'interpretazione del linguaggio e risoluzione anafore.",
                content="Consente a Makima di decodificare query naturali, estrarre target canonici ed eseguire deliberazioni razionali.",
                tags=["nlp", "embeddings", "deliberazione"],
                weight=1.2,
            ),
        ]

        for c in core_concepts:
            nodes[c.id] = c

        # Connetti i concetti fra loro
        edges.append(KnowledgeEdge(source="concept_bayesian_core", target="concept_epistemic_calibration", relation="MATHEMATICAL_FOUNDATION", weight=0.9))
        edges.append(KnowledgeEdge(source="concept_bayesian_core", target="concept_git_telemetry", relation="INFORMS_EMPIRICALLY", weight=0.75))
        edges.append(KnowledgeEdge(source="concept_nlp_lab", target="concept_bayesian_core", relation="DELIBERATES_WITH", weight=0.85))

        # 2. Target Bayesiani Reali (da Rust Core / SQLite)
        if target_summaries:
            for t in target_summaries:
                t_name = t.get("target", "unknown")
                prob = float(t.get("probability", 0.5))
                obs_count = int(t.get("observations_count", 0))
                succ = int(t.get("success_count", 0))
                fail = int(t.get("failure_count", 0))
                variance = float(t.get("uncertainty_variance", 0.05))

                node_id = f"target_{t_name}"
                nodes[node_id] = KnowledgeNode(
                    id=node_id,
                    label=f"🎯 {t_name}",
                    category="target",
                    confidence=prob,
                    connections_count=0,
                    summary=f"Target Stocastico • P={prob*100:.1f}% ({obs_count} evidenze)",
                    content=f"Processo stocastico '{t_name}'. Successi: {succ}, Fallimenti: {fail}. Varianza epistemica: {variance:.4f}. Aggiornato in tempo reale su SQLite WAL.",
                    tags=["target", t_name, "processo_stocastico"],
                    weight=1.0 + min(obs_count * 0.05, 1.0),
                )

                # Connetti target a Git Telemetry o Bayesian Core
                if "git" in t_name or ":" in t_name:
                    edges.append(KnowledgeEdge(source="concept_git_telemetry", target=node_id, relation="MONITORS_STREAM", weight=0.8))
                else:
                    edges.append(KnowledgeEdge(source="concept_bayesian_core", target=node_id, relation="POSTERIOR_DISTRIBUTION", weight=0.8))

        # 3. Memorie Autobiografiche ed Episodiche (da .makima/mind_journal.jsonl)
        memories = self.memory_store.all_memories
        for mem in memories:
            cat_str = "memory"
            if mem.category == MemoryCategory.ERROR_REFLECTION:
                cat_str = "reflection"
            elif mem.category == MemoryCategory.DEVELOPER_FACT:
                cat_str = "fact"

            icon = "🧬" if cat_str == "memory" else ("💡" if cat_str == "reflection" else "📚")
            node_id = f"mem_{mem.experience_id}"
            nodes[node_id] = KnowledgeNode(
                id=node_id,
                label=f"{icon} {mem.summary[:28]}",
                category=cat_str,
                confidence=mem.epistemic_confidence,
                connections_count=0,
                summary=mem.summary,
                content=mem.content,
                tags=mem.tags,
                weight=1.0,
            )

            # Crea archi sinaptici associativi basati sul target collegato o sui tag
            if mem.associated_target:
                tgt_id = f"target_{mem.associated_target}"
                if tgt_id in nodes:
                    edges.append(KnowledgeEdge(source=node_id, target=tgt_id, relation="ASSOCIATED_WITH", weight=0.85))
                elif mem.associated_target == "core":
                    edges.append(KnowledgeEdge(source=node_id, target="concept_bayesian_core", relation="FOUNDATIONAL_ORIGIN", weight=0.95))
                elif mem.associated_target == "epistemic":
                    edges.append(KnowledgeEdge(source=node_id, target="concept_epistemic_calibration", relation="EPISTEMIC_RULE", weight=0.9))
                elif mem.associated_target == "workspace":
                    edges.append(KnowledgeEdge(source=node_id, target="concept_git_telemetry", relation="WORKFLOW_CONTEXT", weight=0.85))

            # Connessioni semantiche aggiuntive per tag
            for tag in mem.tags:
                tag_lower = tag.lower()
                if "identit" in tag_lower or "scopo" in tag_lower:
                    edges.append(KnowledgeEdge(source=node_id, target="concept_bayesian_core", relation="IDENTITY_GROUNDING", weight=0.7))
                elif "svilupp" in tag_lower or "partner" in tag_lower:
                    edges.append(KnowledgeEdge(source=node_id, target="concept_git_telemetry", relation="DEVELOPER_BOND", weight=0.7))

        # Calcola grado delle connessioni
        conn_counter: Dict[str, int] = {k: 0 for k in nodes}
        for e in edges:
            if e.source in conn_counter:
                conn_counter[e.source] += 1
            if e.target in conn_counter:
                conn_counter[e.target] += 1

        for k, count in conn_counter.items():
            if k in nodes:
                nodes[k].connections_count = count

        # Calcola statistiche globali
        node_list = list(nodes.values())
        targets_c = sum(1 for n in node_list if n.category == "target")
        memories_c = sum(1 for n in node_list if n.category == "memory")
        reflections_c = sum(1 for n in node_list if n.category == "reflection")
        facts_c = sum(1 for n in node_list if n.category == "fact")

        avg_conf = sum(n.confidence for n in node_list) / max(len(node_list), 1)
        density = (len(edges) / max(len(node_list), 1))

        stats = {
            "total_nodes": len(node_list),
            "total_edges": len(edges),
            "targets_count": targets_c,
            "memories_count": memories_c,
            "reflections_count": reflections_c,
            "facts_count": facts_c,
            "resonance_score": round(avg_conf * min(density, 2.0) * 50.0, 1),
        }

        return BrainGraph(nodes=node_list, edges=edges, stats=stats)
