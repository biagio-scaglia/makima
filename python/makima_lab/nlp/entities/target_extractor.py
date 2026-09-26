"""Modulo per l'estrazione e il matching semantico di entità e target previsionali."""

from __future__ import annotations
import re
from typing import List, Optional, Tuple
from makima_lab.nlp.embeddings.representation import SemanticRepresentation

# Mappa semantica canonica di parole chiave per target noti
TARGET_SEMANTIC_DESCRIPTORS: dict[str, list[str]] = {
    "git:feature_ratio": [
        "feature", "nuova feature", "nuove feature", "funzionalità", "funzionalita",
        "sviluppo feature", "nuovo modulo", "shipping", "features"
    ],
    "git:test_discipline": [
        "test", "unit test", "test suite", "disciplina test", "coverage", "copertura",
        "regressione", "qualità", "qualita", "testing", "e2e"
    ],
    "git:commit_frequency": [
        "commit", "frequenza commit", "attività", "attivita", "push", "cadenza", "ritmo"
    ],
    "git:bugfix_ratio": [
        "bugfix", "bug", "correzione", "fix", "patch", "errori", "risoluzione bug"
    ],
    "git:chore_ratio": [
        "chore", "manutenzione", "configurazione", "pulizia", "refactoring", "ci"
    ],
    "framework_release": [
        "framework", "framework release", "core release", "versione framework", "prossimo framework"
    ],
    "daily_build": [
        "daily build", "build giornaliera", "compilazione", "ci build", "build"
    ],
    "deploy": [
        "deploy", "deployment", "produzione", "rilascio produzione", "rilascio"
    ],
    "api_gateway": [
        "api gateway", "gateway", "endpoint", "api"
    ],
}


class TargetExtractor:
    """Estrae l'entità target e i token chiave combinando regole posizionali e similarità vettoriale."""

    def __init__(self, representation: SemanticRepresentation | None = None) -> None:
        self.repr = representation or SemanticRepresentation()

    def extract(
        self,
        cleaned_text: str,
        available_targets: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], List[str], float, str]:
        """Estrae il target probabilistico migliore e le entità nominali.
        
        Returns:
            (best_target, extracted_entities, score, match_method)
        """
        extracted_entities = self._extract_candidate_entities(cleaned_text)
        targets_pool = list(available_targets) if available_targets else list(TARGET_SEMANTIC_DESCRIPTORS.keys())

        # 1. Corrispondenza diretta esatta (es. "git:feature_ratio" o "deploy")
        for target in targets_pool:
            if target.lower() in cleaned_text:
                return target, extracted_entities, 1.0, "exact_match"

        # 2. Corrispondenza su descrittori lessicali canonici
        for target, descriptors in TARGET_SEMANTIC_DESCRIPTORS.items():
            if available_targets and target not in available_targets:
                continue
            for desc in descriptors:
                if re.search(rf"\b{re.escape(desc)}\b", cleaned_text):
                    return target, extracted_entities, 0.95, "descriptor_match"

        # 3. Estrazione posizionale da verbi di azione / interrogazione
        pos_match = re.search(
            r"\b(?:prevedi|previsione|rilasc[a-zà-öø-ÿ]*|svilupp[a-zà-öø-ÿ]*|pubblic[a-zà-öø-ÿ]*)\s+(?:se|che|il|la|un|una|del|della|di)?\s*([a-z0-9_\-]+)\b",
            cleaned_text,
        )
        if pos_match:
            candidate = pos_match.group(1).strip()
            stop = {"entro", "quando", "se", "che", "prossimo", "questa", "questo", "non", "il", "la", "nuovo"}
            if candidate not in stop:
                if candidate in ("framework", "core"):
                    return "framework_release", extracted_entities, 0.90, "positional_mapping"
                if candidate in ("feature", "features"):
                    return "git:feature_ratio", extracted_entities, 0.90, "positional_mapping"
                if candidate in ("test", "testing"):
                    return "git:test_discipline", extracted_entities, 0.90, "positional_mapping"
                if candidate in targets_pool:
                    return candidate, extracted_entities, 0.85, "positional_extracted"

        # 4. Risoluzione semantica tramite Cosine Similarity su Embeddings
        best_target = None
        best_similarity = -1.0

        for t in targets_pool:
            descriptors = TARGET_SEMANTIC_DESCRIPTORS.get(t, [t])
            target_context = f"{t} {' '.join(descriptors)}"
            sim = self.repr.similarity(cleaned_text, target_context)
            if sim > best_similarity:
                best_similarity = sim
                best_target = t

        # Soglia minima di significatività semantica
        if best_similarity >= 0.42 and best_target is not None:
            return best_target, extracted_entities, round(best_similarity, 4), "semantic_embedding"

        return None, extracted_entities, 0.0, "none"

    def _extract_candidate_entities(self, text: str) -> List[str]:
        """Estrae sostantivi e sintagmi nominali candidati dal testo."""
        entities = []
        # Estrae parole chiave dopo verbi o indicatori di rilascio/test
        matches = re.findall(r"\b(?:il|la|lo|i|gli|le|un|una|sul|sulla|del|della)\s+([a-z0-9_\-]+(?:\s+[a-z0-9_\-]+)?)\b", text)
        for m in matches:
            if len(m.strip()) > 2 and m not in ("prossimo", "prossima", "questa", "questo"):
                entities.append(m.strip())
        return list(dict.fromkeys(entities))  # Deduplicazione preservando l'ordine
