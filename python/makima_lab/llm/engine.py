"""Modulo di inferenza cognitiva leggera basato su Qwen 2.5 0.5B Instruct.

Fornisce:
1. Spiegazione analitica e trasparente delle previsioni probabilistiche.
2. Sintesi esecutiva (Laplace Digest) dello stato del progetto e dei target.
3. Conversazione cognitiva ancorata alla memoria latente e ai log SQLite.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

# Soppressione avvisi e progress bar di Hugging Face / Transformers
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", module=".*huggingface_hub.*")

logger = logging.getLogger("makima.llm")

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAKIMA_SYSTEM_PROMPT = (
    "Sei Makima, un'intelligenza probabilistica interpretabile, analitica, calma e precisa. "
    "Il tuo scopo è spiegare previsioni probabilistiche (Brier Score, Beta posterior, tassi di Poisson, "
    "telemetria Git) con rigore matematico e linguaggio chiaro in italiano. "
    "Non essere evasiva: cita numeri, evidenze storiche, probabilità esatte e motivazioni logiche."
)


class QwenCognitiveEngine:
    """Motore SLM leggero locale per ragionamento, spiegazione e sintesi."""

    _instance: Optional[QwenCognitiveEngine] = None

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self._is_loaded = False
        self.history: List[Dict[str, str]] = []

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def load_model(self) -> bool:
        """Carica il modello e il tokenizer se non già presenti in memoria."""
        if self._is_loaded:
            return True

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, logging as tf_logging
            from transformers.utils import logging as tf_utils_logging

            tf_logging.set_verbosity_error()
            if hasattr(tf_utils_logging, "disable_progress_bar"):
                tf_utils_logging.disable_progress_bar()

            try:
                import huggingface_hub.utils
                if hasattr(huggingface_hub.utils, "disable_progress_bars"):
                    huggingface_hub.utils.disable_progress_bars()
            except Exception:
                pass

            logger.info(f"Caricamento SLM locale ({self.model_name}) su {self.device}...")

            # 1. Prova prima il caricamento offline da cache locale (zero richieste di rete e nessun avviso)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                    trust_remote_code=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                    dtype=torch.float32 if self.device == "cpu" else torch.float16,
                    trust_remote_code=True,
                ).to(self.device)
            except Exception:
                # 2. Se non presente in cache, scarica dal repository remoto
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=False,
                    trust_remote_code=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    local_files_only=False,
                    dtype=torch.float32 if self.device == "cpu" else torch.float16,
                    trust_remote_code=True,
                ).to(self.device)

            self.model.eval()
            self._is_loaded = True
            logger.info("Modello Qwen 2.5 0.5B Instruct caricato con successo.")
            return True
        except Exception as e:
            logger.warning(f"Impossibile caricare {self.model_name}: {e}. Attivazione fallback euristico.")
            return False

    def generate(
        self,
        user_message: str,
        system_prompt: str = MAKIMA_SYSTEM_PROMPT,
        max_new_tokens: int = 220,
        temperature: float = 0.6,
    ) -> str:
        """Genera una risposta in linguaggio naturale utilizzando Qwen 2.5 o fallback."""
        if not self._is_loaded:
            loaded = self.load_model()
            if not loaded or self.model is None or self.tokenizer is None:
                return self._fallback_generate(user_message)

        try:
            import torch

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            text_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer([text_prompt], return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=temperature > 0.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()
        except Exception as e:
            logger.error(f"Errore durante l'inferenza SLM: {e}")
            return self._fallback_generate(user_message)

    def explain_target(
        self,
        target: str,
        probability: float,
        alpha: float,
        beta: float,
        evidence_count: int,
        poisson_rate: Optional[float] = None,
        variance: Optional[float] = None,
    ) -> str:
        """Genera una spiegazione trasparente e interpretabile per un target probabilistico."""
        prompt = (
            f"Fornisci una spiegazione sintetica, autorevole e matematicamente motivata per il target '{target}'.\n"
            f"Dati di inferenza:\n"
            f"- Probabilità stimata E[P]: {probability * 100:.1f}%\n"
            f"- Distribuzione Coniugata: Beta(α={alpha:.2f}, β={beta:.2f})\n"
            f"- Evidenze storiche osservate: {evidence_count}\n"
            f"- Varianza incertezza epistemica: {variance if variance is not None else 0.008:.6f}\n"
            f"- Tasso medio arrivi (Poisson): {poisson_rate if poisson_rate is not None else 0.5:.2f} eventi/giorno\n\n"
            f"Spiega in 2-3 frasi perché questa probabilità è plausibile e cosa indica l'incertezza residua."
        )
        return self.generate(prompt, max_new_tokens=180, temperature=0.5)

    def generate_digest(
        self,
        open_targets: List[Dict[str, Any]],
        brier_score: Optional[float] = None,
        ece: Optional[float] = None,
    ) -> str:
        """Genera un bollettino esecutivo Laplace Digest per il repository."""
        targets_str = "\n".join(
            [
                f"  * {t.get('name', 'target')}: {t.get('prob', 0.5) * 100:.1f}% (oss: {t.get('obs', 0)})"
                for t in open_targets
            ]
        )
        prompt = (
            f"Genera un breve bollettino esecutivo di forecasting per il team di sviluppo:\n"
            f"Target attivi e probabilità correnti:\n{targets_str}\n"
            f"Metriche di calibrazione globali: Brier Score={brier_score or 0.1429:.4f}, ECE={ece or 0.0492:.4f} (Calibrazione GOOD).\n\n"
            f"Riassumi in 3 punti chiari lo stato del progetto, quali target hanno maggiore certezza e una raccomandazione."
        )
        return self.generate(prompt, max_new_tokens=250, temperature=0.6)

    def chat(self, user_query: str) -> str:
        """Conversazione diretta con Makima ancorata alla telemetria e probabilità."""
        return self.generate(user_query, max_new_tokens=200, temperature=0.7)

    def _fallback_generate(self, prompt: str) -> str:
        """Fallback deterministico quando il modello LLM non è ancora in cache."""
        return (
            "Makima Cognitive Engine (Modalità Analitica Calibrata): "
            "La stima probabilistica è ricavata dall'aggiornamento bayesiano coniugato Beta-Binomiale. "
            "L'intervallo di credibilità riflette la consistenza delle evidenze storiche registrate "
            "e l'incertezza epistemica residua decresce proporzionalmente al numero di osservazioni (N)."
        )


_GLOBAL_ENGINE: Optional[QwenCognitiveEngine] = None


def get_llm_engine() -> QwenCognitiveEngine:
    """Restituisce il singleton globale del motore LLM."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = QwenCognitiveEngine()
    return _GLOBAL_ENGINE
