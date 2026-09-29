"""Motore SLM locale (Qwen 2.5 0.5B) — solo spiegazione ancorata a numeri dati.

Contratto:
- NON calcola probabilità.
- NON estrae StructuredIntent.
- NON inventa evidenze: usa esclusivamente i numeri passati dal chiamante (Rust/store).
- In caso di fallimento modello → fallback deterministico grounded sugli stessi numeri.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", module=".*huggingface_hub.*")

logger = logging.getLogger("makima.llm")

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAKIMA_SYSTEM_PROMPT = (
    "Sei Makima, un assistente di spiegazione probabilistica. "
    "Regole vincolanti:\n"
    "1) Usa SOLO i numeri e i fatti forniti nel messaggio utente.\n"
    "2) NON inventare evidenze, probabilità, Brier score o tassi.\n"
    "3) NON calcolare nuove stime: commenta quelle già fornite.\n"
    "4) Scrivi in italiano, 2-4 frasi, tono analitico e sobrio.\n"
    "5) Se i dati sono insufficienti, dillo esplicitamente."
)


@dataclass(frozen=True)
class LlmResponse:
    """Risposta SLM con provenance esplicita."""

    text: str
    source: str  # "model" | "fallback"
    grounded: bool = True

    def __str__(self) -> str:
        return self.text


class QwenCognitiveEngine:
    """Motore SLM leggero locale per spiegazione e sintesi ancorate ai dati."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self._is_loaded = False
        self.last_source: str = "fallback"

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    @property
    def is_model_ready(self) -> bool:
        return self._is_loaded and self.model is not None and self.tokenizer is not None

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

            logger.info("Caricamento SLM locale (%s) su %s...", self.model_name, self.device)

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                    trust_remote_code=False,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                    dtype=torch.float32 if self.device == "cpu" else torch.float16,
                    trust_remote_code=False,
                ).to(self.device)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=False,
                    trust_remote_code=False,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    local_files_only=False,
                    dtype=torch.float32 if self.device == "cpu" else torch.float16,
                    trust_remote_code=False,
                ).to(self.device)

            self.model.eval()
            self._is_loaded = True
            logger.info("Modello Qwen 2.5 0.5B Instruct caricato.")
            return True
        except Exception as e:
            logger.warning(
                "Impossibile caricare %s: %s. Fallback grounded attivo.",
                self.model_name,
                e,
            )
            return False

    def generate(
        self,
        user_message: str,
        system_prompt: str = MAKIMA_SYSTEM_PROMPT,
        max_new_tokens: int = 220,
        temperature: float = 0.0,
        *,
        fallback_text: Optional[str] = None,
        required_anchors: Optional[List[str]] = None,
    ) -> LlmResponse:
        """Genera testo; se il modello fallisce o non ancora i numeri, usa fallback grounded."""
        deterministic_fallback = fallback_text or self._generic_fallback()

        if not self._is_loaded:
            loaded = self.load_model()
            if not loaded or self.model is None or self.tokenizer is None:
                self.last_source = "fallback"
                return LlmResponse(text=deterministic_fallback, source="fallback", grounded=True)

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

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.eos_token_id,
                "do_sample": temperature > 0.0,
            }
            if temperature > 0.0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_kwargs)

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            text = response.strip()

            if not text or len(text) < 8:
                self.last_source = "fallback"
                return LlmResponse(text=deterministic_fallback, source="fallback", grounded=True)

            if required_anchors and not self._contains_anchors(text, required_anchors):
                # Il modello ha allucinato: anteponiamo il blocco grounded e teniamo il testo come gloss.
                merged = (
                    f"{deterministic_fallback}\n\n"
                    f"[Nota SLM non ancorata — testo modello scartato come fonte numerica]"
                )
                self.last_source = "fallback"
                return LlmResponse(text=merged, source="fallback", grounded=True)

            self.last_source = "model"
            return LlmResponse(text=text, source="model", grounded=True)
        except Exception as e:
            logger.error("Errore durante l'inferenza SLM: %s", e)
            self.last_source = "fallback"
            return LlmResponse(text=deterministic_fallback, source="fallback", grounded=True)

    @staticmethod
    def _contains_anchors(text: str, anchors: List[str]) -> bool:
        """Richiede che almeno metà degli ancore numeriche/testuali compaiano in output."""
        if not anchors:
            return True
        hits = sum(1 for a in anchors if a and a.lower() in text.lower())
        return hits >= max(1, (len(anchors) + 1) // 2)

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
        """Spiegazione ancorata: i numeri sono input obbligatori, non inventati."""
        rate = 0.0 if poisson_rate is None else float(poisson_rate)
        var = (
            (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))
            if variance is None
            else float(variance)
        )
        prob_pct = probability * 100.0
        grounded = self._grounded_explain(
            target=target,
            probability=probability,
            alpha=alpha,
            beta=beta,
            evidence_count=evidence_count,
            poisson_rate=rate,
            variance=var,
        )
        anchors = [
            f"{prob_pct:.1f}",
            f"{alpha:.2f}",
            f"{beta:.2f}",
            str(evidence_count),
            target,
        ]
        prompt = (
            f"Spiega questi dati già calcolati dal core Bayesiano (NON ricalcolare):\n"
            f"- Target: {target}\n"
            f"- E[P] = {prob_pct:.1f}%\n"
            f"- Posterior Beta(α={alpha:.2f}, β={beta:.2f})\n"
            f"- Evidenze empiriche N={evidence_count}\n"
            f"- Varianza epistemica = {var:.6f}\n"
            f"- Tasso Poisson empirico = {rate:.2f} eventi/giorno\n\n"
            f"Scrivi 2-3 frasi in italiano citando questi valori."
        )
        result = self.generate(
            prompt,
            max_new_tokens=160,
            temperature=0.0,
            fallback_text=grounded,
            required_anchors=anchors,
        )
        return str(result)

    def generate_digest(
        self,
        open_targets: List[Dict[str, Any]],
        brier_score: Optional[float] = None,
        ece: Optional[float] = None,
    ) -> str:
        """Digest esecutivo ancorato alla lista target fornita."""
        if not open_targets:
            grounded = (
                "Laplace Digest (grounded): nessun target empirico nello store. "
                "Registra osservazioni con `makima observe` o esegui `makima sync-git`."
            )
            self.last_source = "fallback"
            return grounded

        lines = []
        anchors: List[str] = []
        for t in open_targets:
            name = str(t.get("name", "target"))
            prob = float(t.get("prob", 0.5))
            obs = int(t.get("obs", 0))
            lines.append(f"  * {name}: {prob * 100:.1f}% (oss={obs})")
            anchors.append(name)
            anchors.append(f"{prob * 100:.1f}")

        if brier_score is not None and ece is not None:
            calib = f"Brier={brier_score:.4f}, ECE={ece:.4f}"
            anchors.append(f"{brier_score:.4f}")
        else:
            calib = "calibrazione non disponibile (pochi o nessun ground truth)"

        grounded = self._grounded_digest(open_targets, brier_score, ece)
        prompt = (
            "Bollettino esecutivo basato SOLO su questi dati:\n"
            f"{chr(10).join(lines)}\n"
            f"Calibrazione: {calib}\n\n"
            "Tre punti: stato, target più certi, una raccomandazione operativa. Non inventare numeri."
        )
        result = self.generate(
            prompt,
            max_new_tokens=220,
            temperature=0.0,
            fallback_text=grounded,
            required_anchors=anchors[:6],
        )
        return str(result)

    def chat(self, user_query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Chat ancorata a un contesto esplicito (store/metrics); senza contesto → rifiuto onesto."""
        ctx = context or {}
        grounded = self._grounded_chat(user_query, ctx)
        if not ctx:
            self.last_source = "fallback"
            return grounded

        facts = []
        for k, v in ctx.items():
            facts.append(f"- {k}: {v}")
        prompt = (
            f"Domanda utente: \"{user_query}\"\n"
            f"Contesto verificato:\n{chr(10).join(facts)}\n\n"
            "Rispondi citando solo il contesto. Se non basta, dillo."
        )
        anchors = [str(v) for v in list(ctx.values())[:4]]
        result = self.generate(
            prompt,
            max_new_tokens=180,
            temperature=0.0,
            fallback_text=grounded,
            required_anchors=anchors,
        )
        return str(result)

    def explain_target_detailed(self, **kwargs: Any) -> LlmResponse:
        """Come explain_target ma restituisce LlmResponse con provenance."""
        text = self.explain_target(**kwargs)
        return LlmResponse(text=text, source=self.last_source, grounded=True)

    @staticmethod
    def _grounded_explain(
        target: str,
        probability: float,
        alpha: float,
        beta: float,
        evidence_count: int,
        poisson_rate: float,
        variance: float,
    ) -> str:
        if evidence_count <= 0:
            return (
                f"Makima (fallback grounded): per '{target}' non ci sono evidenze empiriche "
                f"(N=0). Il riferimento è il prior uniforme con E[P]={probability * 100:.1f}% "
                f"da Beta(α={alpha:.2f}, β={beta:.2f}). Registra osservazioni prima di fidarti della stima."
            )
        return (
            f"Makima (fallback grounded): target '{target}' con N={evidence_count} evidenze, "
            f"posterior Beta(α={alpha:.2f}, β={beta:.2f}), E[P]={probability * 100:.1f}%, "
            f"varianza epistemica={variance:.6f}, tasso empirico≈{poisson_rate:.2f}/giorno. "
            f"L'incertezza cala al crescere di N; i numeri provengono dallo store, non dal modello linguistico."
        )

    @staticmethod
    def _grounded_digest(
        open_targets: List[Dict[str, Any]],
        brier_score: Optional[float],
        ece: Optional[float],
    ) -> str:
        ranked = sorted(open_targets, key=lambda t: float(t.get("prob", 0.0)), reverse=True)
        top = ranked[0] if ranked else {"name": "n/d", "prob": 0.0, "obs": 0}
        calib = (
            f"Brier={brier_score:.4f}, ECE={ece:.4f}"
            if brier_score is not None and ece is not None
            else "calibrazione non ancora disponibile"
        )
        listing = ", ".join(
            f"{t.get('name')}={float(t.get('prob', 0)) * 100:.1f}%" for t in ranked[:5]
        )
        return (
            f"Laplace Digest (fallback grounded): monitoro {len(open_targets)} target "
            f"({listing}). Maggiore probabilità corrente: {top.get('name')} "
            f"({float(top.get('prob', 0)) * 100:.1f}%, oss={top.get('obs', 0)}). "
            f"Calibrazione: {calib}. Raccomandazione: continua a registrare outcome reali."
        )

    @staticmethod
    def _grounded_chat(user_query: str, context: Dict[str, Any]) -> str:
        if not context:
            return (
                "Makima (fallback grounded): non ho un contesto empirico allegato a questa chat. "
                f"Per \"{user_query}\" usa `makima query` / `makima predict` oppure fornisci metriche dallo store."
            )
        bits = "; ".join(f"{k}={v}" for k, v in list(context.items())[:8])
        return (
            f"Makima (fallback grounded): rispetto a \"{user_query}\", "
            f"i soli fatti disponibili sono: {bits}."
        )

    @staticmethod
    def _generic_fallback() -> str:
        return (
            "Makima Cognitive Engine (fallback grounded): "
            "senza numeri di input non posso produrre una spiegazione probabilistica. "
            "Passa probabilità/evidenze dallo store o dal core Rust."
        )

    def _fallback_generate(self, prompt: str) -> str:
        """Compatibilità test legacy."""
        # Estrae eventuale target grezzo dal prompt per un messaggio utile.
        m = re.search(r"target['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9_:\-]+)", prompt, re.I)
        target = m.group(1) if m else "target"
        return (
            f"Makima Cognitive Engine (Modalità Analitica Calibrata): "
            f"spiegazione grounded richiesta per '{target}'. "
            "La stima probabilistica deve provenire dall'aggiornamento bayesiano del core Rust; "
            "il modello linguistico non inventa evidenze."
        )


_GLOBAL_ENGINE: Optional[QwenCognitiveEngine] = None


def get_llm_engine() -> QwenCognitiveEngine:
    """Restituisce il singleton globale del motore LLM."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = QwenCognitiveEngine()
    return _GLOBAL_ENGINE
