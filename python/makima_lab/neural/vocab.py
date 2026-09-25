"""
Tokenizer and extended semantic dictionary for the Makima neural mind network.
Includes a rich multi-domain lexicon (Git, software engineering, productivity, cognitive terms)
and character n-gram fallback for Italian and English.
"""

from __future__ import annotations
import re
from typing import List, Dict

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

# Extended real-world vocabulary dictionary
RICH_VOCAB_LEXICON = [
    PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN,
    # --- Interrogativi & Intenti ---
    "quanto", "quando", "quale", "cosa", "come", "perché", "perche", "chi", "se", "dove",
    "will", "when", "what", "how", "why", "who", "if", "is", "are", "where", "forecast", "predict",
    
    # --- Git & Software Engineering ---
    "git", "commit", "commits", "push", "pull", "merge", "branch", "rebase", "checkout",
    "feature", "feat", "fix", "bug", "bugfix", "refactor", "docs", "documentation", "test", "tests",
    "build", "deploy", "release", "ci", "cd", "pipeline", "code", "repo", "repository",
    "rust", "python", "cargo", "crates", "sqlite", "database", "api", "endpoint", "backend", "frontend",
    "issue", "task", "milestone", "pr", "pull request", "review", "log", "diff", "patch",
    "implementa", "aggiorna", "crea", "corregge", "risolve", "integra", "ottimizza", "aggiunge",
    "implemented", "updated", "created", "fixed", "resolved", "integrated", "optimized", "added",

    # --- Verbi & Azioni di Vita/Lavoro ---
    "pioverà", "piovera", "piove", "accadrà", "accadra", "succederà", "succedera",
    "completare", "finito", "completato", "terminato", "iniziato", "sviluppato", "scritto",
    "vincere", "vincerà", "perdere", "salire", "scendere", "lavorare", "studiare", "allenarsi",
    "rain", "happen", "complete", "finish", "done", "started", "developed", "written",
    "win", "lose", "work", "study", "train", "exercise", "read", "sleep",

    # --- Modificatori & Giudizi Probabilistici ---
    "probabile", "probabilità", "probabilita", "impossibile", "certo", "sicuro", "forse", "quasi",
    "likely", "probability", "impossible", "certain", "sure", "maybe", "perhaps", "almost",
    "molto", "poco", "abbastanza", "spesso", "sempre", "mai", "oggi", "domani", "ieri",
    "very", "little", "often", "always", "never", "today", "tomorrow", "yesterday", "now",
    "settimana", "mese", "anno", "giorno", "notte", "mattina", "sera", "orario", "tempo",
    "week", "month", "year", "day", "night", "morning", "evening", "time", "deadline",

    # --- Domini Personali & Benessere ---
    "meteo", "pioggia", "sole", "neve", "temperatura", "clima",
    "weather", "rain", "sun", "snow", "temperature", "climate",
    "palestra", "corsa", "allenamento", "salute", "sonno", "abitudine", "routine", "peso", "energia",
    "gym", "run", "workout", "health", "sleep", "habit", "energy", "stress", "focus",
    
    # --- Identità Utente & Stati Cognitivi ---
    "utente", "io", "mi", "me", "mio", "mia", "sono", "sento", "penso", "credo", "voglio", "spero",
    "user", "i", "me", "my", "am", "feel", "think", "believe", "want", "hope", "happy", "tired",
    "makima", "daemon", "mind", "memory", "bayes", "laplace", "poisson", "beta", "prior", "posterior",
]


class MakimaTokenizer:
    """Offline deterministic subword/word tokenizer with extended vocabulary and hash fallback."""

    def __init__(self, vocab_list: List[str] | None = None, max_vocab_size: int = 2048) -> None:
        self.max_vocab_size = max_vocab_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

        # Special tokens
        for spec in (PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN):
            self._add_word(spec)

        initial_words = vocab_list if vocab_list is not None else RICH_VOCAB_LEXICON
        for word in initial_words:
            self._add_word(word.lower())

    def _add_word(self, word: str) -> int:
        if word not in self.word2idx and len(self.word2idx) < self.max_vocab_size:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            return idx
        return self.word2idx.get(word, self.word2idx.get(UNK_TOKEN, 1))

    @property
    def pad_id(self) -> int:
        return self.word2idx.get(PAD_TOKEN, 0)

    @property
    def unk_id(self) -> int:
        return self.word2idx.get(UNK_TOKEN, 1)

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s\?!\.,;:'-]", " ", text.lower())
        tokens = re.findall(r"\b\w+\b|[?!]", cleaned)
        return tokens

    def encode(self, text: str, max_length: int = 32, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        token_ids: List[int] = []
        if add_special_tokens and SOS_TOKEN in self.word2idx:
            token_ids.append(self.word2idx[SOS_TOKEN])

        for tok in tokens:
            if tok in self.word2idx:
                idx = self.word2idx[tok]
            else:
                # Deterministic hash bucket to avoid information loss on out-of-vocabulary words
                hash_bucket = (abs(hash(tok)) % (self.max_vocab_size - len(self.word2idx) + 1)) + len(self.word2idx)
                idx = min(hash_bucket, self.max_vocab_size - 1)
            token_ids.append(idx)
            if len(token_ids) >= max_length - (1 if add_special_tokens else 0):
                break

        if add_special_tokens and EOS_TOKEN in self.word2idx:
            token_ids.append(self.word2idx[EOS_TOKEN])

        if len(token_ids) < max_length:
            token_ids.extend([self.pad_id] * (max_length - len(token_ids)))
        else:
            token_ids = token_ids[:max_length]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        words = []
        for tid in token_ids:
            word = self.idx2word.get(tid, UNK_TOKEN)
            if skip_special_tokens and word in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN):
                continue
            words.append(word)
        return " ".join(words)
