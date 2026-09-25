"""
Tokenizer and vocabulary dictionary for the Makima neural mind network.
Supports robust offline tokenization for Italian and English natural language.
"""

from __future__ import annotations
import re
from typing import List, Dict

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

# Seed vocabulary containing frequent domain terms, keywords, and Italian/English affixes
BASE_VOCAB = [
    PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN,
    # Question words & intents
    "quanto", "quando", "quale", "cosa", "come", "perché", "perche", "chi", "se",
    "will", "when", "what", "how", "why", "who", "if", "is", "are", "forecast", "predict",
    # Verbs and actions
    "pioverà", "piovera", "piove", "accadrà", "accadra", "succederà", "succedera",
    "completare", "finito", "vincere", "vincerà", "perdere", "salire", "scendere",
    "rain", "happen", "complete", "finish", "win", "lose", "rise", "drop",
    # Modifiers & probabilistic cues
    "probabile", "probabilità", "probabilita", "impossibile", "certo", "sicuro", "forse",
    "likely", "probability", "impossible", "certain", "sure", "maybe", "perhaps",
    "molto", "poco", "abbastanza", "spesso", "sempre", "mai", "oggi", "domani", "ieri",
    "very", "little", "often", "always", "never", "today", "tomorrow", "yesterday",
    # Domain & personal targets
    "meteo", "pioggia", "sole", "neve", "lavoro", "task", "studio", "esame", "progetto",
    "weather", "rain", "sun", "snow", "work", "study", "exam", "project", "deploy",
    "palestra", "corsa", "allenamento", "salute", "sonno", "abitudine", "routine",
    "gym", "run", "workout", "health", "sleep", "habit",
    "utente", "io", "mi", "me", "mio", "mia", "sono", "sento", "penso", "credo",
    "user", "i", "me", "my", "am", "feel", "think", "believe",
]


class MakimaTokenizer:
    """Offline deterministic subword/word tokenizer with fixed vocab indexing."""

    def __init__(self, vocab_list: List[str] | None = None, max_vocab_size: int = 1024) -> None:
        self.max_vocab_size = max_vocab_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

        initial_words = vocab_list if vocab_list is not None else BASE_VOCAB
        # First ensure special tokens are registered exactly
        for spec in (PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN):
            self._add_word(spec)
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
            idx = self.word2idx.get(tok, self.unk_id)
            token_ids.append(idx)
            if len(token_ids) >= max_length - (1 if add_special_tokens else 0):
                break

        if add_special_tokens and EOS_TOKEN in self.word2idx:
            token_ids.append(self.word2idx[EOS_TOKEN])

        # Pad sequence to max_length
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
