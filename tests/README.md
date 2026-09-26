# Makima Test Suite

Questa directory contiene la suite di test per il laboratorio Python e le verifiche cross-language di Makima.

---

## 🧪 Composizione della Test Suite

### 1. Test Unitari e di Integrazione Rust (25 test)
I test del core risiedono in `crates/makima-core/src/`:
- `prob/`: verifica proprietà assiomatiche per Bernoulli, Beta, Poisson e tipi di probabilità clampati.
- `eval.rs`: scoring rules, Brier Score, Brier Skill Score, Log Loss, affidabilità ed ECE.
- `storage.rs`: serializzazione JSON, database SQLite WAL in-memory e sincronizzazione.
- `forecast.rs`: generazione delle stime e riduzione dell'incertezza con osservazioni.

Comando per eseguirli:
```bash
cargo test --all
```

### 2. Test Suite Scientifica Python (45 test)
I file di test presenti in questa cartella verificano:
- **`test_makima_lab.py`**: distribuzioni, momenti, entropia e metriche di valutazione base.
- **`test_nlp.py`**: pipeline NLP, parsing di intenti, estrazione di orizzonti temporali e rejection di chitchat.
- **`test_embeddings.py`**: estrazione vettori densi 384d, normalizzazione e similarità semantica coseno.
- **`test_neural.py`**: tokenizer, modello PyTorch `MakimaMindNet`, online backpropagation e memoria utente.
- **`test_git_observer.py`**: categorizzazione dei commit, estrazione cronologia e sincronizzazione SQLite.
- **`test_llm.py`**: contratto inferenza Qwen 2.5 SLM locale (`explain`, `digest`, `chat`) e fallback deterministico.
- **`test_storage.py`**: calcolo knowledge base da store JSON e gestione percorsi.
- **`test_benchmark.py`**: modelli baseline, proper scoring rules, reliability diagrams e studio di ablazione.

Comando per eseguirli:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## ⚡ Suite di Controllo Qualità Unificata

Per eseguire la verifica completa (formattazione, clippy, test Rust, CLI diagnostica e test Python):

```powershell
# Windows PowerShell
powershell -ExecutionPolicy Bypass -File scripts/check.ps1
```

```bash
# Linux / macOS
./scripts/check.sh
```

