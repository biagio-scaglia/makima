# Makima Scientific Lab (`makima_lab`)

Questo package costituisce l'ambiente di ricerca scientifica, sperimentazione probabilistica e prototipazione NLP per **Makima**.

---

## 🎯 Finalità del Laboratorio

1. **Ricerca e Sperimentazione**: Python viene impiegato come ambiente esplorativo agile per testare modelli probabilistici, distribuzioni a priori, catene di Markov ed euristiche di estrazione temporale.
2. **Validazione Matematica**: ogni nuovo modello teorico viene prima validato empiricamente su dataset storici o sintetici in questo laboratorio prima del porting definitivo in Rust.
3. **Porting verso il Core Rust**: i modelli che dimostrano solidità teorica ed efficacia predittiva vengono implementati nel core Rust ad alte prestazioni (`crates/makima-core`).
4. **Sperimentazione NLP**: classificazione di intenti, riconoscimento di entità temporali e parsing di query in linguaggio naturale.

---

## 🔬 Principi di Sviluppo Scientifico

- **Riproducibilità Rigorosa**: ogni esperimento, benchmark o simulazione deve essere riproducibile (seed deterministici, parametri esplicitati e documentati).
- **Notebook con Scopo Chiaro**: eventuali Jupyter Notebook devono essere motivati da specifiche analisi statistiche o visualizzazioni di calibrazione e non creati per semplice estetica.
- **Isolamento**: il codice sperimentale rimane confinato in questo modulo e non deve creare dipendenze inverse verso la logica di produzione.

---

## 🚀 Setup dell'Ambiente

Per configurare l'ambiente virtuale dedicato senza contaminare l'installazione di sistema:

```bash
# Creazione del virtual environment
python -m venv .venv

# Attivazione (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Installazione in modalità editabile con dipendenze di sviluppo
pip install -e ".[dev,research]"
```

---

## 🧩 Sottosistemi Principali

### 1. Modulo Cognitivo Neurale (`makima_lab.neural`)
Implementa l'architettura profonda **`MakimaMindNet`** in PyTorch:
- **Self-Attention & BiGRU Encoder**: estrazione semantica da testo naturale in italiano/inglese.
- **User Latent Memory State (GRU Cell)**: memoria latente a stato continuo $\mathbf{h}_{user} \in \mathbb{R}^{64}$ che evolve nel tempo.
- **Online Gradient Descent**: apprendimento continuo in tempo reale con ottimizzatore AdamW su ogni interazione dell'utente (`makima tell`).
- **Bayesian Bridge**: stima neurale dei parametri informativi $\text{Beta}(\alpha_0, \beta_0)$ e $\text{Poisson}(\lambda)$ per il calcolo formale senza allucinazioni.

Comandi rapidi:
```bash
python -m makima_lab tell "Oggi ho iniziato un nuovo progetto e sto procedendo spedito"
python -m makima_lab neural "rilasceremo la nuova feature questa settimana?"
python -m makima_lab memory
```

### 2. Modulo NLP Semantico & Vettoriale (`makima_lab.nlp` e `makima_lab.embeddings`)
- **`SemanticQueryParser`**: estrazione di `Intent`, entità temporali (`TemporalWindow`) e target candidato.
- **`SemanticEmbedder`**: embedding densi a 384 dimensioni con Sentence-Transformers (`all-MiniLM-L6-v2`) e fallback su hash subword. Esegue matching a similarità coseno contro i target reali presenti in SQLite.
- **`SemanticForecastPipeline`**: esecuzione end-to-end con calcolo dei parametri posterior e generazione del report formale.

Comandi rapidi:
```bash
python -m makima_lab query "rilasceremo la nuova feature questa settimana?"
python -m makima_lab query "riusciremo a completare i test unitari entro 3 giorni?"
```

### 3. Telemetria Git Reale (`makima_lab.git_observer`)
- Scansione cronologica dei commit dal repository Git locale.
- Categorizzazione semantica bilingue (`feature`, `bugfix`, `test`, `docs`, `refactor`, `chore`).
- Calcolo della frequenza Poisson empirica $\lambda$ e alimentazione automatica del database SQLite WAL.

Comandi rapidi:
```bash
python -m makima_lab sync-git
python -m makima_lab daemon 15
```

### 4. Ragionamento Cognitivo Locale (`makima_lab.llm`)
- Modello SLM locale `Qwen/Qwen2.5-0.5B-Instruct` con caricamento offline prioritario da cache.
- Fornisce spiegazioni trasparenti per i target previsionali (`explain`), bollettini esecutivi (`digest`) e sessione chat interattiva continua (`chat`).

Comandi rapidi:
```bash
python -m makima_lab explain deploy
python -m makima_lab digest
python -m makima_lab chat "come posso interpretare la varianza epistemica?"
python -m makima_lab chat   # Avvia la sessione chat continua
```

### 5. Console Interattiva del Laboratorio
Avviando il modulo senza parametri si accede alla console scientifica Python:
```bash
python -m makima_lab
```


