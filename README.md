# Makima

<p align="center">
  <img src="https://giffiles.alphacoders.com/222/222812.gif" alt="Makima" width="480" />
</p>

> **An interpretable probabilistic forecasting system with cognitive neural memory, semantic vector embeddings, and real-world Git telemetry.**

Makima è un sistema di previsione probabilistica, quantificazione dell'incertezza e apprendimento continuo, ispirato concettualmente alla *Laplace Mail* di *Shin Megami Tensei: Devil Survivor*.

Makima **non è un chatbot generico** e non produce allucinazioni. Integra un **motore bayesiano esatto in Rust**, un **database relazionale SQLite WAL con Event Sourcing**, una **rete neurale profonda in PyTorch (`MakimaMindNet`) con memoria latente dell'utente**, un **motore di embedding semantico vettoriale (`SemanticEmbedder`)** per la risoluzione intelligente dei target, e un **osservatore di telemetria Git reale** in background.

---

## 🏛️ Architettura di Sistema

```text
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   MAKIMA ARCHITECTURE                                  │
 └────────────────────────────────────────────────────────────────────────────────────────┘

    [ Input Utente / Linguaggio Naturale ]       [ Telemetria Git Reale / Commits ]
                       │                                         │
                       ▼                                         ▼
         ┌───────────────────────────┐             ┌───────────────────────────┐
         │  SemanticEmbedder (NLP)   │             │   Git Telemetry Observer  │
         │ (SentenceTransformers 384d│             │ (Categorizzazione commit  │
         │  Cosine Target Matching)  │             │  & Stima Tassi Poisson λ) │
         └─────────────┬─────────────┘             └─────────────┬─────────────┘
                       │                                         │
                       ▼                                         ▼
         ┌─────────────────────────────────────────────────────────────────────┐
         │                 MakimaMindNet (PyTorch Cognitive Net)               │
         │  • Multi-Head Self-Attention & Bidirectional GRU Encoder            │
         │  • User Latent State Memory (GRU Cell) ─── h_user ∈ ℝ⁶⁴            │
         │  • Multi-Task Heads: Intent / Target Embedding / Bayesian Bridge   │
         │  • Online Gradient Backpropagation (AdamW Continuous Learning)      │
         └──────────────────────────────────┬──────────────────────────────────┘
                                            │
                                            ▼
         ┌─────────────────────────────────────────────────────────────────────┐
         │           Dual Persistence Layer (SQLite WAL + Event Sourcing)      │
         │  • .makima/makima.db (journal_mode=WAL, event_log, observations)    │
         │  • .makima/store.json (JSON fallback sincronizzato)                 │
         │  • .makima/makima_brain.pt (Pesi e checkpoint memoria neurale)      │
         └──────────────────────────────────┬──────────────────────────────────┘
                                            │
                                            ▼
         ┌─────────────────────────────────────────────────────────────────────┐
         │                 Rust High-Performance Core Engine                   │
         │  • Inferenza Bayesiana Esatta (Beta-Binomial Conjugate Updating)    │
         │  • Processi Temporali di Poisson (Omogenei & Non-Omogenei)         │
         │  • Entropia Informativa di Shannon (bit) & Incertezza Epistemica    │
         │  • Scoring Rules: Brier Score, Brier Skill Score, Log Loss, ECE     │
         │  • Multi-Target Dashboard & Laplace Mail Generator                  │
         └─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Funzionalità Chiave

1. **Inferenza Bayesiana Trasparente**:
   - Distribuzioni coniugate $\text{Beta}(\alpha, \beta)$ aggiornate deterministicamente all'arrivo di nuove evidenze ($P = \frac{\alpha}{\alpha + \beta}$).
   - Distinzione matematica formale tra incertezza aleatoria e incertezza epistemica (varianza residua).

2. **Processi Temporali di Poisson**:
   - Stima dei tassi di frequenza $\lambda$ e calcolo della probabilità di occorrenza entro finestre temporali:
     $$P(T \le t) = 1 - e^{-\lambda t}$$

3. **Risoluzione Semantica Vettoriale (`SemanticEmbedder`)**:
   - Comprensione del linguaggio naturale con `sentence-transformers` (`all-MiniLM-L6-v2`) e fallback deterministico su proiezioni hash subword.
   - Calcolo della similarità coseno a 384 dimensioni per associare frasi spontanee (es. *"rilasceremo la nuova feature questa settimana?"*) ai target reali della repository (`git:feature_ratio`, `git:test_discipline`).

4. **Mente Neurale Cognitiva (`MakimaMindNet`)**:
   - Encoder con Self-Attention e BiGRU per comprensione del testo in italiano ed inglese.
   - **User Latent Memory**: un vettore di stato continuo $\mathbf{h}_{user} \in \mathbb{R}^{64}$ che evolve nel tempo ad ogni interazione (`makima tell`).
   - **Online Backpropagation**: la rete apprende in tempo reale con discesa del gradiente (AdamW) su ogni fatto o riflessione comunicata.

5. **Telemetria Git Reale & Daemon**:
   - `makima sync-git`: analizza l'intera cronologia Git reale, categorizza i commit (`feature`, `bugfix`, `test`, `refactor`, `docs`) e alimenta i target probabilistici reali (`git:feature_ratio`, `git:test_discipline`).
   - `makima daemon`: monitora in background il repository in tempo reale intercettando nuovi commit ed aggiornando lo stato neurale e bayesiano.

6. **Storage SQLite WAL con Event Sourcing**:
   - Database ad alte prestazioni `.makima/makima.db` con `journal_mode = WAL` e tabella immutabile `event_log`.

7. **Scorecard di Valutazione e Calibrazione**:
   - Calcolo del *Brier Score*, *Log Loss*, *Brier Skill Score* rispetto a baseline climatologiche e *Expected Calibration Error* (ECE).

---

## 💻 Guida ai Comandi CLI

```bash
# Avvio istantaneo (<300ms, senza test preventivi e con binario compilato)
.\avvio.bat

# Avvio con verifica preventiva dell'intera suite di test scientifici
.\start.bat

# Interrogazione semantica in linguaggio naturale (con Sentence-Transformers)
cargo run --bin makima -- query "rilasceremo la nuova feature questa settimana?"
cargo run --bin makima -- query "riusciremo a completare i test unitari entro 3 giorni?"

# Confida un fatto a Makima (aggiorna la memoria neurale e SQLite)
cargo run --bin makima -- tell "Oggi ho completato il refactor dell'architettura e mi sento molto soddisfatto"

# Ispezione dello stato cognitivo neurale e della memoria latente
cargo run --bin makima -- neural "svilupperemo la nuova feature questa settimana?"
cargo run --bin makima -- memory

# Sincronizzazione telemetria Git reale
cargo run --bin makima -- sync-git

# Avvio del daemon di monitoraggio in background (polling ogni 15s)
cargo run --bin makima -- daemon 15

# Dashboard di tutti i target monitorati
cargo run --bin makima -- targets

# Generazione del bollettino previsionale Laplace Mail
cargo run --bin makima -- mail

# Registrazione evidenze ed esiti reali
cargo run --bin makima -- observe git:feature_ratio 1
cargo run --bin makima -- outcome git:feature_ratio 1

# Valutazione di accuratezza e calibrazione
cargo run --bin makima -- evaluate
```

---

## 🧪 Verifica e Test Suite

Per eseguire la suite di verifica completa (formattazione, analisi statica Rust, unit test Rust e suite scientifica Python):

```powershell
# Esecuzione script di verifica unificato
powershell -ExecutionPolicy Bypass -File scripts/check.ps1
```

- **25/25 Test Rust Superati**: algebra delle probabilità, inferenza bayesiana, Poisson, valutatore Brier, storage SQLite in-memory.
- **32/32 Test Python Superati**: distribuzioni, calibrazione ECE, semantic embedding & parser, PyTorch `MakimaMindNet`, online learning, `GitObserver`.
- **Totale 57 Test Unitari/Integrazione**: 100% Superati con **Zero avvisi** `clippy` e `rustfmt`.

---

## 🗺️ Roadmap di Sviluppo

- [x] **Fase 1**: Inizializzazione workspace Rust, tooling di qualità (`rustfmt`, `clippy`) e documentazione di base.
- [x] **Fase 2**: Creazione del crate `makima-core` con entità di dominio (`Observation`, `ObservationId`, `MakimaEngine`).
- [x] **Fase 3**: Creazione del crate `makima-cli` con ritratto ASCII di Makima e diagnostica di stato.
- [x] **Fase 4**: Configurazione del laboratorio scientifico Python (`makima_lab`, `pyproject.toml`).
- [x] **Fase 5**: Specifiche architetturali formali (`docs/architecture.md`) e pilastri di interpretabilità.
- [x] **Fase 6**: Fondamenti matematici (`Bernoulli`, `BetaDistribution`, `PoissonDistribution`) ed inferenza bayesiana.
- [x] **Fase 7**: Valutazione statistica (*Proper Scoring Rules*, Brier Score, Log Loss, Brier Skill Score).
- [x] **Fase 8**: NLP Semantic Parsing (`SemanticQueryParser`, `ForecastQuery`, `TemporalWindow`) e Pipeline semantica NL.
- [x] **Fase 9**: Persistenza dello stato su disco ([`.makima/store.json`](.makima/store.json)) condiviso tra Rust e Python.
- [x] **Fase 10**: Calibrazione empirica (*Expected Calibration Error*, *MCE*, *Reliability Diagrams*).
- [x] **Fase 11**: Multi-Target Dashboard (`makima targets`) e monitoraggio parallelo.
- [x] **Fase 12**: Bollettino previsionale automatico (*Laplace Mail Bulletin* `makima mail`).
- [x] **Fase 13**: Persistenza relazionale SQLite WAL con Event Sourcing (`.makima/makima.db`).
- [x] **Fase 14**: Modulo Cognitivo Neurale PyTorch (`MakimaMindNet`, Self-Attention, Memoria Latente Utente, Online Backprop).
- [x] **Fase 15**: Osservatore di Telemetria Git Reale e Modalità Daemon in background (`makima sync-git`, `makima daemon`).
- [x] **Fase 16**: Motore Vettoriale Semantico (`SemanticEmbedder` Sentence-Transformers `all-MiniLM-L6-v2` & Cosine Target Matching).

---

## 📄 Licenza

Questo progetto è distribuito sotto licenza [MIT](LICENSE).
