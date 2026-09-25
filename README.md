# Makima

<p align="center">
  <img src="https://giffiles.alphacoders.com/222/222812.gif" alt="Makima" width="480" />
</p>

> **An interpretable probabilistic forecasting system with cognitive neural memory and real-world telemetry.**

Makima è un sistema di previsione probabilistica, quantificazione dell'incertezza e apprendimento continuo, ispirato concettualmente alla *Laplace Mail* di *Shin Megami Tensei: Devil Survivor*.

Makima **non è un chatbot generico** e non produce allucinazioni. Integra un **motore bayesiano esatto in Rust**, un **database relazionale SQLite WAL con Event Sourcing**, una **rete neurale profonda in PyTorch (`MakimaMindNet`) con memoria latente dell'utente**, e un **osservatore di telemetria Git reale** in background.

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
         │  Semantic NLP & Lexicon   │             │   Git Telemetry Observer  │
         │ (Subword Hashing Tokeniz) │             │ (Categories & Poisson Rate│
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

3. **Mente Neurale Cognitiva (`MakimaMindNet`)**:
   - Encoder con Self-Attention e BiGRU per comprensione del testo in italiano ed inglese.
   - **User Latent Memory**: un vettore di stato continuo $\mathbf{h}_{user} \in \mathbb{R}^{64}$ che evolve nel tempo ad ogni interazione.
   - **Online Backpropagation**: la rete apprende in tempo reale con discesa del gradiente (AdamW) su ogni fatto o riflessione comunicata.

4. **Telemetria Git Reale & Daemon**:
   - `makima sync-git`: legge la vera cronologia dei commit Git, categorizza i contributi (`feature`, `bugfix`, `test`, `refactor`, `docs`) e genera target probabilistici reali (`git:feature_ratio`, `git:test_discipline`).
   - `makima daemon`: monitora in background il repository in tempo reale intercettando nuovi commit ed aggiornando lo stato neurale e bayesiano.

5. **Storage SQLite WAL con Event Sourcing**:
   - Database ad alte prestazioni `.makima/makima.db` con `journal_mode = WAL` e tabella immutabile `event_log`.

6. **Scorecard di Valutazione e Calibrazione**:
   - Calcolo del *Brier Score*, *Log Loss*, *Brier Skill Score* rispetto a baseline climatologiche e *Expected Calibration Error* (ECE).

---

## 💻 Guida ai Comandi CLI

```bash
# Avvio rapido tramite launcher grafico/interattivo
.\start.bat

# Interrogazione semantica in linguaggio naturale
cargo run --bin makima -- query "pioverà a Milano entro domani?"

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
- **24/24 Test Python Superati**: distribuzioni, calibrazione ECE, semantic parser, PyTorch `MakimaMindNet`, online learning, `GitObserver`.
- **Zero avvisi** `clippy` e `rustfmt`.

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

---

## 📄 Licenza

Questo progetto è distribuito sotto licenza [MIT](LICENSE).
