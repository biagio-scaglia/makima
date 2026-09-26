# Makima — Pipeline, Data Flow & Module Architecture

Questo documento descrive in dettaglio il funzionamento interno di **Makima**: il ciclo di vita dei dati, la responsabilità di ciascun modulo, il meccanismo di cooperazione tra Rust e Python, e l'architettura della pipeline di elaborazione semantica e probabilistica.

---

## 1. Visione d'Insieme del Flusso dei Dati

Makima è progettato attorno al paradigma della **trasparenza probabilistica** ed **Event Sourcing**: qualsiasi stima numerica deve essere riconducibile a evidenze storiche esatte e a distribuzioni coniugate analitiche.

```mermaid
flowchart TD
    subgraph Input ["1. Ingestion Layer"]
        CLI["CLI Commands (observe, outcome, predict)"]
        NL["Query in Linguaggio Naturale"]
        GIT["Telemetria Git Reale (commit log)"]
        USER_FACTS["Confidenze / Riflessioni (tell)"]
    end

    subgraph Processing ["2. Elaborazione e Risoluzione Semantica"]
        PARSER["SemanticQueryParser (Intent & Temporal)"]
        EMB["SemanticEmbedder (MiniLM 384d / Cosine Matching)"]
        MIND["MakimaMindNet (PyTorch Self-Attention & Latent Memory)"]
        GIT_OBS["GitObserver (Categorizzazione & Tasso Poisson)"]
    end

    subgraph Persistence ["3. Dual Persistence Layer (Event Sourcing)"]
        SQLITE[("SQLite WAL Database (.makima/makima.db)")]
        JSON[(".makima/store.json (JSON fallback sincronizzato)")]
        BRAIN[(".makima/makima_brain.pt (Pesi e Memoria Utente)")]
    end

    subgraph CoreEngine ["4. Motore Probabilistico Deterministico (Rust Core)"]
        BETA["Beta-Binomial Conjugate Updating: P = α / (α + β)"]
        POISSON["Processo Temporale Poisson: P(T ≤ t) = 1 - exp(-λt)"]
        UNCERT["Scomposizione Incertezza (Varianza ed Entropia bit)"]
        SCORING["Proper Scoring Rules (Brier, Log Loss, BSS, ECE)"]
        LEDGER["Forecast Lifecycle Ledger (Stato e Risoluzione)"]
    end

    subgraph Output ["5. Presentazione & Spiegazione"]
        REPORT["Forecast Report ASCII / Dashboard Target"]
        MAIL["Bollettino Laplace Mail"]
        SLM["Qwen 2.5 SLM Locale (Spiegazione Analitica & Digest)"]
    end

    CLI -->|Scrive evidenze| SQLITE
    CLI -->|Sincronizza| JSON
    NL --> PARSER
    PARSER --> EMB
    EMB -->|Risoluzione Target| SQLITE
    USER_FACTS --> MIND
    MIND -->|Online Backprop| BRAIN
    MIND -->|Registra Journal| SQLITE
    GIT --> GIT_OBS
    GIT_OBS -->|Osservazioni feature_ratio / test_discipline| SQLITE
    GIT_OBS -->|Addestramento continuo| MIND

    SQLITE -->|Lettura evidenze| CoreEngine
    CoreEngine --> REPORT
    CoreEngine --> MAIL
    CoreEngine --> LEDGER
    REPORT --> SLM
```

---

## 2. Responsabilità dei Moduli

Il repository è ripartito rigorosamente tra runtime compilato di produzione (`crates/`) e laboratorio scientifico/NLP (`python/` ed `experiments/`):

### 2.1 Crate Rust (`crates/`)

| Modulo / Crate | Percorso | Responsabilità Primaria |
| :--- | :--- | :--- |
| **`makima-core`** | `crates/makima-core/` | **Nucleo di dominio e calcolo probabilistico**: definisce entità immutabili (`Observation`, `ObservationId`), distribuzioni analitiche (`Bernoulli`, `BetaDistribution`, `PoissonDistribution`), motore centrale (`MakimaEngine`), valutatore statistico (`Evaluator`, `Scoring`), ledger delle previsioni (`ForecastLedger`) e layer di persistenza nativo (`MakimaStore`, `MakimaDb`). |
| **`makima-cli`** | `crates/makima-cli/` | **Interfaccia da riga di comando ad alte prestazioni**: parsing dei comandi del terminale, visualizzazione diagnostica, rendering del ritratto ASCII di Makima, dispatch dei comandi previdenziali e ponte di esecuzione verso i sottoprocessi Python del laboratorio. |

### 2.2 Moduli Python (`python/makima_lab/`)

| Sottosistema | Percorso | Responsabilità Primaria |
| :--- | :--- | :--- |
| **`makima_lab.nlp`** | `python/makima_lab/nlp/` | **Analisi del linguaggio naturale**: normalizzazione semantica del testo, classificazione degli intenti (`Intent`), estrazione di orizzonti temporali (`TemporalWindow`), risoluzione automatica del target tramite embedding vettoriali e orchestrazione della pipeline previsionale (`SemanticForecastPipeline`). |
| **`makima_lab.embeddings`** | `python/makima_lab/embeddings.py` | **Vettorizzazione semantica densa**: genera embedding a 384 dimensioni mediante Sentence-Transformers (`all-MiniLM-L6-v2`) con fallback deterministico su proiezioni hash subword; calcola la similarità coseno per associare query utente a target reali del database. |
| **`makima_lab.neural`** | `python/makima_lab/neural/` | **Rete Neurale Cognitiva (`MakimaMindNet`)**: architettura PyTorch con BiGRU, Multi-Head Self-Attention, cella di memoria latente ricorrente dell'utente ($\mathbf{h}_{user} \in \mathbb{R}^{64}$), classificazione multi-task e apprendimento online continuo (AdamW). |
| **`makima_lab.git_observer`** | `python/makima_lab/git_observer.py` | **Telemetria Git reale & Daemon**: estrazione dello storico dei commit dal repository, categorizzazione semantica (italiano/inglese), stima empirica della frequenza Poisson ($\lambda$) e monitoraggio in background per nuovi commit. |
| **`makima_lab.llm`** | `python/makima_lab/llm/` | **Ragionamento e spiegazione con Small Language Model (SLM)**: inferenza locale su `Qwen/Qwen2.5-0.5B-Instruct` con caricamento offline prioritario da cache, generazione di spiegazioni trasparenti delle formule (`explain`), sintesi esecutive (`digest`) e sessione chat interattiva continua (`chat`). |
| **`makima_lab.storage`** | `python/makima_lab/storage.py` | **Astrazione storage lato Python**: interfacciamento con `.makima/makima.db` (SQLite WAL) e `.makima/store.json`, calcolo della Knowledge Base e registrazione di voci di diario. |

### 2.3 Modulo di Validazione Sperimentale (`experiments/`)

| Directory | Percorso | Responsabilità Primaria |
| :--- | :--- | :--- |
| **`experiments.forecasting`** | `experiments/forecasting/` | **Suite scientifica comparativa**: generatori di dataset sintetici con Ground Truth nota (`stationary_stream`, `regime_shifts`, `poisson_arrivals`), implementazione di 7 modelli baseline comparativi, motore di calcolo Proper Scoring Rules (Brier, LogLoss, BSS, ECE) e runner di Ablation Study. |

---

## 3. Comunicazione e Sincronizzazione tra Rust e Python

Makima adotta un'architettura **ibrida asincrona/disaccoppiata**:

```text
  ┌─────────────────────────────────────────────────────────────┐
  │                    Spazio Applicativo                       │
  ├──────────────────────────────┬──────────────────────────────┤
  │       Rust Runtime Core      │      Python Research Lab     │
  │      (Prestazioni & ACID)    │    (NLP, PyTorch & SLM)      │
  └──────────────┬───────────────┴──────────────┬───────────────┘
                 │                              │
                 ▼                              ▼
  ┌─────────────────────────────────────────────────────────────┐
  │               SQLite WAL Database (.makima/makima.db)       │
  │   - journal_mode = WAL (Write-Ahead Logging)                │
  │   - Letture concorrenti non bloccanti                       │
  │   - Tabella immutabile: event_log & observations            │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │               JSON Snapshot (.makima/store.json)            │
  │   - Formato human-readable per backup e ispezione manuale   │
  └─────────────────────────────────────────────────────────────┘
```

### 3.1 Canali di Cooperazione

1. **Storage Condiviso SQLite WAL (`.makima/makima.db`)**:
   - Sia Rust (`rusqlite`) sia Python (`sqlite3`) aprono il database con `PRAGMA journal_mode = WAL;`.
   - Il WAL (*Write-Ahead Logging*) consente a processi concorrenti di leggere senza bloccare chi scrive, evitando conflitti di lock tra la CLI Rust e i daemon/script Python.
2. **Snapshot JSON di Fallback (`.makima/store.json`)**:
   - Ad ogni mutazione, `MakimaEngine` sincronizza la rappresentazione di stato su JSON, permettendo al laboratorio Python di leggere lo storico completo anche in assenza di driver relazionali.
3. **Dispatch Subprocess da CLI Rust a Python**:
   - Quando l'utente invoca comandi che richiedono modelli linguistici o neurali (`makima query`, `makima tell`, `makima neural`, `makima sync-git`), `makima-cli` richiama l'interprete Python configurando `PYTHONPATH=python` ed eseguendo il modulo `makima_lab`.
   - Se Python o i moduli opzionali non sono disponibili, la CLI Rust degrada elegantemente informando l'utente senza provocare panic.

---

## 4. Pipeline NLP: Intent, Target e Temporal Window

La comprensione del linguaggio naturale è affidata a [`makima_lab.nlp`](file:///c:/Users/biagio.scaglia/Desktop/makima/python/makima_lab/nlp). Il suo scopo è convertire una frase arbitraria in una struttura fortemente tipizzata: [`ForecastQuery`](file:///c:/Users/biagio.scaglia/Desktop/makima/python/makima_lab/nlp/models.py).

### 4.1 Ciclo di Elaborazione Semantica

```text
Frase Utente: "Riusciremo a completare i test unitari entro 3 giorni?"
                             │
                             ▼
 1. Normalizzazione Lessicale (lowercase, rimozione punteggiatura ridondante)
                             │
                             ▼
 2. Filtro Rifiuto Chitchat  ──► Rifiuta query estetiche/chitchat (es. "quanto è bello?")
                             │
                             ▼
 3. Intent Classification    ──► Intent.RELEASE_PREDICTION / Intent.FORECAST
                             │
                             ▼
 4. Target Extraction        ──► Candidato testuale: "test_unitari"
                             │   Interroga SemanticEmbedder:
                             │   Cosine similarity tra "test unitari" e target SQLite:
                             │   -> Match vincente: "git:test_discipline" (sim: 0.84)
                             │
                             ▼
 5. Temporal Window Parsing  ──► TemporalRelation.WITHIN_DAYS (days = 3)
                             │
                             ▼
 6. Validazione ForecastQuery ──► is_valid_forecast = True
                             │
                             ▼
 7. Esecuzione Calcolo       ──► Legge osservazioni storiche da SQLite (s=12, f=2)
                                 Aggiorna Beta(13.0, 3.0) -> E[P] = 81.25%
                                 Calcola Poisson con λ=0.85/giorno, finestra=3gg
                                 Probabilità temporale = 1 - exp(-0.85 * 3) = 92.2%
```

### 4.2 Tassonomia di Dominio NLP

#### `Intent`
- **`FORECAST`**: Richiesta generica di probabilità su un target noto.
- **`RELEASE_PREDICTION`**: Richiesta legata a rilasci software, deploy, merge o completamento task.
- **`OBSERVATION_RECORD`**: Registrazione di un evento storico.
- **`STATUS_QUERY`**: Domanda sullo stato o la salute del motore.
- **`UNSUPPORTED`**: Domande non probabilistiche, filosofiche o chitchat. Vengono rifiutate per evitare allucinazioni.

#### `TemporalRelation` & `TemporalWindow`
- **`NEXT`**: Prossimo evento atteso (es. *"il prossimo framework"*).
- **`BEFORE`**: Limite temporale superiore assoluto (es. *"entro dicembre"*).
- **`WITHIN_DAYS`**: Finestra temporale parametrica di $N$ giorni (es. *"entro 14 giorni"*).
- **`THIS_WEEK`**: Orizzonte circoscritto alla settimana solare corrente.
- **`THIS_MONTH`**: Orizzonte circoscritto al mese solare corrente.
- **`UNSPECIFIED`**: Orizzonte temporale non delimitato (previsione puramente Bernoulli/Beta).

---

## 5. Ciclo di Vita delle Previsioni (Forecast Lifecycle Ledger)

Ogni previsione formale emessa da Makima attraversa un ciclo di vita controllato e tracciato nel ledger:

1. **Emissione (`Pending`)**:
   - Viene generato un identificativo incrementale (`ForecastId`).
   - Vengono congelati nel ledger: probabilità $E[P]$, finestra temporale, numero di evidenze utilizzate e nome del modello.
2. **Attesa dell'Esito Reale**:
   - La previsione rimane nello stato `PENDING` nel ledger (`makima forecasts`).
3. **Risoluzione con Ground Truth (`Resolved`)**:
   - L'utente registra l'evento effettivo (`makima outcome <target> <1|0>`).
   - Lo stato transita a `RESOLVED` con registrazione dell'esito reale ($1$ o $0$).
4. **Scoring e Calibrazione**:
   - Viene calcolato istantaneamente il Brier Score individuale $(p - y)^2$ e la Logarithmic Loss.
   - La previsione viene assegnata a uno dei bin di calibrazione empirica per il calcolo cumulativo dell'ECE (*Expected Calibration Error*).
