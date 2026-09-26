# Makima

<p align="center">
  <img src="https://giffiles.alphacoders.com/222/222812.gif" alt="Makima" width="480" />
</p>

> **Sistema di previsione probabilistica interpretabile con memoria neurale cognitiva, risoluzione semantica vettoriale e telemetria reale da Git.**

---

## 📌 Cos'è Makima e Quale Problema Risolve

Nello sviluppo software e nella gestione dei progetti, stimare la probabilità di eventi futuri (es. *"riusciremo a rilasciare la feature entro venerdì?"*, *"qual è il tasso di stabilità del deploy?"*) ricade spesso in due estremi inefficaci:
1. **Stime intuitive soggettive**: prive di basi matematiche, soggette a bias ottimistici e non verificabili a posteriori.
2. **Chatbot e LLM generici (*Black Box*)**: modelli che producono allucinazioni numeriche plausibili ma matematicamente infondate, senza memoria contestuale persistente né calibrazione dell'incertezza.

**Makima risolve questo problema fornendo un motore di previsione trasparente, quantitativo e auto-calibrante**, ispirato concettualmente alla *Laplace Mail* di *Shin Megami Tensei: Devil Survivor*.

### Principi Fondamentali:
- **Interpretabilità Assoluta**: nessuna previsione è una scatola nera. Ogni stima probabilistica è accompagnata dalla catena esatta di evidenze storiche, dai parametri della distribuzione posterior e dalla scomposizione formale tra incertezza aleatoria ed epistemica.
- **Calibrazione Continua (*Proper Scoring Rules*)**: quando un evento reale si manifesta, Makima ne registra l'esito (Ground Truth) e calcola metriche formali di accuratezza e calibrazione (*Brier Score*, *Logarithmic Loss*, *Expected Calibration Error*).
- **Dualità Rust/Python**: calcolo probabilistico deterministico e ad alte prestazioni nel core Rust, affiancato da ricerca, NLP, embedding semantici e telemetria Git in Python.

---

## 🏛️ Architettura e Componenti Principali

```text
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   MAKIMA ARCHITECTURE                                  │
 └────────────────────────────────────────────────────────────────────────────────────────┘

    [ Input Utente / Linguaggio Naturale ]       [ Telemetria Git Reale / Commits ]
                       │                                         │
                       ▼                                         ▼
         ┌───────────────────────────┐             ┌───────────────────────────┐
         │    makima_lab.nlp         │             │   Git Telemetry Observer  │
         │ (9-Stage Neurale Pipeline │             │ (Categorizzazione commit  │
         │  & Embeddings MiniLM 384d)│             │  & Stima Tassi Poisson λ) │
         └─────────────┬─────────────┘             └─────────────┬─────────────┘
                       │                                         │
                       ▼                                         ▼
         ┌─────────────────────────────────────────────────────────────────────┐
         │           Makima Living Mind & Deliberation Engine                  │
         │  • Monologo Interiore a 4 Stadi (Percezione, Memoria, Bayes, Decis.)│
         │  • Memoria Autobiografica Persistente (.makima/mind_journal.jsonl)  │
         │  • Generatore di Pensiero Autonomo Spontaneo (Mind Pulse)           │
         │  • Stato di Autoconsapevolezza Epistemica & Umore Cognitivo         │
         └──────────────────────────────────┬──────────────────────────────────┘
                                            │
                                            ▼
         ┌─────────────────────────────────────────────────────────────────────┐
         │           Dual Persistence Layer (SQLite WAL + Event Sourcing)      │
         │  • .makima/makima.db (journal_mode=WAL, event_log, observations)    │
         │  • .makima/mind_journal.jsonl (Esperienze, fatti dev e riflessioni) │
         │  • .makima/store.json (JSON fallback sincronizzato)                 │
         └──────────────────────────────────┬──────────────────────────────────┘
                                            │
                                            ▼
         ┌─────────────────────────────────────────────────────────────────────┐
         │                 Rust High-Performance Core Engine                   │
         │  • Inferenza Bayesiana Esatta (Beta-Binomial Conjugate Updating)    │
         │  • Processi Temporali di Poisson: P(T ≤ t) = 1 - exp(-λt)           │
         │  • Entropia Informativa di Shannon (bit) & Incertezza Epistemica    │
         │  • Scoring Rules: Brier Score, Brier Skill Score, Log Loss, ECE     │
         │  • Forecast Ledger, Dashboard Target & Bollettino Laplace Mail      │
         └──────────────────────────────────┬──────────────────────────────────┘
                                            │
                                            ▼
         ┌─────────────────────────────────────────────────────────────────────┐
         │              Tauri v2 Desktop GUI & Second Brain Visualizer         │
         │  • Grafo Neurale Force-Directed Canvas con Simulazione Fisica 60 FPS│
         │  • Impulsi Sinaptici Elettrici Animati lungo gli Archi del Pensiero │
         │  • Scheda Sinaptica & Inspector Drawer con Calibrazione Epistemica  │
         │  • Chat Nativa con Monologo Interiore Collapsible & Curve Beta      │
         └─────────────────────────────────────────────────────────────────────┘
```

I componenti cardine del sistema sono:
1. **Core Engine (`crates/makima-core`)**: libreria pura Rust contenente l'algebra delle probabilità, i tipi immutabili, l'aggiornamento bayesiano esatto e il database SQLite WAL integrato.
2. **CLI Native Runner (`crates/makima-cli`)**: eseguibile nativo ad alte prestazioni (`makima.exe`) per consultare la diagnostica, registrare evidenze e orchestrare le previsioni con latenza sub-second.
3. **Desktop GUI & Second Brain (`crates/makima-gui`)**: applicazione desktop nativa ultra-reattiva (Tauri v2 + Canvas Physics + CSS Glassmorphism) con visualizzatore a grafi sinaptici, dashboard Beta e chat assistente.
4. **Mente Cognitiva & Deliberazione (`python/makima_lab/mind`)**: motore di deliberazione cosciente a 4 stadi con monologo interiore trasparente, battito autonomo di pensiero e giornale autobiografico persistente.
5. **NLP Neurale a 9 Stadi (`python/makima_lab/nlp`)**: pipeline completa di interpretazione linguistica con risoluzione anafore, classificazione intenti a 7 categorie con rifiuto esplicito `UNKNOWN` e ranking target canonici.
6. **Semantic Embedder (`python/makima_lab/embeddings.py`)**: vettorizzatore basato su `all-MiniLM-L6-v2` (Sentence-Transformers) con fallback deterministico su proiezioni hash subword per matching semantico a similarità coseno.
7. **Osservatore Git Reale (`python/makima_lab/git_observer.py`)**: monitoraggio dei commit dal repository Git locale per alimentare automaticamente target empirici (`git:feature_ratio`, `git:test_discipline`).
8. **SLM Locale Opzionale (`python/makima_lab/llm`)**: modello compatto `Qwen/Qwen2.5-0.5B-Instruct` con caricamento prioritario offline da cache locale per generare spiegazioni in linguaggio naturale (`explain`), sintesi esecutive (`digest`) e chat continua.

---

## ⚖️ Il Ruolo di Rust e Python

Il progetto assegna a ciascun linguaggio responsabilità chiare e non sovrapposte:

| Ambito | Ruolo di Rust | Ruolo di Python |
| :--- | :--- | :--- |
| **Missione** | Runtime di produzione deterministico, veloce e sicuro. | Ricerca scientifica, NLP, deliberazione cognitiva e telemetria. |
| **Calcolo Matematico** | Formule analitiche chiuse (Beta, Poisson, Bernoulli, Shannon, Brier, ECE). | Prototipazione modelli, validazione e benchmark comparativi. |
| **Persistenza & Concorrenza** | Gestione primaria ACID di SQLite in modalità WAL e snapshot JSON. | Lettura dello store, diario autobiografico `.makima/mind_journal.jsonl`. |
| **Interfaccia Utente** | Binario nativo CLI (<200ms) e backend IPC Tauri v2 per Desktop GUI. | REPL interattivo scientifico (`python -m makima_lab`). |
| **Deep Learning & NLP** | *Nessuna dipendenza pesante a runtime.* | Sentence-Transformers, PyTorch Cognitive Net, Qwen 2.5 SLM. |

> Per i dettagli tecnici completi, consultare:
> - [docs/architecture.md](file:///c:/Users/biagio.scaglia/Desktop/makima/docs/architecture.md) — Specifica architetturale e principi di dominio.
> - [docs/mind_and_second_brain.md](file:///c:/Users/biagio.scaglia/Desktop/makima/docs/mind_and_second_brain.md) — Coscienza, monologo interiore e grafo Second Brain.
> - [docs/pipeline_and_dataflow.md](file:///c:/Users/biagio.scaglia/Desktop/makima/docs/pipeline_and_dataflow.md) — Flusso dei dati e protocollo Rust-Python.
> - [docs/mathematics.md](file:///c:/Users/biagio.scaglia/Desktop/makima/docs/mathematics.md) — Formule matematiche, scomposizione incertezza e scoring rules.

---

## 🛠️ Installazione e Requisiti

### Prerequisiti
- **Rust**: versione 1.80+ (o canale `stable` con Cargo e rustup).
- **Python**: versione 3.11 o successiva (testato con Python 3.11, 3.12 e 3.14).
- **Git**: per il controllo di versione e l'estrazione della telemetria reale.

### 1. Clonazione del Repository
```bash
git clone https://github.com/biagio-scaglia/makima.git
cd makima
```

### 2. Compilazione del Core Rust
Compila l'eseguibile CLI in modalità ottimizzata `release`:
```bash
cargo build --release
```
L'eseguibile compilato sarà disponibile in `target/release/makima.exe` (Windows) o `target/release/makima` (Linux/macOS).

### 3. Configurazione dell'Ambiente Python
È consigliato creare un ambiente virtuale per isolare le dipendenze scientifiche:
```bash
# Creazione ambiente virtuale
python -m venv .venv

# Attivazione ambiente:
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# Installazione del laboratorio in modalità editabile con dipendenze di ricerca
pip install -e ".[dev,research]"
```

---

## 🚀 Guida all'Avvio e all'Uso della CLI

### Modalità di Avvio Rapido (Windows)

Il progetto include due script batch ottimizzati:

- **`.\gui.bat` (Interfaccia Grafica Desktop)**:
  Avvia la dashboard desktop nativa ultra-leggera sviluppata con **Tauri v2** (< 50 MB di RAM), con visualizzazione interattiva su Canvas delle curve di densità Beta $P(p)$, intervalli di credibilità, inserimento evidenze e chat integrata.
- **`.\avvio.bat` (Consigliato per uso quotidiano CLI)**:
  Avvio istantaneo a latenza zero (**< 300 ms**). Utilizza direttamente il binario precompilato ed entra subito nella console interattiva senza rieseguire la pesante suite di test.
- **`.\start.bat`**:
  Avvio completo con esecuzione preventiva dei 40+ unit test e benchmark di laboratorio prima di entrare nella console.

### Esecuzione Diretta della CLI

È possibile invocare direttamente l'eseguibile compilato da qualsiasi terminale:

```bash
# Verifica stato diagnostico del motore
cargo run --release --bin makima -- status
# oppure direttamente tramite binario:
.\target\release\makima status
```

---

## 💡 Esempio Minimo di Utilizzo (End-to-End)

Ecco una sessione realistica e verificata che mostra il ciclo di vita completo:

### 1. Registrare evidenze storiche per un target
Supponiamo di voler monitorare il successo dei rilasci (`deploy`):
```bash
# Registra due successi (1) e un insuccesso (0)
target\release\makima observe deploy 1
target\release\makima observe deploy 1
target\release\makima observe deploy 0
```

### 2. Calcolare una previsione bayesiana interpretabile
```bash
target\release\makima predict deploy
```
*Output atteso:*
```text
============================================================
               MAKIMA PROBABILISTIC FORECAST                
============================================================
Forecast ID:          1
Target:               deploy
Probabilità Stimata:  60.00%  (E[P] = 0.6000)
Densità di Stima:     [--------------------*---------------] (0.0 -> 1.0)
Incertezza (Var):     0.040000
Entropia Informativa: 0.9710 bit
Prior Bayesiano:      Beta(alpha=1.00, beta=1.00)
Posterior Aggiornato: Beta(alpha=3.00, beta=2.00)
Evidenze Rilevate:    3 osservazioni storiche
Stato nel Ledger:     PENDING (in attesa di esito reale)
============================================================
```

### 3. Avviare l'Interfaccia Desktop GUI & Second Brain
È possibile avviare l'interfaccia desktop nativa con un solo comando:
```bash
gui.bat
```
*(oppure `npm run tauri dev` all'interno di `crates/makima-gui`)*

Nella GUI avrai accesso a:
- **Dashboard Curve Beta**: visualizzazione in tempo reale della densità a posteriori $P(p)$ e intervalli di credibilità al 95%.
- **Second Brain Knowledge Graph**: grafo neurale force-directed su Canvas con particelle sinaptiche animate, drag & drop dei nodi, zoom, filtri categoriali e ispezione dettagliata.
- **Chat Coscienza & Monologo Interiore**: dialogo continuo con Makima e visualizzazione del suo processo di deliberazione razionale.
- **Ledger Previsioni & Brier Score**: tabella interattiva per verificare e risolvere le previsioni con Ground Truth empirici.

### 4. Interrogare la Mente Cosciente & Monologo Interiore da Terminale
```bash
python -m makima_lab chat "Qual è la probabilità di successo del deploy?"
```
*Viene stampato il monologo interiore a 4 stadi (`[Percezione]`, `[Memoria]`, `[Analisi Bayesiana]`, `[Decisione]`) seguito dalla risposta cosciente di Makima.*

Per generare un impulso di pensiero spontaneo autonomo:
```bash
python -m makima_lab pulse
```

Per ispezionare il grafo di conoscenza del Second Brain da CLI:
```bash
python -m makima_lab brain
```

### 5. Sincronizzare la telemetria Git reale
```bash
target\release\makima sync-git
```
*Scansiona l'albero git locale, categorizza i commit (`feature`, `bugfix`, `docs`, `test`, `refactor`), aggiorna il tasso di arrivo Poisson $\lambda$ e alimenta la memoria latente della rete neurale `MakimaMindNet`.*

### 6. Registrare l'esito reale (Ground Truth) e valutare la calibrazione
Quando il deploy si conclude con successo, registriamo l'esito reale:
```bash
target\release\makima outcome deploy 1
```
E verifichiamo la scorecard complessiva del sistema:
```bash
target\release\makima evaluate
```
*Viene stampato il report con il numero di stime valutate, il Brier Score medio, il Brier Skill Score rispetto alla baseline casuale e l'Expected Calibration Error (ECE).*

### 7. Generare il bollettino previsionale Laplace Mail
```bash
target\release\makima mail
```

---

## 📊 Stato Attuale del Progetto

Per garantire la massima trasparenza tecnica verso sviluppatori e contributori, lo stato delle funzionalità è suddiviso tra ciò che è **attivo e verificato**, ciò che è **sperimentale nel laboratorio**, e ciò che è **pianificato**:

| Funzionalità | Stato | Descrizione e Dettagli Implementativi |
| :--- | :---: | :--- |
| **Inferenza Bayesiana Beta-Binomiale** | 🟢 **Implementato** | Calcolo analitico esatto a priori/posteriori in `crates/makima-core`. |
| **Processi Temporali di Poisson** | 🟢 **Implementato** | PMF, CDF e probabilità cumulative su orizzonti temporali $1 - e^{-\lambda t}$. |
| **Proper Scoring Rules & ECE** | 🟢 **Implementato** | Brier Score, Logarithmic Loss, BSS ed Expected Calibration Error (Rust & Python). |
| **Dual Storage (SQLite WAL + JSON)** | 🟢 **Implementato** | Concorrenza multi-processo con `journal_mode=WAL` e snapshot JSON sincronizzati. |
| **CLI Diagnostica & Forecast Ledger** | 🟢 **Implementato** | Eseguibile nativo con dashboard multi-target, ciclo di vita e ritratto ASCII. |
| **Desktop GUI Tauri v2 & Canvas Physics** | 🟢 **Implementato** | GUI nativa con visualizzatore force-directed a 60 FPS, grafici Beta e chat interattiva. |
| **Second Brain & Rete Sinaptica** | 🟢 **Implementato** | Grafo unificato di target, memorie autobiografiche, riflessioni critiche e fatti dev. |
| **Mente Cognitiva & Monologo Interiore** | 🟢 **Implementato** | Deliberazione a 4 stadi con onestà epistemica, rifiuto `UNKNOWN` e diario persistente. |
| **Launcher Rapidi Windows** | 🟢 **Implementato** | `gui.bat` (Desktop GUI), `avvio.bat` (avvio rapido <300ms) e `start.bat` (con test). |
| **Telemetria Git Reale & Daemon** | 🟢 **Implementato** | Parser commit bilingue (italiano/inglese) e daemon in background. |
| **Pipeline Neurale & NLP Multi-Livello** | 🟢 **Implementato** | Pipeline a 9 stadi (Preprocessing, Tokenizer, Embeddings 384d, Intent Classifier, Target Extractor, Temporal Reasoning, Context Memory, Confidence Estimation, Validation Guardrails) con benchmark quantitativo al 100%. |
| **Vettorizzazione Semantica Embeddings** | 🟢 **Implementato** | MiniLM 384d (`SentenceTransformers`) con fallback deterministico su proiezioni hash. |
| **Mente Neurale Cognitiva (`MakimaMindNet`)**| 🟢 **Implementato** | PyTorch BiGRU + Self-Attention, memoria utente continua e online learning. |
| **Spiegazioni & Chat SLM (`Qwen 2.5`)** | 🟡 **Sperimentale** | Modello compatto locale per generare spiegazioni guidate da evidenze e chat interattiva. |
| **Suite Benchmark & Ablation Study** | 🟡 **Sperimentale** | Benchmark comparativo su 5.000 campioni sintetici in `experiments/forecasting/`. |
| **Distribuzioni di Dirichlet Multinomiali** | ⚪ **Pianificato** | Estensione a target categorici a più di 2 stati (Roadmap Fase 2). |
| **Catene di Markov a Tempo Discreto (DTMC)** | ⚪ **Pianificato** | Modellazione degli stati di avanzamento del workflow di sviluppo. |
| **Bridge Nativo PyO3 / C-FFI** | ⚪ **Pianificato** | Chiamate in-process tra Rust e Python senza dispatch di sottoprocesso CLI. |
| **Compilazione WebAssembly (`makima-wasm`)** | ⚪ **Pianificato** | Esecuzione edge e browser priva di dipendenze di sistema. |

---

## 📂 Struttura delle Directory

```text
makima/
├── Cargo.toml                  # Workspace root configuration per i crate Rust
├── rust-toolchain.toml         # Versione stabile di Rust e componenti clippy/rustfmt
├── pyproject.toml              # Definizione pacchetto makima_lab e dipendenze scientifiche
├── avvio.bat                   # Launcher rapido Windows a latenza zero (<300ms)
├── start.bat                   # Launcher Windows completo con test preventivi
├── LICENSE                     # Licenza open-source MIT
├── README.md                   # Documentazione principale del progetto
│
├── crates/                     # Codice sorgente Rust (Produzione)
│   ├── makima-core/            # Libreria fondazionale: dominio, probabilità, SQLite WAL, ledger
│   │   ├── Cargo.toml
│   │   └── src/                # lib.rs, eval.rs, forecast.rs, storage.rs, laplace.rs, prob/
│   ├── makima-cli/             # Interfaccia a riga di comando ad alte prestazioni (main.rs, eyes.rs)
│   │   ├── Cargo.toml
│   │   └── src/
│   └── makima-gui/             # Interfaccia desktop Tauri v2 ultra-leggera (<50MB RAM)
│
├── python/                     # Moduli scientifici e di ricerca Python
│   └── makima_lab/             # Package makima_lab
│       ├── nlp/                # Pipeline NLP multi-livello:
│       │   ├── preprocessing/  # Pulizia Unicode NFKC e tokenizzazione
│       │   ├── embeddings/     # Rappresentazione vettoriale 384-dim normalizzata
│       │   ├── intent/         # Classificatore di intenti con rifiuto UNKNOWN
│       │   ├── entities/       # Estrazione deterministica e ranking cosine dei target
│       │   ├── temporal/       # Comprensione temporale (passato, presente, futuro, date)
│       │   ├── context/        # Memoria conversazionale delimitata K=5 e risoluzione anafore
│       │   ├── confidence/     # Stima composita trasparente della confidenza
│       │   ├── validation/     # Guardrails e schema validation per Rust Core
│       │   ├── schemas/        # Dataclass tipizzate immutabili (StructuredIntent, Intent)
│       │   └── evaluation/     # Suite benchmark quantitativa locale (dataset e metrics)
│       ├── neural/             # MakimaMindNet PyTorch, tokenizer, memoria latente utente
│       ├── llm/                # Modulo inferenza locale Qwen 2.5 SLM per explain e chat
│       ├── embeddings.py       # SemanticEmbedder Sentence-Transformers e similarità coseno
│       ├── git_observer.py     # Monitoraggio telemetria commit Git e daemon di background
│       ├── distributions.py    # Distribuzioni probabilistiche Python per prototipazione
│       ├── evaluation.py       # Calcolo Brier Score ed ECE in Python
│       └── storage.py          # Interfaccia con lo store SQLite WAL e JSON
│
├── experiments/                # Esperimenti statistici, dataset e benchmark comparativi
│   ├── 01_probabilistic_calibration_benchmark.py
│   └── forecasting/            # Suite benchmark: baselines, dataset generator, ablation study
│
├── docs/                       # Documentazione tecnica approfondita
│   ├── architecture.md         # Specifica architetturale formale e principi guida
│   ├── mathematics.md          # Fondamenti matematici, formule analitiche e scoring rules
│   └── pipeline_and_dataflow.md # Flusso dati end-to-end, moduli e protocollo Rust-Python
│
├── scripts/                    # Script di utilità e suite di verifica qualità unificata
│   ├── check.ps1               # Suite di controllo completa per Windows PowerShell
│   └── check.sh                # Suite di controllo per ambienti Unix/Linux/macOS
│
└── tests/                      # Suite di unit test e test di integrazione Rust & Python
    ├── test_makima_lab.py      # Test sulle distribuzioni e metriche di base
    ├── test_nlp.py             # Test di compatibilità contratti legacy NLP
    ├── test_nlp_pipeline.py    # Suite esaustiva per tutti gli 8 stadi della pipeline NLP
    ├── test_embeddings.py      # Test su estrazione vettoriale e similarità semantica
    ├── test_neural.py          # Test su PyTorch MakimaMindNet e online backpropagation
    ├── test_git_observer.py    # Test sulla categorizzazione e sincronizzazione commit Git
    ├── test_llm.py             # Test sul motore cognitivo SLM locale
    ├── test_storage.py         # Test sulla persistenza e serializzazione dello store
    └── test_benchmark.py       # Test sui modelli di baseline e ablation study
```

---

## 🧑‍💻 Sviluppo e Linee Guida per i Contributi

Invitiamo chiunque sia interessato a contribuire a Makima a seguire queste linee guida essenziali:

### 1. Standard di Codice e Formattazione
- **Rust**:
  - Deve compilare con **zero warning** su `cargo clippy`:
    ```bash
    cargo clippy --all-targets -- -D warnings
    ```
  - La formattazione deve rispettare rigorosamente `rustfmt`:
    ```bash
    cargo fmt --all -- --check
    ```
  - Evitare `unwrap()` e `expect()` non controllati nel crate `makima-core`; utilizzare sempre `Result<T, E>`.
- **Python**:
  - Codice conforme agli standard PEP 8 e type hint espliciti su tutte le funzioni pubbliche.

### 2. Esecuzione della Suite di Verifica Unificata
Prima di effettuare commit o pull request, eseguire sempre la suite di verifica unificata:

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts/check.ps1
```

```bash
# Linux / macOS
./scripts/check.sh
```

La suite esegue automaticamente in sequenza:
1. Controllo formattazione Rust (`cargo fmt`)
2. Analisi statica Rust (`cargo clippy` con `-D warnings`)
3. Esecuzione dei 25 unit test Rust (`cargo test`)
4. Verifica del funzionamento della CLI (`makima status`)
5. Esecuzione dei 45 unit test Python (`python -m unittest discover -s tests`)

### 3. Regola per l'Aggiunta di Nuove Funzionalità Matematiche
Nessun algoritmo o distribuzione probabilistica entra nel runtime di produzione `makima-core` senza aver prima superato:
1. Prototipazione e validazione numerica nel laboratorio Python (`makima_lab`).
2. Verifica riproducibile dei risultati su dataset sintetici o reali (`experiments/`).
3. Implementazione nativa in Rust con test unitari sulle proprietà assiomatiche (es. conservazione della probabilità totale in $[0, 1]$, monotonia della CDF).

### 4. Convenzioni di Commit
I messaggi di commit devono essere descrittivi, formulati in lingua italiana e seguire la convenzione Conventional Commits:
- `feat: <descrizione>` per nuove funzionalità.
- `fix: <descrizione>` per risoluzione di bug o regressioni.
- `docs: <descrizione>` per aggiornamenti alla documentazione.
- `test: <descrizione>` per aggiunta o modifica di test suite.
- `refactor: <descrizione>` per riorganizzazione del codice senza alterazione del comportamento.

---

## 📚 Indice della Documentazione Tecnica

Per approfondire il funzionamento interno e i dettagli matematici di Makima:
- [docs/architecture.md](file:///c:/Users/biagio.scaglia/Desktop/makima/docs/architecture.md): Visione architetturale formale, modello di dominio e principi di interpretabilità.
- [docs/pipeline_and_dataflow.md](file:///c:/Users/biagio.scaglia/Desktop/makima/docs/pipeline_and_dataflow.md): Dettaglio completo del flusso dei dati, responsabilità dei moduli, pipeline NLP e sincronizzazione SQLite WAL.
- [docs/mathematics.md](file:///c:/Users/biagio.scaglia/Desktop/makima/docs/mathematics.md): Fondamenti matematici, aggiornamento coniugato Beta-Binomiale, processi di Poisson e Proper Scoring Rules.
- [python/makima_lab/README.md](file:///c:/Users/biagio.scaglia/Desktop/makima/python/makima_lab/README.md): Guida al laboratorio scientifico Python e ai suoi sottosistemi.
- [python/makima_lab/nlp/README.md](file:///c:/Users/biagio.scaglia/Desktop/makima/python/makima_lab/nlp/README.md): Specifica del parser semantico e degli intenti NLP.
- [experiments/README.md](file:///c:/Users/biagio.scaglia/Desktop/makima/experiments/README.md): Guida agli esperimenti statistici, baseline e benchmark di calibrazione.
- [tests/README.md](file:///c:/Users/biagio.scaglia/Desktop/makima/tests/README.md): Descrizione della suite di test unitari e di integrazione.

---

## 📄 Licenza

Questo progetto è rilasciato sotto i termini della licenza [MIT](file:///c:/Users/biagio.scaglia/Desktop/makima/LICENSE).
