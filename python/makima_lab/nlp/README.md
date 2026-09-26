# Makima Lab — Architettura Neurale & Pipeline NLP Multi-Livello

Il modulo `makima_lab.nlp` implementa l'architettura neurale e di elaborazione del linguaggio naturale (**Natural Language Processing & Semantic Understanding**) di Makima.

A differenza dei sistemi monolitici basati su prompt generativi non vincolati (*Black-Box LLM prompting*), Makima adotta una **pipeline modulare a livelli specializzati**, in cui ogni componente ha una responsabilità tecnica determinata, un input tipizzato, un output verificabile e metriche empiriche misurabili.

---

## 1. Visione Architetturale della Pipeline

L'interpretazione semantica di una query dell'utente evolve progressivamente attraverso 8 stadi indipendenti prima di raggiungere il runtime deterministico di **Rust Core**:

```text
                       Input Utente (NL)
                              │
                              ▼
                     [ 1. Preprocessing ]
               (Pulizia NFKC, rimozione rumore)
                              │
                              ▼
                     [ 2. Tokenization ]
            (Token parole, stopword, n-grammi)
                              │
                              ▼
             [ 3. Semantic Representation ]
             (Embeddings 384-dim L2 Normalizzati)
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
     [ 4. Intent ]    [ 5. Target/Entity ]  [ 6. Temporal ]
    (Classification)      (Extraction)       (Understanding)
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
               [ 7. Confidence Estimation ]
             (Aggregazione trasparente segnali)
                              │
                              ▼
               [ 8. Validation & Guardrails ]
           (Schema, soglia minima, rifiuto UNKNOWN)
                              │
                              ▼
                     StructuredIntent DTO
                              │
                              ▼
                    Rust Core Execution
            (Aggiornamento Bayesiano & Stima)
```

---

## 2. Dettaglio dei Componenti e Responsabilità

### 2.1 Preprocessing (`preprocessing/cleaner.py`)
- **Responsabilità**: Normalizzazione lessicale e ortografica deterministica.
- **Input**: Stringa grezza (`str`).
- **Operazioni**: Normalizzazione Unicode `NFKC`, uniformazione di apostrofi e virgolette tipografiche, rimozione di caratteri di controllo non stampabili, preservazione di identificatori con namespace (es. `git:feature_ratio`).
- **Output**: Testo normalizzato in minuscolo privo di ambiguità tipografiche.

### 2.2 Tokenizzazione (`preprocessing/tokenizer.py`)
- **Responsabilità**: Suddivisione in unità lessicali, filtraggio selettivo di stopword in lingua italiana ed estrazione di n-grammi continui ($N=2, 3$).
- **Utilizzo**: Riconoscimento di entità composte (es. *"daily build"*, *"unit test"*, *"feature ratio"*).

### 2.3 Rappresentazione Semantica & Embeddings (`embeddings/representation.py`)
- **Responsabilità**: Mappatura dello spazio lessicale in uno spazio vettoriale denso continuo.
- **Specifiche Tecniche**:
  - **Modello**: `all-MiniLM-L6-v2` (*Sentence-Transformers*) con fallback deterministico offline su proiezioni subword hash.
  - **Dimensionalità**: $D = 384$ float32.
  - **Normalizzazione**: Euclidea L2 ($\|\mathbf{v}\|_2 = 1.0$).
  - **Metrica**: *Cosine Similarity* $\in [-1.0, 1.0]$.
- **Utilizzo**: Ranking dei candidati di target semantici e disambiguazione di sinonimi e parafrasi nel dominio software engineering (es. *"rilasciare software"* $\leftrightarrow$ *"distribuzione release"*).

### 2.4 Classificazione degli Intenti (`intent/classifier.py`)
- **Responsabilità**: Riconoscimento dell'obiettivo semantico primario.
- **Insieme di Intenti Supportati**:
  - `QUERY`: Richiesta di stima o probabilità su un target noto.
  - `TEMPORAL_QUERY`: Richiesta con orizzonte temporale specifico (es. *"quando rilascerò..."*).
  - `COMMAND`: Istruzione operativa diretta per il motore (es. sincronizzazione Git, ricalibrazione).
  - `INFORMATION`: Domande esplicative di dominio (es. *"spiegami il Brier Score"*).
  - `OBSERVATION`: Registrazione di un'evidenza empirica osservata.
  - `STATUS`: Ispezione dello stato interno e calibrazione.
  - `UNKNOWN`: Rifiuto esplicito di input fuori dominio, chitchat o frasi prive di significato semantico verificabile.

> [!IMPORTANT]
> Un sistema intelligente deve poter dichiarare esplicitamente `UNKNOWN` invece di allucinare una risposta probabilistica inventata.

### 2.5 Estrazione Target ed Entità (`entities/target_extractor.py`)
- **Responsabilità**: Risoluzione del target probabilistico specifico.
- **Strategia Multi-Stadio**:
  1. *Corrispondenza esatta* su identificatori di registro (es. `deploy`, `api_gateway`).
  2. *Descrittori lessicali canonici* (es. *"compilazione"* $\rightarrow$ `daily_build`).
  3. *Estrazione sintattica posizionale* a valle di verbi di azione.
  4. *Risoluzione semantica vettoriale* con soglia minima di significatività ($\text{sim} \ge 0.42$).

### 2.6 Comprensione Temporale (`temporal/analyzer.py`)
- **Responsabilità**: Distinzione rigorosa e tipizzata dell'orizzonte temporale.
- **Relazioni Temporali Supportate**:
  - `PAST`: Evento o serie nel passato (es. *"cosa ho fatto ieri"*).
  - `PRESENT`: Attività corrente o stato in corso (es. *"cosa sto facendo attualmente"*).
  - `FUTURE`: Evento futuro o prossimo rilascio (es. *"quando rilascerò il prossimo framework"*).
  - `RELATIVE_INTERVAL`: Finestra relativa con conteggio giorni esplicito (es. *"entro 7 giorni"*, *"questa settimana"*).
  - `SPECIFIC_DATE`: Data di calendario puntuale conforme ISO-8601 o mese esplicito (es. `2026-12-31`, *"entro dicembre"*).
  - `UNKNOWN`: Orizzonte non specificato.

### 2.7 Memoria Conversazionale Delimitata (`context/memory.py`)
- **Responsabilità**: Risoluzione controllata di riferimenti anaforici e pronomi relativi senza accumulo indefinito di token.
- **Struttura**: Coda FIFO circolare a capacità fissa ($K = 5$ turni).
- **Risoluzione Anaforica**:
  - Riferimenti retrospettivi: *"e per quello precedente?"* $\rightarrow$ seleziona il penultimo target dalla cronologia.
  - Riferimenti pronominali diretti: *"quando lo rilascio?"* $\rightarrow$ associa il target dell'ultimo turno attivo.

### 2.8 Stima Composita della Confidenza (`confidence/estimator.py`)
- **Responsabilità**: Calcolo trasparente, tracciabile e motivato della confidenza numerica.
- **Formula di Combinazione**:
  $$\text{Conf} = w_{\text{intent}} \cdot S_{\text{intent}} + w_{\text{target}} \cdot S_{\text{target}} + w_{\text{temporal}} \cdot S_{\text{temporal}} - P_{\text{coherence}}$$
  dove:
  - $w_{\text{intent}} = 0.40$, $w_{\text{target}} = 0.40$, $w_{\text{temporal}} = 0.20$
  - $P_{\text{coherence}} = 0.25$ se l'intento richiede un target probabilistico che non è stato individuato.

### 2.9 Validazione Strutturata e Guardrails (`validation/validator.py`)
- **Responsabilità**: Firewall semantico che garantisce la conformità formale prima di trasmettere i dati al runtime Rust.
- **Regole Applicate**:
  - Rifiuto categorico di richieste `Intent.UNKNOWN`.
  - Verifica della soglia minima di confidenza ($\tau \ge 0.25$).
  - Imposizione della presenza di un target valido per `QUERY` e `OBSERVATION`.
  - Generazione di `validation_notes` diagnostiche in caso di rifiuto.

---

## 3. Ruolo del Modello Linguistico Locale (Qwen 2.5 1B)

Il modello **Qwen 2.5 1B** (`makima_lab.llm.QwenCognitiveEngine`) opera come **motore cognitivo specializzato** e non come decisore unico:
- **Spiegazione & Interpretazione**: Generazione di sintesi testuali in linguaggio naturale per spiegare la distribuzione Beta posteriore, la varianza e i Brier Score calcolati dal core.
- **Digest di Sintesi**: Riassunto strutturato dello stato multi-target e dell'Expected Calibration Error.
- **Isolamento**: Il modello generativo non può sovrascrivere i parametri probabilistici deterministici né emettere previsioni numeriche non convalidate dalla pipeline.

---

## 4. Confine Rust Core vs Python NLP Lab

| Dominio | Ambiente | Responsabilità |
| :--- | :---: | :--- |
| **NLP & Semantica** | Python | Preprocessing, Tokenizzazione, Embeddings 384-dim, Intent Classification, Entity Extraction, Temporal Reasoning, Confidence Estimation, Context Memory. |
| **Validazione & Schemi** | Python $\rightarrow$ Rust | Serializzazione `StructuredIntent` JSON, validazione guardrails, diagnosi di rigetto. |
| **Inferenza Deterministica** | Rust Core | Aggiornamento Bayesiano Beta/Bernoulli, simulazioni Poisson, calcolo Brier Score, persistenza SQLite e runtime Tauri. |

---

## 5. Valutazione e Benchmark Quantitativo

La suite di benchmark (`evaluation/benchmark.py`) esegue test quantitativi automatici rispetto al dataset di riferimento (`evaluation/dataset.py`).

### Metriche Rilevate:
- **Intent Accuracy**: Misura la frazione di intenti correttamente classificati rispetto alla ground truth.
- **Target Accuracy**: Misura la frazione di target identificati esattamente o correttamente omessi.
- **Temporal Accuracy**: Misura la precisione nel riconoscimento della relazione temporale e dei giorni di orizzonte.
- **Core Validity Agreement**: Coerenza tra la decisione di ammissione al core e l'ammissibilità attesa.
- **Invalid Output Rate**: Percentuale di query ambigue o fuori dominio correttamente respinte.
- **Latenza (Media e P95)**: Tempo di esecuzione per query in millisecondi.

```bash
# Esecuzione del benchmark locale
python -m makima_lab.nlp.evaluation.benchmark
```

---

## 6. Matrice di Stato dei Componenti

| Componente | Stato | Descrizione |
| :--- | :---: | :--- |
| **TextCleaner & SimpleTokenizer** | `IMPLEMENTED` | Normalizzazione NFKC, apostrofi, stopword, n-grammi. |
| **Semantic Embedder (384-dim)** | `IMPLEMENTED` | MiniLM L6 v2 con fallback vettoriale offline subword. |
| **IntentClassifier** | `IMPLEMENTED` | 7 categorie realistiche con rifiuto esplicito `UNKNOWN`. |
| **TargetExtractor** | `IMPLEMENTED` | Descrittori canonici, posizionali e ranking cosine similarity. |
| **TemporalAnalyzer** | `IMPLEMENTED` | Passato, presente, futuro, intervalli relativi, date ISO. |
| **ConversationContext** | `IMPLEMENTED` | Buffer circolare FIFO $K=5$ con risoluzione anafore. |
| **ConfidenceEstimator** | `IMPLEMENTED` | Funzione trasparente multi-segnale pesata. |
| **IntentValidator** | `IMPLEMENTED` | Guardrails per l'interfaccia verso Rust Core. |
| **Qwen 2.5 1B Explanation Engine** | `IMPLEMENTED` | Generazione spiegazioni e digest guidati da evidenze. |
| **Fine-tuning Neurale Dedicato per Intent**| `PLANNED` | Distillazione di un piccolo classificatore neurale su dataset annotato. |
