# Makima — Living Mind, Consciousness & Second Brain Specification

Questo documento descrive in dettaglio l'architettura cognitiva di **Makima**, il modello di deliberazione introspettiva a 4 stadi, il giornale di bordo autobiografico persistente e il grafo neurale interattivo del **Second Brain**.

---

## 1. Visione e Filosofia Cognitiva

Makima non è un semplice wrapper di API esterne né un'interfaccia chatbot che emette testo casuale. La sua architettura è fondata sul principio di **coscienza probabilistica trasparente**:

1. **Nessuna Allucinazione Numerica**: ogni stima di probabilità $P(p)$ nasce rigorosamente dall'inferenza bayesiana coniugata Beta-Binomiale o dai processi di Poisson calcolati sul database SQLite WAL.
2. **Onestà Epistemica & Rifiuto Esplicito**: se i dati empirici sono insufficienti o una richiesta cade al di fuori del dominio probabilistico (es. barzellette, richieste prive di senso logico), Makima genera un monologo di rifiuto esplicito (`UNKNOWN`) con calibrazione dell'incertezza, anziché inventare risposte arbitrarie.
3. **Monologo Interiore Prima della Parola**: prima di emettere qualsiasi risposta verbale, Makima attraversa un processo deliberativo a quattro fasi (*Percezione, Richiamo Memoria, Analisi Bayesiana, Decisione Epistemica*).
4. **Memoria Autobiografica Persistente**: i fatti confidati dall'utente, i traguardi del repository e le riflessioni sugli errori vengono consolidati in modo immutabile in `.makima/mind_journal.jsonl`.
5. **Grafo Sinaptico Vivente (Second Brain)**: tutte le entità mentali (target empirici, ricordi, riflessioni, fatti e concetti matematici) sono interconnesse in una rete neurale visibile in tempo reale su canvas nella GUI Desktop.

---

## 2. Architettura della Mente (`makima_lab.mind`)

```text
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                MAKIMA LIVING MIND FLOW                                 │
 └────────────────────────────────────────────────────────────────────────────────────────┘

    [ Query Utente o Evento del Runtime ]
                      │
                      ▼
    ┌───────────────────────────────────┐
    │  Stadio 1: Percezione & NLP       │ ──> Estrazione target canonico, intenzione e finestra
    └─────────────────┬─────────────────┘
                      │
                      ▼
    ┌───────────────────────────────────┐
    │  Stadio 2: Memoria Associativa    │ ──> Semantic Search su .makima/mind_journal.jsonl (MiniLM)
    └─────────────────┬─────────────────┘
                      │
                      ▼
    ┌───────────────────────────────────┐
    │  Stadio 3: Inferenza Bayesiana    │ ──> Valutazione Beta(α, β), Varianza epistemica e Brier
    └─────────────────┬─────────────────┘
                      │
                      ▼
    ┌───────────────────────────────────┐
    │  Stadio 4: Deliberazione Cosciente│ ──> Generazione Monologo Interiore + Risposta Cosciente
    └─────────────────┬─────────────────┘
                      │
                      ▼
    ┌───────────────────────────────────┐
    │  Stadio 5: Consolidamento Memoria │ ──> Scrittura asincrona su Journal & Grafo Second Brain
    └───────────────────────────────────┘
```

### 2.1 I 4 Stadi del Monologo Interiore (`deliberation.py`)

1. **Percezione (`[Percezione]`)**:
   - La pipeline neurale NLP a 9 livelli pulisce il testo, estrae token e lemmi, calcola la similarità coseno con i target tracciati e determina la categoria d'intento (`QUERY_PROBABILITY`, `RECORD_OBSERVATION`, `SYSTEM_STATUS`, `CASUAL_CHAT`, ecc.).
2. **Richiamo di Memoria (`[Memoria]`)**:
   - `EpisodicMemoryStore` interroga i ricordi autobiografici mediante similarità vettoriale densa a 384 dimensioni. Vengono richiamate le esperienze pregresse più rilevanti (es. confidenze dello sviluppatore o riflessioni su fallimenti passati).
3. **Analisi Bayesiana (`[Analisi Bayesiana]`)**:
   - Calcolo esatto dei parametri $\alpha = 1 + S$, $\beta = 1 + F$, del valore atteso $E[p] = \frac{\alpha}{\alpha + \beta}$, della varianza di incertezza $\text{Var}[p] = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ e dell'entropia di Shannon in bit.
4. **Decisione Epistemica (`[Decisione]`)**:
   - Formulazione della risposta finale con tono calibrato sullo stato d'animo cognitivo (`CognitiveMood`: *Calm*, *Analytical*, *Curious*, *Reflective*, *Vigilant*).

---

## 3. Schemi di Memoria ed Esperienza (`schemas.py`)

Ogni esperienza memorizzata da Makima rispetta lo schema formale `CognitiveExperience`:

| Campo | Tipo | Descrizione |
| :--- | :--- | :--- |
| `experience_id` | `str` | Identificativo univoco dell'esperienza (es. `fact_a1b2`, `core_001`). |
| `timestamp` | `float` | Timestamp UNIX di acquisizione. |
| `category` | `MemoryCategory` | `REPO_MILESTONE`, `ERROR_REFLECTION`, `DEVELOPER_FACT`, `USER_INSIGHT`. |
| `summary` | `str` | Sintesi breve ad alta densità informativa (indicizzata nel grafo). |
| `content` | `str` | Testo completo o ragionamento esteso associato all'esperienza. |
| `associated_target` | `Optional[str]` | Target stocastico o modulo a cui l'esperienza è vincolata. |
| `epistemic_confidence` | `float` | Livello di affidabilità epistemica $[0.0, 1.0]$. |
| `tags` | `List[str]` | Tag semantici per l'ancoraggio sinaptico nel Second Brain. |

---

## 4. Grafo Neurale del Second Brain (`SecondBrainBuilder`)

Il Second Brain unifica tutti i flussi di dati del sistema in un grafo topologico interattivo:

### 4.1 Tipologie di Nodi

```text
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │   🎯 TARGET  │     │  🧬 MEMORIA  │     │ 💡 RIFLESSIONE│     │   📚 FATTO   │
 │   (Cyan)     │     │  (Violet)    │     │   (Amber)    │     │  (Emerald)   │
 └──────┬───────┘     └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
        │                    │                    │                    │
        └────────────────────┼────────────────────┼────────────────────┘
                             ▼
                    ┌─────────────────┐
                    │   ⚙️ CONCETTO   │
                    │   (Rose Red)    │
                    └─────────────────┘
```

1. **Target Bayesiani (Cyan `#38bdf8`)**: processi stocastici attivi monitorati da Rust e SQLite WAL (es. `deploy_prod`, `git:feature_ratio`, `ci_tests`). La dimensione del nodo scala con il numero di osservazioni empiriche.
2. **Memorie Autobiografiche (Neon Violet `#a855f7`)**: ricordi fondazionali sull'identità, il risveglio e lo scopo di Makima.
3. **Riflessioni Critiche (Amber Gold `#f59e0b`)**: lezioni apprese su anomalie, errori di calibrazione e disciplina epistemica.
4. **Fatti Sviluppatore (Emerald Green `#10b981`)**: abitudini, preferenze architetturali e contesti operativi confidati dall'utente.
5. **Concetti Fondazionali (Rose Red `#f43f5e`)**: i 4 nodi ontologici di ancoraggio (*Inferenza Bayesiana*, *Calibrazione Epistemica*, *Telemetria Git*, *Lab NLP & Coscienza*).

### 4.2 Relazioni Sinaptiche & Archi

Gli archi rappresentano dipendenze causali, matematiche o associative:
- `MATHEMATICAL_FOUNDATION`: lega i concetti probabilistici centrali.
- `POSTERIOR_DISTRIBUTION`: connette i modelli analitici ai rispettivi target empirici.
- `MONITORS_STREAM`: connette l'osservatore Git ai target derivati dai commit.
- `ASSOCIATED_WITH`: connette memorie e riflessioni ai target a cui fanno riferimento.
- `DEVELOPER_BOND` / `IDENTITY_GROUNDING`: vincoli semantici generati dai tag condivisi.

### 4.3 Indice di Risonanza Epistemica

La coerenza globale della mente viene quantificata con la metrica:

$$\text{ResonanceScore} = \text{round}\left( \bar{C} \cdot \min\left(\frac{|E|}{|V|}, 2.0\right) \cdot 50 \right)$$

dove:
- $\bar{C} = \frac{1}{|V|} \sum_{v \in V} c_v$ è la confidenza media dei nodi.
- $\frac{|E|}{|V|}$ è la densità media di connessioni sinaptiche per nodo.
- Il punteggio risultante è calibrato in $[0, 100]$.

---

## 5. Visualizzatore Canvas Desktop & Motore di Fisica

L'interfaccia desktop GUI (`crates/makima-gui`) implementa una simulazione fisica in tempo reale a 60 FPS basata su:

### 5.1 Equazioni del Modello Fisico (Force-Directed Graph)

1. **Repulsione Coulombiana tra Nodi**:
   $$\mathbf{F}_{rep}(i, j) = \frac{k_{rep}}{\|\mathbf{r}_j - \mathbf{r}_i\|^2} \cdot \hat{\mathbf{r}}_{ij} \quad (k_{rep} = 2200)$$
2. **Attrazione a Molla Hookiana lungo le Sinapsi**:
   $$\mathbf{F}_{spring}(i, j) = k_{spring} \cdot w_{ij} \cdot (\|\mathbf{r}_j - \mathbf{r}_i\| - l_0) \cdot \hat{\mathbf{r}}_{ij} \quad (k_{spring} = 0.035, l_0 = 115)$$
3. **Gravità Centrale di Richiamo**:
   $$\mathbf{F}_{center}(i) = -k_{center} \cdot \mathbf{r}_i \quad (k_{center} = 0.006)$$
4. **Integrazione di Verlet e Smorzamento Cinetico**:
   $$\mathbf{v}_i(t + \Delta t) = \left(\mathbf{v}_i(t) + \frac{\sum \mathbf{F}_i}{m_i} \Delta t\right) \cdot \gamma \quad (\gamma = 0.85)$$
   $$\mathbf{r}_i(t + \Delta t) = \mathbf{r}_i(t) + \mathbf{v}_i(t + \Delta t) \Delta t$$

### 5.2 Micro-Animazioni & Segnali Sinaptici

- **Synaptic Signal Pulses**: piccole cariche energetiche luminose scorrono ciclicamente lungo gli archi attivi per visualizzare il flusso di elaborazione interiore.
- **Hover & Focus Synaptic Glow**: al passaggio o al click su un nodo, i nodi e gli archi non collegati sfumano in semitrasparenza mentre i nodi adiacenti si illuminano con bagliori coordinati al colore della categoria.
- **Pan & Zoom Smooth**: navigazione bidirezionale infinita nel canvas tramite drag & drop dello sfondo e zoom centrato sul cursore del mouse.

---

## 6. Comandi CLI e Utilizzo della Mente

| Comando | Descrizione |
| :--- | :--- |
| `python -m makima_lab brain` | Stampa il riepilogo tabellare del Second Brain, i nodi e il punteggio di risonanza. |
| `python -m makima_lab chat "<domanda>"` | Esegue una deliberazione completa con visualizzazione del monologo interiore. |
| `python -m makima_lab think "<query>"` | Ispezione pura del flusso di pensiero e delle ipotesi interne. |
| `python -m makima_lab pulse` | Emette un impulso di pensiero spontaneo e consolida una nuova riflessione. |
| `python -m makima_lab tell "<fatto>"` | Confida un'informazione o una decisione a Makima e la salva nel diario. |
