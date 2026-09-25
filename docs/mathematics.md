# Makima — Mathematical Foundations Specification

Questo documento delinea i **fondamenti matematici, probabilistici e statistici previsti** per l'evoluzione di **Makima**. 

> [!NOTE]
> Questo testo definisce il **quadro teorico e la roadmap scientifica**. I componenti elencati descrivono le aree formali che verranno introdotte progressivamente e non intendono rappresentare algoritmi già implementati nel runtime corrente.

---

## 1. Principio Scientifico di Implementazione

Per evitare l'accumulo di formule fini a se stesse (*math washing*) o astrazioni opache, ogni futuro modulo matematico in Makima dovrà seguire obbligatoriamente questo ciclo di vita:

```text
Definizione Matematica Formale
             ↓
Validazione & Prototipazione nel Lab Python (`makima_lab`)
             ↓
Implementazione Idiomatica ad Alte Prestazioni in Rust (`makima-core`)
             ↓
Test Unitari & Property-Based Testing (invarianti assiomatici)
             ↓
Validazione Empirica su Dataset Sintetici e Storici
             ↓
Documentazione, Esempio Applicativo & Benchmark
```

---

## 2. Aree Matematiche Previste

### 2.1 Teoria della Probabilità Fondazionale
- **Spazio di probabilità**: formalizzazione della terna $(\Omega, \mathcal{F}, P)$ per definire in modo rigoroso gli eventi previsionali.
- **Assiomi di Kolmogorov**: rispetto delle proprietà di non-negatività, normalizzazione ($P(\Omega) = 1$) e additività numerabile.
- **Probabilità condizionata e indipendenza**: calcolo esplicito di $P(A \mid B) = \frac{P(A \cap B)}{P(B)}$.

### 2.2 Variabili Aleatorie (Discrete e Continue)
- **Variabili Discrete**: conteggio di eventi, stati discreti, occorrenze entro orizzonti finiti.
- **Variabili Continue**: orizzonti temporali continui, durata stimata per il verificarsi di un evento.
- **Funzione di Ripartizione (CDF)** e **Funzione di Densità/Massa di Probabilità (PDF/PMF)** calcolabili per ogni stima.

### 2.3 Distribuzioni di Probabilità
Le distribuzioni parametriche e non parametriche che costituiranno i mattoni analitici del motore:
- **Bernoulli e Binomiale**: stima della probabilità di accadimento di singoli eventi binari (successo/insuccesso).
- **Poisson ed Esponenziale**: modellazione della frequenza di accadimento di eventi in intervalli temporali fissi e del tempo di attesa inter-evento.
- **Beta e Dirichlet**: distribuzioni a priori coniugate per proporzioni e probabilità categoriche.
- **Gamma e Normale/Log-Normale**: modellazione di durate continue e variabilità simmetrica/asimmetrica.

### 2.4 Inferenza Bayesiana
- **Teorema di Bayes**: aggiornamento razionale dello stato di conoscenza:
  $$P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) P(\theta)}{P(\mathcal{D})}$$
- **Aggiornamento Sequenziale**: capacità di Makima di aggiornare la distribuzione a posteriori (*posterior*) ogni volta che viene registrata una nuova `Observation` empirica senza dover riaddestrare modelli monolitici.
- **Prior Coniugati e Stime Non-Informative**: utilizzo di prior trasparenti e dichiarati per gestire contesti con scarse osservazioni empiriche.

### 2.5 Catene di Markov e Processi Stocastici
- **Spazio degli Stati**: modellazione delle transizioni tra fasi discrete di un processo osservabile.
- **Matrici di Transizione**: stima delle probabilità di passaggio $P(S_{t+1} = j \mid S_t = i)$.
- **Distribuzione Stazionaria e Tempi di Primo Passaggio**: calcolo del tempo medio atteso per raggiungere un determinato stato obiettivo.

### 2.6 Metodi Monte Carlo
- **Campionamento Stocastico Deterministico**: generazione di traiettorie future tramite campionamento pseudocasuale con seed riproducibili.
- **Simulazione di Scenari Futuri**: aggregazione di migliaia di possibili evoluzioni per ricavare la distribuzione empirica della data o del risultato previsto.
- **Stima dell'Incertezza tramite Bootstrap**: quantificazione della variabilità campionaria sui parametri stimati.

### 2.7 Serie Temporali e Processi di Punto
- **Analisi degli Intervalli Temporali**: studio della distribuzione dei ritardi temporali tra osservazioni successive.
- **Rilevazione di Trend e Stagionalità**: decomposizione della frequenza di osservazione per identificare pattern comportamentali.

### 2.8 Teoria dell'Informazione ed Entropia di Shannon
- **Entropia**: misura dell'incertezza residua di una distribuzione discreta $H(X) = -\sum P(x) \log_2 P(x)$.
- **Diagnostica di Dispersione**: utilizzo dell'entropia per comunicare all'utente quanto la previsione sia concentrata (bassa entropia) o dispersa/incerta (alta entropia).

### 2.9 Divergenza di Kullback-Leibler (KL)
- **Misura del Guadagno Informativo**: quantificazione della variazione tra la distribuzione a priori e la distribuzione a posteriori all'arrivo di nuove evidenze:
  $$D_{\text{KL}}(P \parallel Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$$
- **Valutazione della Rilevanza delle Evidenze**: misurare quanto una singola osservazione abbia effettivamente modificato la credenza del modello.

### 2.10 Calibrazione Probabilistica
- **Reliability Diagrams**: raggruppamento delle previsioni per classi di probabilità (es. tutte le previsioni emesse al 70%) per verificare che la frequenza empirica reale sia pari al 70%.
- **Curve di Calibrazione**: identificazione di errori sistematici di *overconfidence* (eccesso di sicurezza) o *underconfidence* (eccesso di cautela).

### 2.11 Proper Scoring Rules
Metriche di valutazione che incoraggiano l'onestà e la precisione probabilistica penalizzando previsioni vaghe o sovrasicure:
- **Brier Score**: misura dell'errore quadratico medio per eventi probabilistici binari o multinomiali.
- **Logarithmic Score (Negative Log-Likelihood)**: penalizzazione logaritmica severa per eventi imprevisti a cui era stata attribuita probabilità quasi nulla.

### 2.12 Quantificazione dell'Incertezza
Distinzione analitica esplicita nell'output di previsione:
- **Incertezza Aleatoria**: variabilità intrinseca e non riducibile del processo stocastico.
- **Incertezza Epistemica**: incertezza dovuta alla mancanza di dati storici o conoscenza del dominio, riducibile raccogliendo ulteriori osservazioni.
- **Intervalli di Credibilità**: fornitura di bande al 50%, 80% e 95% per ogni previsione su orizzonti continui.

---

## 3. Criteri di Accettazione per Nuovi Moduli Matematici

Nessun nuovo modulo matematico potrà essere mergiato nel repository senza soddisfare tutti i seguenti requisiti:
1. **Definizione formale** nel modulo o nella documentazione di riferimento.
2. **Implementazione puramente Rust** priva di panics non controllati o allocazioni superflue.
3. **Test di Proprietà (Property Tests)** che dimostrino la conservazione delle proprietà matematiche (es. probabilità nell'intervallo $[0, 1]$, somme pari a $1$, non-negatività della divergenza KL).
4. **Validazione preliminare** archiviata negli esperimenti (`experiments/`).
