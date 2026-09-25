# Makima — Mathematical Foundations Specification

Questo documento delinea i **fondamenti matematici, probabilistici e statistici** alla base di **Makima**, descrivendo sia i moduli attivi nel runtime sia la roadmap di estensione formale.

---

## 1. Principio Scientifico di Implementazione

Per evitare l'accumulo di formule fini a se stesse (*math washing*) o astrazioni opache, ogni modulo matematico in Makima segue rigorosamente questo ciclo di vita:

```text
Definizione Matematica Formale
             ↓
Validazione & Prototipazione nel Lab Python (`makima_lab`)
             ↓
Implementazione Idiomatica ad Alte Prestazioni in Rust (`makima-core`)
             ↓
Test Unitari & Property-Based Testing (invarianti assiomatici)
             ↓
Validazione Empirica su Dati Reali (Git Telemetry & Event Log)
             ↓
Documentazione, Esempio Applicativo & Benchmark
```

---

## 2. Fondamenti Matematici Attivi nel Runtime

### 2.1 Distribuzione di Bernoulli ed Entropia di Shannon
Per un evento binario $X \in \{0, 1\}$ con parametro $p = P(X = 1)$:
- **Valore Atteso**: $E[X] = p$
- **Varianza**: $\text{Var}(X) = p(1 - p)$
- **Entropia Informativa di Shannon** (in bit):
  $$H(p) = -p \log_2(p) - (1-p) \log_2(1-p)$$
  con la convenzione $0 \log_2 0 = 0$. Rappresenta la quantificazione dell'incertezza informativa residua (massima per $p = 0.5$ con $H = 1.0\text{ bit}$, nulla per eventi deterministici $p \in \{0, 1\}$).

### 2.2 Inferenza Bayesiana con Coniugata Beta-Binomiale
Dato un prior $\text{Beta}(\alpha_0, \beta_0)$ e $n$ osservazioni empiriche contenenti $s$ successi e $f = n - s$ fallimenti:
- **Funzione di Densità di Probabilità**:
  $$f(p; \alpha, \beta) = \frac{1}{\text{B}(\alpha, \beta)} p^{\alpha - 1} (1 - p)^{\beta - 1}$$
- **Aggiornamento Bayesiano Analitico**:
  $$\alpha' = \alpha_0 + s, \quad \beta' = \beta_0 + f$$
- **Probabilità Attesa (Media del Posterior)**:
  $$E[P] = \frac{\alpha'}{\alpha' + \beta'}$$
- **Incertezza Epistemica (Varianza del Posterior)**:
  $$\text{Var}(P) = \frac{\alpha' \beta'}{(\alpha' + \beta')^2 (\alpha' + \beta' + 1)}$$
  All'aumentare delle evidenze $n \to \infty$, $\text{Var}(P) \to 0$, riducendo progressivamente l'incertezza epistemica.

### 2.3 Processi Temporali di Poisson
Per il conteggio di eventi che avvengono con un tasso costante $\lambda > 0$ eventi per unità di tempo:
- **Funzione di Massa (PMF)**:
  $$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$
- **Funzione di Ripartizione Cumulativa (CDF)**:
  $$P(X \le k) = e^{-\lambda} \sum_{i=0}^k \frac{\lambda^i}{i!}$$
- **Probabilità di Almeno un Accadimento in una Finestra $T$**:
  $$P(T \le t) = 1 - e^{-\lambda t}$$
  utilizzata per le interrogazioni con finestra temporale (*"questa settimana"*, *"entro 30 giorni"*).

### 2.4 Proper Scoring Rules & Calibrazione Empirica
Makima valuta la qualità probabilistica a fronte degli esiti reali $o \in \{0, 1\}$:
- **Brier Score**:
  $$BS = (p - o)^2 \in [0, 1]$$
- **Logarithmic Score (Log Loss)**:
  $$LL = -\left(o \ln(p) + (1 - o) \ln(1 - p)\right)$$
- **Brier Skill Score (BSS)** rispetto a baseline non informativa ($p_{\text{ref}} = 0.5$):
  $$BSS = 1 - \frac{\overline{BS}}{BS_{\text{ref}}}$$
- **Expected Calibration Error (ECE)**:
  $$ECE = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

### 2.5 Rappresentazione Semantica e Similarità Coseno
Per associare richieste spontanee in linguaggio naturale ai target monitorati, `SemanticEmbedder` mappa stringhe $S$ in vettori normalizzati $\mathbf{u} \in \mathbb{R}^{384}$:
$$\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

---

## 3. Aree Matematiche della Roadmap

Le seguenti aree teoriche verranno integrate nelle fasi future:
- **Distribuzioni di Dirichlet**: estensione multinomiale della Beta per target categorici multi-stato.
- **Catene di Markov a Tempo Discreto (DTMC)**: matrici di transizione $P(S_{t+1} = j \mid S_t = i)$ per modellare stati di avanzamento del software (Design, Testing, Review, Deploy).
- **Simulazioni Monte Carlo Riproducibili**: campionamento vettoriale con seed deterministici per scenari complessi multi-variabile.
- **Divergenza di Kullback-Leibler ($D_{\text{KL}}$)**: quantificazione del guadagno informativo istantaneo:
  $$D_{\text{KL}}(P \parallel Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$

---

## 4. Criteri di Accettazione per Nuovi Moduli Matematici

Nessun nuovo modulo matematico potrà essere mergiato nel repository senza soddisfare tutti i seguenti requisiti:
1. **Definizione formale** nel modulo o nella documentazione di riferimento.
2. **Implementazione puramente Rust** priva di panics non controllati o allocazioni superflue.
3. **Test di Proprietà (Property Tests)** che dimostrino la conservazione delle proprietà matematiche (es. probabilità nell'intervallo $[0, 1]$, somme pari a $1$, non-negatività della divergenza KL).
4. **Validazione preliminare** archiviata negli esperimenti (`experiments/`) o nella test suite Python.
