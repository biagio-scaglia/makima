# Makima Experiments

Questa directory ospita gli esperimenti statistici, gli script di validazione numerica, i benchmark comparativi e gli studi di ablazione tra modelli previsionali.

---

## 🔬 Suite di Forecasting Comparativo (`experiments/forecasting/`)

La sottocartella `forecasting/` implementa una testbed scientifica completa e riproducibile su oltre 5.000 campioni:

### 1. Modelli Comparativi Implementati (`baselines.py`)
1. **`constant_50`**: Baseline ingenua che assegna costantemente $P = 0.50$ (massima entropia).
2. **`climatological`**: Frequenza empirica storica mobile (base rate).
3. **`static_prior`**: Prior $\text{Beta}(1, 1)$ statico non aggiornato.
4. **`bayesian_conjugate`**: Aggiornamento esatto coniugato Beta-Binomiale.
5. **`bayesian_nlp`**: Modello bayesiano con pesatura della confidenza del parser semantico.
6. **`bayesian_neural`**: Modello bayesiano modulato dallo stato di polarità e attenzione della rete neurale `MakimaMindNet`.
7. **`full_makima`**: Modello completo che combina aggiornamento coniugato, modulazione neurale e processi temporali di Poisson $P(T \le t) = 1 - e^{-\lambda t}$.

### 2. Generatori di Dataset Sintetici (`datasets.py`)
- **Flussi Stazionari (`stationary_stream`)**: Processi Bernoulli i.i.d. con Ground Truth deterministica nota.
- **Cambi di Regime (`regime_shifts`)**: Simulazioni non stazionarie con salti improvvisi di probabilità.
- **Arrivi Temporali di Poisson (`poisson_arrivals`)**: Sequenze con intervalli inter-arrivo esponenziali $\Delta t \sim \text{Exp}(\lambda)$.

### 3. Metriche di Valutazione e Calibrazione (`calibration.py` & `evaluate.py`)
- **Brier Score** ($BS \in [0, 1]$): Errore quadratico medio di calibrazione.
- **Logarithmic Loss** (Cross-Entropy).
- **Brier Skill Score** ($BSS$): Guadagno percentuale rispetto alla baseline casuale o climatologica.
- **Expected Calibration Error** ($ECE$) e **Reliability Diagrams** ASCII.

---

## 🚀 Esecuzione dei Benchmark

Dalla console Makima o direttamente tramite terminale Python:

```bash
# Esecuzione del benchmark comparativo completo su tutti i modelli
python -m makima_lab benchmark

# Esecuzione dell'Ablation Study (impatto della rimozione di ciascun sottosistema)
python -m makima_lab ablation

# Esecuzione diretta dello script di benchmark
python experiments/forecasting/run_benchmarks.py
```

---

## 📐 Linee Guida per gli Esperimenti

1. **Riproducibilità**: ogni script deve impostare esplicitamente i seed dei generatori pseudo-casuali (`numpy.random.seed` o `rand::SeedableRng`).
2. **Documentazione**: ogni esperimento deve specificare chiaramente l'ipotesi di partenza, il dataset utilizzato, le metriche di valutazione e le conclusioni.
3. **Validazione prima del porting**: nessun algoritmo probabilistico viene inserito nel core Rust senza una precedente validazione sperimentale documentata.

