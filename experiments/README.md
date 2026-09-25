# Makima Experiments

Questa directory ospita gli esperimenti statistici, gli script di validazione numerica e i benchmark comparativi tra modelli previsionali.

---

## Linee Guida per gli Esperimenti

1. **Riproducibilità**: ogni script deve impostare esplicitamente i seed dei generatori pseudo-casuali (es. `numpy.random.seed` o `rand::SeedableRng`).
2. **Documentazione**: ogni esperimento deve specificare chiaramente l'ipotesi di partenza, il dataset utilizzato, le metriche di valutazione (es. Brier Score, log-likelihood) e le conclusioni.
3. **Validazione prima del porting**: nessun algoritmo probabilistico viene inserito nel core Rust senza una precedente validazione sperimentale documentata.
