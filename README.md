# Makima

<p align="center">
  <img src="https://preview.redd.it/makima-eyes-reflect-what-we-saw-about-her-personality-the-v0-mmu71drhyu2g1.png?auto=webp&s=bd5f8ebc1bb092f0d20e1f67eabe660dceb02956" alt="Makima" width="480" />
</p>

> **An interpretable probabilistic forecasting system.**

Makima è un sistema di previsione probabilistica e quantificazione dell'incertezza, ispirato concettualmente alla *Laplace Mail* di *Shin Megami Tensei: Devil Survivor*.

Makima **non è un chatbot** e **non è costruita attorno a un LLM**. Il suo scopo è elaborare evidenze empiriche osservabili per produrre distribuzioni di probabilità interpretabili su eventi futuri, simulare scenari, quantificare rigorosamente l'incertezza e misurare la propria calibrazione nel tempo.

---

## 🎯 Obiettivi del Progetto

L'obiettivo finale di Makima è realizzare una pipeline end-to-end che consenta di:

1. **Raccogliere osservazioni storiche**: tracciare eventi, timestamp, comportamenti e segnali osservabili.
2. **Elaborare interrogazioni in linguaggio naturale**: trasformare domande espresse liberamente in rappresentazioni semantiche e temporali strutturate (*Intent*, *Target*, *Event*, *Temporal Horizon*).
3. **Costruire previsioni probabilistiche**: utilizzare modelli statistici e bayesiani trasparenti e matematicamente fondati.
4. **Simulare scenari futuri**: eseguire simulazioni Monte Carlo e modelli a catena di Markov per mappare le traiettorie possibili.
5. **Quantificare l'incertezza**: generare intervalli di confidenza/credibilità, distribuzioni di densità e stime entropiche.
6. **Confrontare previsioni e realtà**: registrare gli esiti effettivi degli eventi per calcolare *proper scoring rules* (Brier score, log score) e curve di calibrazione.
7. **Aggiornamento progressivo**: raffinare a posteriori i modelli e i prior bayesiani all'arrivo di nuove osservazioni.

---

## 🏛️ Architettura Tecnologica

Makima adotta un'architettura ibrida **Rust + Python** con una netta separazione delle responsabilità:

```
makima/
├── Cargo.toml                  # Workspace root Rust
├── rust-toolchain.toml         # Configurazione toolchain Rust stabile
├── pyproject.toml              # Specifiche e configurazione ambiente Python
├── crates/                     # Componenti del motore computazionale
│   ├── makima-core/            # Motore probabilistico, matematico e di dominio
│   └── makima-cli/             # Interfaccia da riga di comando
├── python/
│   └── makima_lab/             # Laboratorio di ricerca scientifica e NLP
├── experiments/                # Esperimenti e benchmark riproducibili
├── docs/                       # Documentazione architetturale e matematica
└── tests/                      # Suite di test e integrazione
```

### Ruolo di Rust (Core Engine)
Rust costituisce il cuore computazionale e applicativo di Makima:
- Teoria della probabilità, distribuzioni continue e discrete.
- Inferenza bayesiana e catene di Markov.
- Simulazioni Monte Carlo e calcolo dell'entropia / divergenza KL.
- Pipeline di forecasting e quantificazione dell'incertezza.
- Metriche di calibrazione e scoring rules.
- CLI e API applicative.

### Ruolo di Python (Scientific Lab & NLP)
Python funge da laboratorio scientifico e ambiente di prototipazione:
- Sperimentazione statistica e validazione matematica dei modelli prima del porting in Rust.
- Analisi ed esplorazione di dataset storici.
- Sperimentazione NLP (classificazione intent, estrazione entità e parsing temporale).
- Benchmark comparativi e validazione empirica.

L'integrazione tra i due mondi è predisposta per avvenire tramite **PyO3** e **maturin** esponendo selettivamente le primitive Rust ad alta performance all'ambiente Python.

---

## 🔬 Filosofia Scientifica e Riproducibilità

Makima segue un approccio rigoroso e trasparente:
- **Nessuna Black Box**: ogni previsione è tracciabile alle evidenze storiche e al modello che l'ha generata.
- **Fondamento Matematico**: ogni algoritmo implementato deve possedere una chiara definizione formale, test unitari e validazione su dati sintetici o reali.
- **Misurazione dell'Errore**: le previsioni passate vengono sistematicamente valutate per correggere eventuali bias sistematici (overconfidence / underconfidence).
- **Riproducibilità**: ogni esperimento e simulazione deve poter essere rieseguito con semi deterministici e condizioni controllate.

---

## 🗺️ Roadmap Iniziale

- [x] **Fase 1**: Inizializzazione workspace Rust, tooling di qualità e documentazione di base.
- [ ] **Fase 2**: Creazione del crate `makima-core` con entità di dominio e motore iniziale.
- [ ] **Fase 3**: Creazione del crate `makima-cli` con comando di health/status minimale.
- [ ] **Fase 4**: Configurazione del laboratorio scientifico Python (`pyproject.toml`, `makima_lab`).
- [ ] **Fase 5**: Documentazione architetturale dettagliata (`docs/architecture.md`).
- [ ] **Fase 6**: Documentazione dei fondamenti matematici e probabilistici previsti (`docs/mathematics.md`).
- [ ] **Fase 7**: Tooling avanzato di test e controlli di qualità continui.

---

## 📄 Licenza

Questo progetto è distribuito sotto licenza [MIT](LICENSE).
