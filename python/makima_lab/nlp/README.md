# Makima Lab — NLP Semantic Query Parser

Il modulo `makima_lab.nlp` costituisce la componente di **Semantic Parsing** del laboratorio di ricerca Makima.
Il suo obiettivo è tradurre espressioni in linguaggio naturale in una rappresentazione formale e interpretabile (`ForecastQuery`), senza ricorrere a modelli a scatola nera (Black-Box LLM) per compiti deterministici e verificabili.

---

## 1. Pipeline di Parsing

```text
               Testo Utente (NL)
                      │
                      ▼
             Semantic Normalization
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Intent       Target      Temporal
     Classifier   Extractor     Extractor
          │           │           │
          └───────────┼───────────┘
                      ▼
                ForecastQuery
                      │
       (Validazione & Interpretabilità)
                      │
                      ▼
               makima-core API
```

---

## 2. Modelli di Dominio Semantico

### `Intent`
- `FORECAST`: Richiesta generale di stima probabilistica.
- `RELEASE_PREDICTION`: Previsione specifica su eventi di rilascio software / deploy.
- `OBSERVATION_RECORD`: Registrazione di un fatto/evidenza empirica.
- `STATUS_QUERY`: Ispezione dello stato interno del motore.
- `UNSUPPORTED`: Query non pertinente (es. chitchat, complimenti, domande aperte non probabilistiche) che viene esplicitamente rifiutata.

### `TemporalRelation` & `TemporalWindow`
- `NEXT`: Evento immediatamente successivo (es. *"il prossimo framework"*).
- `BEFORE`: Limite temporale superiore (es. *"entro dicembre"*, *"prima del 2026-12-31"*).
- `WITHIN_DAYS`: Finestra di $N$ giorni (es. *"entro 30 giorni"*).
- `THIS_WEEK`: Orizzonte settimanale corrente (es. *"questa settimana"*).
- `THIS_MONTH`: Orizzonte mensile corrente (es. *"questo mese"*).
- `UNSPECIFIED`: Orizzonte temporale non specificato esplicitamente.

### `ForecastQuery`
Rappresentazione intermedia generata:
```python
ForecastQuery(
    raw_query="Quando rilascerò il prossimo framework?",
    intent=Intent.RELEASE_PREDICTION,
    target="framework_release",
    temporal_window=TemporalWindow(relation=TemporalRelation.NEXT),
    confidence=0.90
)
```

---

## 3. Esempi di Query Supportate

| Query Naturale | Intent | Target Risolto | Temporal Window | Valid |
| :--- | :--- | :--- | :--- | :---: |
| *"Quando rilascerò il prossimo framework?"* | `RELEASE_PREDICTION` | `framework_release` | `NEXT` | Sì |
| *"Qual è la probabilità che rilasci framework entro dicembre?"* | `RELEASE_PREDICTION` | `framework_release` | `BEFORE(dicembre)` | Sì |
| *"Riuscirò a rilasciare framework entro 30 giorni?"* | `RELEASE_PREDICTION` | `framework_release` | `WITHIN_DAYS(30)` | Sì |
| *"Quanto è probabile che framework venga rilasciato questa settimana?"* | `RELEASE_PREDICTION` | `framework_release` | `THIS_WEEK` | Sì |
| *"rilasceremo la nuova feature questa settimana?"* | `RELEASE_PREDICTION` | `git:feature_ratio` (via embeddings) | `THIS_WEEK` | Sì |
| *"riusciremo a completare i test unitari entro 3 giorni?"* | `FORECAST` | `git:test_discipline` (via embeddings) | `WITHIN_DAYS(3)` | Sì |
| *"Quanto è bello il mio framework?"* | `UNSUPPORTED` | `None` | `UNSPECIFIED` | No |

---

## 4. Esecuzione da Riga di Comando

```bash
# Esecuzione diretta della pipeline semantica
python -m makima_lab query "rilasceremo la nuova feature questa settimana?"

# Dalla CLI Makima (eseguibile Rust)
makima query "riusciremo a completare i test unitari entro 3 giorni?"
```

