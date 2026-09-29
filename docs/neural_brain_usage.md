# Cervello neurale Makima — come usarlo

Makima ha un **cervello operativo** (`MakimaMindNet` + `BrainLoop`): percezione, memoria latente, attenzione episodica, scelte di azione. Non è sentience: i numeri di probabilità restano in Rust.

## Prerequisiti

```bash
pip install -e ".[research]"
# oppure almeno: pip install torch
```

Checkpoint pesi: `.makima/makima_brain.pt` (creato al primo `learn_step` / tick con learn).

## Comandi lab (Python)

```bash
# Un tick completo del cervello (monologo + azione)
python -m makima_lab think "Quando rilascerò il prossimo framework?"

# Chat con monologo
python -m makima_lab chat "Qual è lo stato del sistema?"

# Chat interattiva
python -m makima_lab chat

# Impulso idle
python -m makima_lab pulse

# Solo ispezione rete (senza deliberazione Mind)
python -m makima_lab neural "oggi ho completato il deploy"
```

## Forecast (numeri veri)

Se l’azione è `REQUEST_RUST_FORECAST`, il cervello **non** inventa la probabilità: ti indica:

```bash
makima query "Quando rilascerò il prossimo framework?"
```

Flusso corretto:

1. `think` / `chat` → BrainLoop capisce e decide l’azione  
2. `makima query ...` → core Rust calcola Beta/Poisson  
3. (opzionale) `python -m makima_lab explain <target>` → SLM spiega i numeri già calcolati  

## Azioni del cervello

| Azione | Significato |
|--------|-------------|
| `SPEAK` | Risposta narrativa / informativa |
| `REQUEST_RUST_FORECAST` | Serve probabilità → usa CLI Rust |
| `REMEMBER` | Consolida evidenza / osservazione |
| `ASK_CLARIFY` | Intent poco chiaro / fuori dominio |
| `IDLE` | Nessuna azione urgente |

## API Python

```python
from makima_lab.brain import get_brain

brain = get_brain()
result = brain.tick("Quanto è probabile il deploy?")
print(result.action)                 # BrainAction.REQUEST_RUST_FORECAST
print(result.pulse.inner_monologue)
print(result.rust_forecast_hint)     # makima query "..."
```

## Cosa non fa

- Non calcola probabilità di produzione  
- Non scrive prior neurali nello store Bayes  
- Non è una coscienza fenomenica: è una rete con memoria e politiche di azione  
