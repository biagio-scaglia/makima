//! Persistenza su file e storage atomico per lo stato empirico di Makima.

use crate::{MakimaEngine, Observation, ObservationId, Outcome};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter};
use std::path::{Path, PathBuf};

/// Struttura dati serializzata su disco in formato JSON per la persistenza dello stato.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MakimaStore {
    /// Versione del formato dello store.
    pub version: String,
    /// Elenco completo delle osservazioni empiriche registrate.
    pub observations: Vec<Observation>,
    /// Elenco degli esiti reali registrati per la valutazione di calibrazione.
    pub outcomes: Vec<Outcome>,
}

impl Default for MakimaStore {
    fn default() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            observations: Vec::new(),
            outcomes: Vec::new(),
        }
    }
}

impl MakimaStore {
    /// Restituisce il percorso standard per il file di stato: `.makima/store.json`.
    #[must_use]
    pub fn default_path() -> PathBuf {
        PathBuf::from(".makima").join("store.json")
    }

    /// Carica lo store da file o inizializza lo storage con i dati di riferimento se non esiste.
    pub fn load_or_init(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        if path.exists() {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            let store: Self = serde_json::from_reader(reader)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            Ok(store)
        } else {
            let sample = Self::sample_store();
            sample.save(path)?;
            Ok(sample)
        }
    }

    /// Salva lo stato corrente su disco garantendo la creazione delle cartelle intermedie.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self).map_err(io::Error::other)?;
        Ok(())
    }

    /// Crea un set iniziale di evidenze storiche per popolare il dataset di avvio.
    #[must_use]
    pub fn sample_store() -> Self {
        let samples = [
            ("framework_release", 1.0, 1_700_000_000),
            ("framework_release", 1.0, 1_700_086_400),
            ("framework_release", 0.0, 1_700_172_800),
            ("framework_release", 1.0, 1_700_259_200),
            ("framework_release", 1.0, 1_700_345_600),
            ("framework_release", 1.0, 1_700_432_000),
            ("framework_release", 0.0, 1_700_518_400),
            ("framework_release", 1.0, 1_700_604_800),
            ("daily_build", 1.0, 1_700_000_000),
            ("daily_build", 1.0, 1_700_086_400),
            ("daily_build", 1.0, 1_700_172_800),
            ("api_gateway", 1.0, 1_700_000_000),
            ("api_gateway", 1.0, 1_700_086_400),
            ("api_gateway", 0.0, 1_700_172_800),
            ("api_gateway", 1.0, 1_700_259_200),
        ];

        let observations = samples
            .iter()
            .enumerate()
            .map(|(idx, (target, val, ts))| {
                Observation::new(ObservationId((idx + 1) as u64), *target, *ts, *val)
            })
            .collect();

        let outcomes = vec![
            Outcome::new("framework_release", true, 1_700_650_000),
            Outcome::new("daily_build", true, 1_700_200_000),
        ];

        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            observations,
            outcomes,
        }
    }

    /// Applica i dati dello store a un'istanza di [`MakimaEngine`].
    pub fn apply_to_engine(&self, engine: &mut MakimaEngine) {
        for obs in &self.observations {
            engine.record_observation(obs.clone());
        }
        for out in &self.outcomes {
            engine.record_outcome(&out.target, out.occurred, out.timestamp_sec);
        }
    }

    /// Estrae lo stato corrente da un'istanza di [`MakimaEngine`].
    #[must_use]
    pub fn from_engine(engine: &MakimaEngine) -> Self {
        Self {
            version: engine.version().to_string(),
            observations: engine.observations().to_vec(),
            outcomes: engine.outcomes().to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_store_creation_and_apply() {
        let store = MakimaStore::sample_store();
        assert!(!store.observations.is_empty());
        assert!(!store.outcomes.is_empty());

        let mut engine = MakimaEngine::new();
        store.apply_to_engine(&mut engine);

        assert_eq!(engine.observations().len(), store.observations.len());
        assert_eq!(engine.outcomes().len(), store.outcomes.len());
    }

    #[test]
    fn test_store_roundtrip_json() {
        let store = MakimaStore::sample_store();
        let json_str = serde_json::to_string(&store).expect("serializzazione riuscita");
        let decoded: MakimaStore =
            serde_json::from_str(&json_str).expect("deserializzazione riuscita");

        assert_eq!(store, decoded);
    }
}
