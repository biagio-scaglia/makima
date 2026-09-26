use crate::forecast::{ForecastId, ForecastRecord, ForecastStatus};
use crate::prob::Probability;
use crate::{MakimaEngine, Observation, ObservationId, Outcome};
use rusqlite::{params, Connection};
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
    /// Registro storico del ciclo di vita delle previsioni emesse.
    #[serde(default)]
    pub forecasts: Vec<ForecastRecord>,
}

impl Default for MakimaStore {
    fn default() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            observations: Vec::new(),
            outcomes: Vec::new(),
            forecasts: Vec::new(),
        }
    }
}

impl MakimaStore {
    /// Restituisce il percorso standard per il file di stato JSON: `.makima/store.json`.
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

        let mut forecasts = Vec::new();
        let mut f1 = ForecastRecord::new_pending(
            ForecastId(1),
            "framework_release",
            1_700_600_000,
            Probability::from_clamped(0.70),
            "7 days",
            8,
            "BayesianConjugate",
        );
        f1.resolve(true, 1_700_650_000);
        forecasts.push(f1);

        let mut f2 = ForecastRecord::new_pending(
            ForecastId(2),
            "daily_build",
            1_700_150_000,
            Probability::from_clamped(0.80),
            "24 hours",
            3,
            "BayesianConjugate",
        );
        f2.resolve(true, 1_700_200_000);
        forecasts.push(f2);

        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            observations,
            outcomes,
            forecasts,
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
            forecasts: engine.ledger().records().to_vec(),
        }
    }
}

/// Gestore del database relazionale incorporato SQLite con supporto WAL ed Event Sourcing.
pub struct MakimaDb {
    conn: Connection,
}

impl MakimaDb {
    /// Restituisce il percorso standard per il file database SQLite: `.makima/makima.db`.
    #[must_use]
    pub fn default_path() -> PathBuf {
        PathBuf::from(".makima").join("makima.db")
    }

    /// Apre o crea una connessione al database SQLite, abilitando WAL mode e creando lo schema.
    pub fn open(path: impl AsRef<Path>) -> rusqlite::Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        let conn = Connection::open(path)?;

        // Configura la modalità WAL per letture/scritture concorrenti e veloci
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;

        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    /// Crea lo schema delle tabelle di Event Sourcing, osservazioni, esiti, ledger e diario se non presenti.
    fn init_schema(&self) -> rusqlite::Result<()> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS event_log (
                offset INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                target TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                timestamp_sec INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp_sec INTEGER NOT NULL,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target TEXT NOT NULL,
                occurred INTEGER NOT NULL,
                timestamp_sec INTEGER NOT NULL,
                brier_score REAL,
                log_loss REAL
            );

            CREATE TABLE IF NOT EXISTS forecast_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target TEXT NOT NULL,
                created_at_sec INTEGER NOT NULL,
                probability REAL NOT NULL,
                window_desc TEXT NOT NULL,
                evidence_count INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                status TEXT NOT NULL,
                outcome INTEGER,
                brier_score REAL,
                calibration_bucket TEXT,
                resolved_at_sec INTEGER
            );

            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_text TEXT NOT NULL,
                extracted_target TEXT,
                extracted_intent TEXT,
                timestamp_sec INTEGER NOT NULL
            );
            "#,
        )?;
        Ok(())
    }

    /// Registra una nuova osservazione empirica nel database SQLite.
    pub fn insert_observation(
        &self,
        target: &str,
        value: f64,
        timestamp_sec: i64,
        notes: Option<&str>,
    ) -> rusqlite::Result<i64> {
        self.conn.execute(
            "INSERT INTO observations (target, value, timestamp_sec, notes) VALUES (?1, ?2, ?3, ?4)",
            params![target, value, timestamp_sec, notes],
        )?;
        let id = self.conn.last_insert_rowid();

        // Registra anche nell'event_log append-only (Event Sourcing)
        let payload = serde_json::json!({
            "observation_id": id,
            "target": target,
            "value": value,
            "timestamp_sec": timestamp_sec,
            "notes": notes
        });
        self.conn.execute(
            "INSERT INTO event_log (topic, target, payload_json, timestamp_sec) VALUES (?1, ?2, ?3, ?4)",
            params!["observation.recorded", target, payload.to_string(), timestamp_sec],
        )?;

        Ok(id)
    }

    /// Registra l'esito reale (Ground Truth) nel database SQLite.
    pub fn insert_outcome(
        &self,
        target: &str,
        occurred: bool,
        timestamp_sec: i64,
        brier: Option<f64>,
        log_loss: Option<f64>,
    ) -> rusqlite::Result<i64> {
        self.conn.execute(
            "INSERT INTO outcomes (target, occurred, timestamp_sec, brier_score, log_loss) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![target, if occurred { 1 } else { 0 }, timestamp_sec, brier, log_loss],
        )?;
        let id = self.conn.last_insert_rowid();

        // Registra nell'event_log
        let payload = serde_json::json!({
            "outcome_id": id,
            "target": target,
            "occurred": occurred,
            "timestamp_sec": timestamp_sec,
            "brier_score": brier,
            "log_loss": log_loss
        });
        self.conn.execute(
            "INSERT INTO event_log (topic, target, payload_json, timestamp_sec) VALUES (?1, ?2, ?3, ?4)",
            params!["outcome.verified", target, payload.to_string(), timestamp_sec],
        )?;

        Ok(id)
    }

    /// Popola il database SQLite a partire da uno store JSON (sincronizzazione iniziale).
    pub fn sync_from_store(&self, store: &MakimaStore) -> rusqlite::Result<()> {
        let count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM observations", [], |row| row.get(0))?;

        if count == 0 {
            for obs in &store.observations {
                self.insert_observation(&obs.target, obs.value, obs.timestamp_sec, None)?;
            }
            for out in &store.outcomes {
                self.insert_outcome(&out.target, out.occurred, out.timestamp_sec, None, None)?;
            }
            for f in &store.forecasts {
                let (status_str, outcome, brier, bucket, res_ts) = match &f.status {
                    ForecastStatus::Pending => ("PENDING", None, None, None, None),
                    ForecastStatus::Resolved {
                        actual,
                        brier_score,
                        calibration_bucket,
                        resolved_at_sec,
                    } => (
                        "RESOLVED",
                        Some(if *actual { 1 } else { 0 }),
                        Some(*brier_score),
                        Some(calibration_bucket.as_str()),
                        Some(*resolved_at_sec),
                    ),
                };
                let _ = self.conn.execute(
                    "INSERT INTO forecast_ledger (id, target, created_at_sec, probability, window_desc, evidence_count, model_name, status, outcome, brier_score, calibration_bucket, resolved_at_sec) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                    params![
                        f.id.0 as i64,
                        f.target,
                        f.created_at_sec,
                        f.probability.value(),
                        f.window_desc,
                        f.evidence_count as i64,
                        f.model_name,
                        status_str,
                        outcome,
                        brier,
                        bucket,
                        res_ts,
                    ],
                );
            }
        }
        Ok(())
    }

    /// Carica tutte le osservazioni, esiti e ledger dal database SQLite all'interno di [`MakimaEngine`].
    pub fn load_into_engine(&self, engine: &mut MakimaEngine) -> rusqlite::Result<()> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, target, value, timestamp_sec FROM observations ORDER BY id ASC")?;

        let obs_iter = stmt.query_map([], |row| {
            Ok(Observation::new(
                ObservationId(row.get(0)?),
                row.get::<_, String>(1)?,
                row.get(3)?,
                row.get(2)?,
            ))
        })?;

        for obs in obs_iter {
            engine.record_observation(obs?);
        }

        let mut out_stmt = self
            .conn
            .prepare("SELECT target, occurred, timestamp_sec FROM outcomes ORDER BY id ASC")?;

        let out_iter = out_stmt.query_map([], |row| {
            Ok(Outcome::new(
                row.get::<_, String>(0)?,
                row.get::<_, i32>(1)? != 0,
                row.get(2)?,
            ))
        })?;

        for out in out_iter {
            let o = out?;
            engine.record_outcome(&o.target, o.occurred, o.timestamp_sec);
        }

        let mut f_stmt = self
            .conn
            .prepare("SELECT id, target, created_at_sec, probability, window_desc, evidence_count, model_name, status, outcome, brier_score, calibration_bucket, resolved_at_sec FROM forecast_ledger ORDER BY id ASC")?;

        let f_iter = f_stmt.query_map([], |row| {
            let id: u64 = row.get(0)?;
            let target: String = row.get(1)?;
            let created_at_sec: i64 = row.get(2)?;
            let prob_val: f64 = row.get(3)?;
            let window_desc: String = row.get(4)?;
            let evidence_count: usize = row.get(5)?;
            let model_name: String = row.get(6)?;
            let status_str: String = row.get(7)?;

            let status = if status_str == "RESOLVED" {
                let actual = row.get::<_, Option<i32>>(8)?.unwrap_or(0) != 0;
                let brier_score = row.get::<_, Option<f64>>(9)?.unwrap_or(0.0);
                let calibration_bucket = row.get::<_, Option<String>>(10)?.unwrap_or_default();
                let resolved_at_sec = row.get::<_, Option<i64>>(11)?.unwrap_or(0);
                ForecastStatus::Resolved {
                    actual,
                    brier_score,
                    calibration_bucket,
                    resolved_at_sec,
                }
            } else {
                ForecastStatus::Pending
            };

            Ok(ForecastRecord {
                id: ForecastId(id),
                target,
                created_at_sec,
                probability: Probability::from_clamped(prob_val),
                window_desc,
                evidence_count,
                model_name,
                status,
            })
        })?;

        for rec in f_iter.flatten() {
            let exists = engine.ledger().records().iter().any(|r| r.id == rec.id);
            if !exists {
                // Inject or update
                let _ = engine.register_forecast_in_ledger(
                    rec.target,
                    rec.created_at_sec,
                    rec.probability,
                    rec.window_desc,
                    rec.evidence_count,
                    rec.model_name,
                );
            }
        }

        Ok(())
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

    #[test]
    fn test_sqlite_db_in_memory() {
        let db = MakimaDb::open(":memory:").expect("apertura db in memoria");
        let obs_id = db
            .insert_observation("test_target", 1.0, 1_700_000_000, Some("nota test"))
            .expect("inserimento osservazione");
        assert_eq!(obs_id, 1);

        let out_id = db
            .insert_outcome("test_target", true, 1_700_100_000, Some(0.04), Some(0.1))
            .expect("inserimento esito");
        assert_eq!(out_id, 1);

        let mut engine = MakimaEngine::new();
        db.load_into_engine(&mut engine)
            .expect("caricamento in engine");

        assert_eq!(engine.observations().len(), 1);
        assert_eq!(engine.outcomes().len(), 1);
    }
}
