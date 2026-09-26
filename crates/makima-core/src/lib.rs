//! # Makima Core
//!
//! `makima-core` costituisce il motore computazionale e probabilistico di Makima.
//!
//! Questo crate ospita il dominio fondamentale per la raccolta di evidenze empiriche,
//! la modellazione statistica interpretabile, la quantificazione dell'incertezza,
//! l'esecuzione di simulazioni previsionali e la valutazione della calibrazione.
pub mod eval;
pub mod forecast;
pub mod laplace;
pub mod prob;
pub mod storage;

pub use eval::{CalibrationBin, CalibrationRating, EvaluationReport, Evaluator, Outcome, Scoring};
pub use forecast::{
    Forecast, ForecastError, ForecastId, ForecastLedger, ForecastRecord, ForecastStatus,
};
pub use laplace::{LaplaceMail, TargetSummary};
pub use prob::{
    Bernoulli, BetaDistribution, ContinuousDistribution, DiscreteDistribution, Distribution,
    PoissonDistribution, ProbError, Probability,
};
pub use storage::{MakimaDb, MakimaStore};

use serde::{Deserialize, Serialize};
use std::fmt;

/// Stato operativo dell'istanza del motore Makima.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum EngineState {
    /// Il motore è inizializzato e pronto a ricevere osservazioni o richieste di previsione.
    #[default]
    Ready,
    /// Il motore sta elaborando o aggiornando i modelli statistici interni.
    Calibrating,
}

impl fmt::Display for EngineState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ready => write!(f, "ready"),
            Self::Calibrating => write!(f, "calibrating"),
        }
    }
}

/// Riepilogo sullo stato operativo e diagnostico del motore.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineStatus {
    /// Versione del core engine.
    pub version: &'static str,
    /// Stato corrente di funzionamento.
    pub state: EngineState,
    /// Numero totale di osservazioni storiche registrate.
    pub total_observations: usize,
    /// Numero totale di esiti reali registrati per la valutazione.
    pub total_outcomes: usize,
    /// Numero di previsioni totali nel ledger.
    pub total_forecasts: usize,
    /// Numero di previsioni pendenti in attesa di verifica.
    pub pending_forecasts: usize,
}

/// Identificativo univoco per una singola osservazione empirica.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ObservationId(pub u64);

/// Rappresentazione di una singola evidenza o osservazione registrata nel sistema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    /// Identificativo dell'osservazione.
    pub id: ObservationId,
    /// Etichetta semantica del target o dell'evento osservato.
    pub target: String,
    /// Timestamp Unix (in secondi) in cui l'osservazione è avvenuta o è stata registrata.
    pub timestamp_sec: i64,
    /// Valore numerico o scalare associato all'osservazione.
    pub value: f64,
}

impl Observation {
    /// Crea una nuova osservazione empirica.
    #[must_use]
    pub fn new(
        id: ObservationId,
        target: impl Into<String>,
        timestamp_sec: i64,
        value: f64,
    ) -> Self {
        Self {
            id,
            target: target.into(),
            timestamp_sec,
            value,
        }
    }
}

/// Motore principale di previsione probabilistica e valutazione.
///
/// Gestisce il ciclo di vita delle evidenze storiche, lo stato dei modelli,
/// la generazione di previsioni calibrate e la misurazione continua degli errori.
#[derive(Debug, Default)]
pub struct MakimaEngine {
    state: EngineState,
    observations: Vec<Observation>,
    outcomes: Vec<Outcome>,
    evaluator: Evaluator,
    ledger: ForecastLedger,
}

impl MakimaEngine {
    /// Crea una nuova istanza di [`MakimaEngine`] pronta all'uso.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Restituisce la versione di `makima-core`.
    #[must_use]
    pub fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    /// Restituisce lo stato diagnostico corrente del motore.
    #[must_use]
    pub fn status(&self) -> EngineStatus {
        EngineStatus {
            version: self.version(),
            state: self.state,
            total_observations: self.observations.len(),
            total_outcomes: self.outcomes.len(),
            total_forecasts: self.ledger.total_count(),
            pending_forecasts: self.ledger.pending_count(),
        }
    }

    /// Registra una nuova osservazione empirica nel motore.
    pub fn record_observation(&mut self, observation: Observation) {
        self.observations.push(observation);
    }

    /// Registra un'osservazione binaria (successo/fallimento) con timestamp e ID autoincrementale.
    pub fn record_binary(
        &mut self,
        target: impl Into<String>,
        success: bool,
        timestamp_sec: i64,
    ) -> ObservationId {
        let id = ObservationId((self.observations.len() + 1) as u64);
        let obs = Observation::new(id, target, timestamp_sec, if success { 1.0 } else { 0.0 });
        self.record_observation(obs);
        id
    }

    /// Registra l'esito reale verificatosi per un target, associandolo all'ultima previsione emessa.
    pub fn record_outcome(
        &mut self,
        target: impl Into<String>,
        occurred: bool,
        timestamp_sec: i64,
    ) {
        let target_str = target.into();
        // Calcola la previsione che il modello avrebbe fatto prima dell'esito
        let forecast = self.predict_target(&target_str);
        self.evaluator
            .add_prediction_outcome(forecast.probability, occurred);
        self.ledger
            .resolve_for_target(&target_str, occurred, timestamp_sec);
        self.outcomes
            .push(Outcome::new(target_str, occurred, timestamp_sec));
    }

    /// Risolve una singola previsione univoca specificata tramite `ForecastId` registrando l'esito e valutando il Brier Score.
    pub fn resolve_forecast_by_id(
        &mut self,
        id: ForecastId,
        occurred: bool,
        timestamp_sec: i64,
        ground_truth_event_id: Option<u64>,
    ) -> bool {
        let record_opt = self.ledger.records().iter().find(|r| r.id == id).cloned();
        if let Some(rec) = record_opt {
            if matches!(rec.status, ForecastStatus::Pending) {
                self.evaluator
                    .add_prediction_outcome(rec.probability, occurred);
                let resolved = self.ledger.resolve_by_id_with_event(
                    id,
                    occurred,
                    timestamp_sec,
                    ground_truth_event_id,
                );
                self.outcomes
                    .push(Outcome::new(rec.target, occurred, timestamp_sec));
                return resolved;
            }
        }
        false
    }

    /// Registra formalmente una nuova previsione nel ledger con stato `Pending`.
    pub fn register_forecast_in_ledger(
        &mut self,
        target: impl Into<String>,
        created_at_sec: i64,
        probability: Probability,
        window_desc: impl Into<String>,
        evidence_count: usize,
        model_name: impl Into<String>,
    ) -> ForecastId {
        self.ledger.register_forecast(
            target,
            created_at_sec,
            probability,
            window_desc,
            evidence_count,
            model_name,
        )
    }

    /// Restituisce la lista di osservazioni storiche attualmente caricate.
    #[must_use]
    pub fn observations(&self) -> &[Observation] {
        &self.observations
    }

    /// Restituisce la lista di esiti reali registrati.
    #[must_use]
    pub fn outcomes(&self) -> &[Outcome] {
        &self.outcomes
    }

    /// Restituisce il riferimento al ledger di tracciamento delle previsioni.
    #[must_use]
    pub fn ledger(&self) -> &ForecastLedger {
        &self.ledger
    }

    /// Restituisce il riferimento mutabile al ledger.
    pub fn ledger_mut(&mut self) -> &mut ForecastLedger {
        &mut self.ledger
    }

    /// Genera una stima probabilistica per il target specificato utilizzando un prior uniforme.
    #[must_use]
    pub fn predict_target(&self, target: &str) -> Forecast {
        self.predict_target_with_prior(target, BetaDistribution::uniform())
    }

    /// Genera una stima probabilistica per il target specificato a partire da un prior esplicito.
    #[must_use]
    pub fn predict_target_with_prior(&self, target: &str, prior: BetaDistribution) -> Forecast {
        Forecast::from_observations(target, &self.observations, prior)
    }

    /// Restituisce l'elenco ordinato e univoco di tutti i target tracciati nelle osservazioni.
    #[must_use]
    pub fn tracked_targets(&self) -> Vec<String> {
        let mut set = std::collections::BTreeSet::new();
        for obs in &self.observations {
            set.insert(obs.target.clone());
        }
        set.into_iter().collect()
    }

    /// Calcola un riepilogo statistico e previsionale per un singolo target.
    #[must_use]
    pub fn summarize_target(&self, target: &str) -> TargetSummary {
        let forecast = self.predict_target(target);
        let mut successes = 0;
        let mut failures = 0;
        let mut last_ts = 0;
        let mut timestamps = Vec::new();

        for obs in &self.observations {
            if obs.target == target {
                if obs.value >= 0.5 {
                    successes += 1;
                } else {
                    failures += 1;
                }
                if obs.timestamp_sec > last_ts {
                    last_ts = obs.timestamp_sec;
                }
                timestamps.push(obs.timestamp_sec);
            }
        }

        let total = successes + failures;
        let daily_rate = if timestamps.len() >= 2 {
            let min_ts = *timestamps.iter().min().unwrap_or(&0);
            let max_ts = *timestamps.iter().max().unwrap_or(&0);
            let span_days = ((max_ts - min_ts) as f64 / 86400.0).max(1.0);
            timestamps.len() as f64 / span_days
        } else {
            (total as f64 / 30.0).max(0.05)
        };

        TargetSummary {
            target: target.to_string(),
            observations_count: total,
            success_count: successes,
            failure_count: failures,
            probability: forecast.probability,
            uncertainty_variance: forecast.uncertainty_variance,
            entropy_bits: forecast.entropy_bits,
            last_timestamp_sec: last_ts,
            estimated_daily_rate: daily_rate,
        }
    }

    /// Restituisce la lista di riassunti previsionali per tutti i target tracciati.
    #[must_use]
    pub fn target_summaries(&self) -> Vec<TargetSummary> {
        self.tracked_targets()
            .iter()
            .map(|t| self.summarize_target(t))
            .collect()
    }

    /// Genera un bollettino previsionale consolidato Laplace Mail.
    #[must_use]
    pub fn generate_laplace_mail(&self, timestamp_sec: i64) -> LaplaceMail {
        LaplaceMail::generate(self, timestamp_sec)
    }

    /// Genera il report di valutazione delle performance previsionali (Brier Score, Skill Score, ECE).
    #[must_use]
    pub fn evaluate_performance(&self) -> Option<EvaluationReport> {
        self.evaluator
            .evaluate_with_ledger(self.ledger.total_count(), self.ledger.pending_count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_initial_state() {
        let engine = MakimaEngine::new();
        let status = engine.status();

        assert_eq!(status.state, EngineState::Ready);
        assert_eq!(status.total_observations, 0);
        assert_eq!(status.total_outcomes, 0);
        assert_eq!(status.version, "0.1.0");
        assert!(engine.observations().is_empty());
    }

    #[test]
    fn test_record_observation() {
        let mut engine = MakimaEngine::new();
        let obs = Observation::new(ObservationId(1), "framework_release", 1_700_000_000, 1.0);

        engine.record_observation(obs.clone());

        let status = engine.status();
        assert_eq!(status.total_observations, 1);
        assert_eq!(engine.observations().len(), 1);
        assert_eq!(engine.observations()[0], obs);
    }

    #[test]
    fn test_record_binary_and_predict() {
        let mut engine = MakimaEngine::new();
        engine.record_binary("framework_release", true, 1_700_000_000);
        engine.record_binary("framework_release", true, 1_700_086_400);
        engine.record_binary("framework_release", false, 1_700_172_800);

        let forecast = engine.predict_target("framework_release");

        assert_eq!(forecast.evidence_count, 3);
        assert_eq!(forecast.evidence_ids.len(), 3);
        assert!((forecast.probability.value() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_engine_evaluation() {
        let mut engine = MakimaEngine::new();
        engine.record_binary("framework_release", true, 1_700_000_000);
        engine.record_binary("framework_release", true, 1_700_086_400);
        engine.record_outcome("framework_release", true, 1_700_172_800);

        let report = engine.evaluate_performance();
        assert!(report.is_some());
        let rep = report.expect("report presente");
        assert_eq!(rep.resolved_outcomes, 1);
        assert!(rep.mean_brier_score < 0.25);
    }

    #[test]
    fn test_engine_state_display() {
        assert_eq!(EngineState::Ready.to_string(), "ready");
        assert_eq!(EngineState::Calibrating.to_string(), "calibrating");
    }

    #[test]
    fn test_engine_multi_target_and_laplace_mail() {
        let mut engine = MakimaEngine::new();
        engine.record_binary("framework_release", true, 1_700_000_000);
        engine.record_binary("framework_release", true, 1_700_086_400);
        engine.record_binary("daily_build", true, 1_700_000_000);

        let targets = engine.tracked_targets();
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0], "daily_build");
        assert_eq!(targets[1], "framework_release");

        let summaries = engine.target_summaries();
        assert_eq!(summaries.len(), 2);

        let mail = engine.generate_laplace_mail(1_700_200_000);
        assert_eq!(mail.all_summaries.len(), 2);
        assert!(!mail.issue_id.is_empty());
    }
}
