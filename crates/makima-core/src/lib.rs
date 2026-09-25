//! # Makima Core
//!
//! `makima-core` costituisce il motore computazionale e probabilistico di Makima.
//!
//! Questo crate ospita il dominio fondamentale per la raccolta di evidenze empiriche,
//! la modellazione statistica interpretabile, la quantificazione dell'incertezza
//! e l'esecuzione di simulazioni previsionali.
pub mod forecast;
pub mod prob;

pub use forecast::{Forecast, ForecastError};
pub use prob::{
    Bernoulli, BetaDistribution, ContinuousDistribution, DiscreteDistribution, Distribution,
    PoissonDistribution, ProbError, Probability,
};

use std::fmt;

/// Stato operativo dell'istanza del motore Makima.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineStatus {
    /// Versione del core engine.
    pub version: &'static str,
    /// Stato corrente di funzionamento.
    pub state: EngineState,
    /// Numero totale di osservazioni storiche registrate.
    pub total_observations: usize,
}

/// Identificativo univoco per una singola osservazione empirica.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObservationId(pub u64);

/// Rappresentazione di una singola evidenza o osservazione registrata nel sistema.
#[derive(Debug, Clone, PartialEq)]
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

/// Motore principale di previsione probabilistica.
///
/// Gestisce il ciclo di vita delle evidenze storiche, lo stato dei modelli e
/// la generazione di previsioni calibrate.
#[derive(Debug, Default)]
pub struct MakimaEngine {
    state: EngineState,
    observations: Vec<Observation>,
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

    /// Restituisce la lista di osservazioni storiche attualmente caricate.
    #[must_use]
    pub fn observations(&self) -> &[Observation] {
        &self.observations
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
    fn test_engine_state_display() {
        assert_eq!(EngineState::Ready.to_string(), "ready");
        assert_eq!(EngineState::Calibrating.to_string(), "calibrating");
    }
}
