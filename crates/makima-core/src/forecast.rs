//! Modulo di previsione e quantificazione dell'incertezza per Makima.

use crate::prob::{Bernoulli, BetaDistribution, Distribution, Probability};
use crate::{Observation, ObservationId};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Errori generabili durante il calcolo o l'elaborazione di una previsione.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForecastError {
    /// Nessuna evidenza storica trovata per il target specificato.
    NoObservationsFound(String),
    /// Errore generico nel calcolo del modello previsionale.
    ModelError(String),
}

impl fmt::Display for ForecastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoObservationsFound(target) => {
                write!(
                    f,
                    "nessuna osservazione storica registrata per il target '{target}'"
                )
            }
            Self::ModelError(msg) => {
                write!(f, "errore nel modello di previsione: {msg}")
            }
        }
    }
}

impl std::error::Error for ForecastError {}

/// Identificativo univoco per una previsione emessa.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ForecastId(pub u64);

impl fmt::Display for ForecastId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Stato del ciclo di vita di una previsione.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ForecastStatus {
    /// Previsione emessa ma l'esito reale non è ancora avvenuto o non è stato verificato.
    Pending,
    /// Previsione verificata a fronte del Ground Truth reale.
    Resolved {
        /// Esito reale osservato (true = successo, false = insuccesso).
        actual: bool,
        /// Brier score calcolato per questa singola previsione: $(p - y)^2$.
        brier_score: f64,
        /// Fascia di probabilità in cui ricadeva la stima (es. "70-80%").
        calibration_bucket: String,
        /// Timestamp Unix della risoluzione.
        resolved_at_sec: i64,
    },
}

impl fmt::Display for ForecastStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "PENDING"),
            Self::Resolved {
                actual,
                brier_score,
                calibration_bucket,
                ..
            } => {
                write!(
                    f,
                    "RESOLVED (outcome={}, brier={:.4}, bucket={})",
                    if *actual { "TRUE" } else { "FALSE" },
                    brier_score,
                    calibration_bucket
                )
            }
        }
    }
}

/// Scheda formale di una previsione con identità e tracciamento del ciclo di vita.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastRecord {
    /// Identificativo univoco della previsione (es. `#184`).
    pub id: ForecastId,
    /// Target o sistema oggetto della stima.
    pub target: String,
    /// Timestamp Unix di emissione.
    pub created_at_sec: i64,
    /// Probabilità prevista $P(E)$.
    pub probability: Probability,
    /// Descrizione dell'orizzonte temporale associato (es. "7 days", "relative").
    pub window_desc: String,
    /// Numero di evidenze storiche impiegate nel calcolo.
    pub evidence_count: usize,
    /// Composizione e architettura del modello impiegato.
    pub model_name: String,
    /// Stato corrente del ciclo di vita (`Pending` o `Resolved`).
    pub status: ForecastStatus,
}

impl ForecastRecord {
    /// Crea un nuovo record di previsione nello stato `Pending`.
    #[must_use]
    pub fn new_pending(
        id: ForecastId,
        target: impl Into<String>,
        created_at_sec: i64,
        probability: Probability,
        window_desc: impl Into<String>,
        evidence_count: usize,
        model_name: impl Into<String>,
    ) -> Self {
        Self {
            id,
            target: target.into(),
            created_at_sec,
            probability,
            window_desc: window_desc.into(),
            evidence_count,
            model_name: model_name.into(),
            status: ForecastStatus::Pending,
        }
    }

    /// Risolve la previsione confrontandola con l'esito reale osservato.
    pub fn resolve(&mut self, actual: bool, resolved_at_sec: i64) {
        let p = self.probability.value();
        let y = if actual { 1.0 } else { 0.0 };
        let brier = (p - y) * (p - y);

        let bucket_low = (p * 10.0).floor() as usize * 10;
        let bucket_high = (bucket_low + 10).min(100);
        let bucket = format!("{bucket_low}-{bucket_high}%");

        self.status = ForecastStatus::Resolved {
            actual,
            brier_score: brier,
            calibration_bucket: bucket,
            resolved_at_sec,
        };
    }
}

/// Registro e ledger di tutte le previsioni emesse nel ciclo di vita di Makima.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastLedger {
    records: Vec<ForecastRecord>,
}

impl ForecastLedger {
    /// Crea un ledger vuoto.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra una nuova previsione e assegna un ID monotonico incrementale.
    pub fn register_forecast(
        &mut self,
        target: impl Into<String>,
        created_at_sec: i64,
        probability: Probability,
        window_desc: impl Into<String>,
        evidence_count: usize,
        model_name: impl Into<String>,
    ) -> ForecastId {
        let id = ForecastId((self.records.len() + 1) as u64);
        let record = ForecastRecord::new_pending(
            id,
            target,
            created_at_sec,
            probability,
            window_desc,
            evidence_count,
            model_name,
        );
        self.records.push(record);
        id
    }

    /// Risolve tutte le previsioni pendenti per un target specifico a fronte di un nuovo esito reale.
    pub fn resolve_for_target(
        &mut self,
        target: &str,
        actual: bool,
        resolved_at_sec: i64,
    ) -> usize {
        let mut count = 0;
        for rec in &mut self.records {
            if rec.target == target && rec.status == ForecastStatus::Pending {
                rec.resolve(actual, resolved_at_sec);
                count += 1;
            }
        }
        count
    }

    /// Restituisce il numero totale di previsioni registrate.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.records.len()
    }

    /// Restituisce il numero di previsioni ancora in attesa di verifica (`Pending`).
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.records
            .iter()
            .filter(|r| matches!(r.status, ForecastStatus::Pending))
            .count()
    }

    /// Restituisce il numero di previsioni verificate e risolte (`Resolved`).
    #[must_use]
    pub fn resolved_count(&self) -> usize {
        self.records
            .iter()
            .filter(|r| matches!(r.status, ForecastStatus::Resolved { .. }))
            .count()
    }

    /// Restituisce il riferimento a tutti i record del ledger.
    #[must_use]
    pub fn records(&self) -> &[ForecastRecord] {
        &self.records
    }
}

/// Risultato strutturato e interpretabile di una previsione probabilistica.
#[derive(Debug, Clone, PartialEq)]
pub struct Forecast {
    /// Nome del target o evento oggetto della stima.
    pub target: String,
    /// Valore atteso della probabilità di accadimento: $\mathbb{E}[P(E)]$.
    pub probability: Probability,
    /// Distribuzione a priori (*Prior*) utilizzata nel calcolo.
    pub prior: BetaDistribution,
    /// Distribuzione a posteriori (*Posterior*) risultante dall'aggiornamento bayesiano.
    pub posterior: BetaDistribution,
    /// Incertezza epistemica residua misurata tramite varianza a posteriori: $\text{Var}(\theta)$.
    pub uncertainty_variance: f64,
    /// Entropia di Shannon in bit della stima puntuale.
    pub entropy_bits: f64,
    /// Conteggio totale delle evidenze osservate per questo target.
    pub evidence_count: usize,
    /// Identificativi univoci esatti delle osservazioni utilizzate come prova (*Evidence*).
    pub evidence_ids: Vec<ObservationId>,
}

impl Forecast {
    /// Costruisce una previsione a partire da un set di evidenze e un prior bayesiano.
    #[must_use]
    pub fn from_observations(
        target: impl Into<String>,
        observations: &[Observation],
        prior: BetaDistribution,
    ) -> Self {
        let target_str = target.into();
        let mut successes = 0;
        let mut failures = 0;
        let mut evidence_ids = Vec::with_capacity(observations.len());

        for obs in observations {
            if obs.target == target_str {
                evidence_ids.push(obs.id);
                if obs.value > 0.5 {
                    successes += 1;
                } else {
                    failures += 1;
                }
            }
        }

        let posterior = prior.bayesian_update(successes, failures);
        let prob_val = Probability::from_clamped(posterior.mean());
        let bernoulli_estimate = Bernoulli::from_probability(prob_val);

        Self {
            target: target_str,
            probability: prob_val,
            prior,
            posterior,
            uncertainty_variance: posterior.variance(),
            entropy_bits: bernoulli_estimate.entropy_bits(),
            evidence_count: evidence_ids.len(),
            evidence_ids,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forecast_with_no_observations_uses_prior() {
        let prior = BetaDistribution::uniform();
        let forecast = Forecast::from_observations("framework_release", &[], prior);

        assert_eq!(forecast.target, "framework_release");
        assert_eq!(forecast.probability.value(), 0.5);
        assert_eq!(forecast.evidence_count, 0);
        assert!(forecast.evidence_ids.is_empty());
        assert!((forecast.uncertainty_variance - (1.0 / 12.0)).abs() < 1e-9);
        assert!((forecast.entropy_bits - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_forecast_with_observations_reduces_uncertainty() {
        let prior = BetaDistribution::uniform();
        let observations = vec![
            Observation::new(ObservationId(1), "framework_release", 1_700_000_000, 1.0),
            Observation::new(ObservationId(2), "framework_release", 1_700_086_400, 1.0),
            Observation::new(ObservationId(3), "framework_release", 1_700_172_800, 0.0),
        ];

        let forecast = Forecast::from_observations("framework_release", &observations, prior);

        assert_eq!(forecast.evidence_count, 3);
        assert_eq!(
            forecast.evidence_ids,
            vec![ObservationId(1), ObservationId(2), ObservationId(3)]
        );
        // Prior Beta(1, 1) + 2 successi, 1 fallimento -> Posterior Beta(3, 2)
        // Media = 3 / (3 + 2) = 0.6
        assert!((forecast.probability.value() - 0.6).abs() < 1e-9);
        // Varianza (3*2)/(25*6) = 6/150 = 0.04 < 0.0833 (incertezza calata)
        assert!(forecast.uncertainty_variance < (1.0 / 12.0));
    }
}
