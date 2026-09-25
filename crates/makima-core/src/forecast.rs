//! Modulo di previsione e quantificazione dell'incertezza per Makima.

use crate::prob::{Bernoulli, BetaDistribution, Distribution, Probability};
use crate::{Observation, ObservationId};
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
