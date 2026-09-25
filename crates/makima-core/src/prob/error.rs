//! Tipi di errore per le operazioni probabilistiche e statistiche.

use std::fmt;

/// Errori derivanti da parametri o calcoli probabilistici non validi.
#[derive(Debug, Clone, PartialEq)]
pub enum ProbError {
    /// Il valore di probabilità non rientra nell'intervallo chiuso [0.0, 1.0].
    InvalidProbability(f64),
    /// I parametri di forma o scala della distribuzione violano i vincoli di positività o validità.
    InvalidParameter {
        parameter: &'static str,
        value: f64,
        reason: &'static str,
    },
    /// Errore generico nel calcolo numerico (ad es. divisione per zero o overflow).
    CalculationError(&'static str),
}

impl fmt::Display for ProbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProbability(val) => {
                write!(
                    f,
                    "valore di probabilità non valido ({val}): deve essere in [0.0, 1.0]"
                )
            }
            Self::InvalidParameter {
                parameter,
                value,
                reason,
            } => {
                write!(f, "parametro '{parameter}' non valido ({value}): {reason}")
            }
            Self::CalculationError(msg) => {
                write!(f, "errore nel calcolo probabilistico: {msg}")
            }
        }
    }
}

impl std::error::Error for ProbError {}
