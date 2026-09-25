//! Tipo di dato tipizzato e sicuro per rappresentare valori di probabilità in [0.0, 1.0].

use super::error::ProbError;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Rappresenta un valore di probabilità matematicamente valido, garantito nell'intervallo [0.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Probability(f64);

impl Probability {
    /// Probabilità minima (evento impossibile: 0.0).
    pub const ZERO: Self = Self(0.0);

    /// Probabilità massima (evento certo: 1.0).
    pub const ONE: Self = Self(1.0);

    /// Crea un nuovo valore di [`Probability`].
    ///
    /// # Errori
    /// Restituisce [`ProbError::InvalidProbability`] se `value` è minore di 0.0, maggiore di 1.0, o `NaN`.
    pub fn new(value: f64) -> Result<Self, ProbError> {
        if value.is_nan() || !(0.0..=1.0).contains(&value) {
            Err(ProbError::InvalidProbability(value))
        } else {
            Ok(Self(value))
        }
    }

    /// Crea una probabilità assicurando che il valore sia bloccato (*clamped*) in [0.0, 1.0].
    /// In caso di `NaN`, restituisce `Probability::ZERO`.
    #[must_use]
    pub fn from_clamped(value: f64) -> Self {
        if value.is_nan() {
            Self::ZERO
        } else {
            Self(value.clamp(0.0, 1.0))
        }
    }

    /// Restituisce il valore scalare `f64` della probabilità.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.0
    }

    /// Calcola il complemento dell'evento: $P(A^c) = 1 - P(A)$.
    #[must_use]
    pub fn complement(self) -> Self {
        Self((1.0 - self.0).clamp(0.0, 1.0))
    }
}

impl fmt::Display for Probability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_probability() {
        let p = Probability::new(0.75).expect("valore valido");
        assert_eq!(p.value(), 0.75);
        assert_eq!(p.complement().value(), 0.25);
    }

    #[test]
    fn test_invalid_probability() {
        assert!(Probability::new(-0.01).is_err());
        assert!(Probability::new(1.05).is_err());
        assert!(Probability::new(f64::NAN).is_err());
    }

    #[test]
    fn test_clamped_probability() {
        assert_eq!(Probability::from_clamped(1.5).value(), 1.0);
        assert_eq!(Probability::from_clamped(-0.5).value(), 0.0);
        assert_eq!(Probability::from_clamped(f64::NAN).value(), 0.0);
    }
}
