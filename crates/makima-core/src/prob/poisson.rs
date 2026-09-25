//! Distribuzione di Poisson ($\text{Poisson}(\lambda)$) per il conteggio di eventi nel tempo.

use super::error::ProbError;
use super::probability::Probability;
use super::traits::{DiscreteDistribution, Distribution};

/// Distribuzione di Poisson per la frequenza di accadimento di eventi con tasso atteso $\lambda > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoissonDistribution {
    lambda: f64,
}

impl PoissonDistribution {
    /// Crea una nuova distribuzione $\text{Poisson}(\lambda)$.
    ///
    /// # Errori
    /// Restituisce [`ProbError::InvalidParameter`] se $\lambda \le 0.0$ o `NaN`.
    pub fn new(lambda: f64) -> Result<Self, ProbError> {
        if lambda.is_nan() || lambda <= 0.0 {
            return Err(ProbError::InvalidParameter {
                parameter: "lambda",
                value: lambda,
                reason: "il parametro di tasso lambda deve essere strettamente positivo (> 0.0)",
            });
        }
        Ok(Self { lambda })
    }

    /// Restituisce il tasso atteso $\lambda$.
    #[must_use]
    pub const fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Calcola il logaritmo del fattoriale $\ln(k!)$ per evitare overflow con $k$ elevati.
    fn ln_factorial(k: u64) -> f64 {
        let mut sum = 0.0;
        for i in 2..=k {
            sum += (i as f64).ln();
        }
        sum
    }
}

impl Distribution for PoissonDistribution {
    fn mean(&self) -> f64 {
        self.lambda
    }

    fn variance(&self) -> f64 {
        self.lambda
    }
}

impl DiscreteDistribution for PoissonDistribution {
    fn pmf(&self, k: u64) -> Probability {
        // P(X = k) = exp(k * ln(lambda) - lambda - ln(k!))
        let log_pmf = (k as f64) * self.lambda.ln() - self.lambda - Self::ln_factorial(k);
        let prob_val = log_pmf.exp();
        Probability::from_clamped(prob_val)
    }

    fn cdf(&self, k: u64) -> Probability {
        let mut cumulative = 0.0;
        for i in 0..=k {
            cumulative += self.pmf(i).value();
        }
        Probability::from_clamped(cumulative)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_moments() {
        let p = PoissonDistribution::new(4.0).expect("valido");
        assert_eq!(p.mean(), 4.0);
        assert_eq!(p.variance(), 4.0);
        assert_eq!(p.std_dev(), 2.0);
    }

    #[test]
    fn test_poisson_pmf() {
        // Con lambda = 2.0:
        // P(X=0) = e^(-2) ~= 0.135335
        // P(X=1) = 2 * e^(-2) ~= 0.27067
        // P(X=2) = 2^2 / 2 * e^(-2) ~= 0.27067
        let p = PoissonDistribution::new(2.0).expect("valido");
        assert!((p.pmf(0).value() - 0.135335).abs() < 1e-4);
        assert!((p.pmf(1).value() - 0.27067).abs() < 1e-4);
        assert!((p.pmf(2).value() - 0.27067).abs() < 1e-4);
    }

    #[test]
    fn test_poisson_cdf_monotonic() {
        let p = PoissonDistribution::new(3.0).expect("valido");
        let cdf_0 = p.cdf(0).value();
        let cdf_3 = p.cdf(3).value();
        let cdf_10 = p.cdf(10).value();

        assert!(cdf_0 <= cdf_3);
        assert!(cdf_3 <= cdf_10);
        assert!(cdf_10 <= 1.0);
    }
}
