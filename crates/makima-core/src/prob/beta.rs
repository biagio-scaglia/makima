//! Distribuzione Beta ($\text{Beta}(\alpha, \beta)$) e prior coniugato bayesiano.

use super::error::ProbError;
use super::traits::Distribution;

/// Distribuzione di probabilità Beta definita su $[0, 1]$ con parametri di forma $\alpha > 0, \beta > 0$.
///
/// Funge da prior coniugato naturale per i processi bernoulliani e binomiali,
/// permettendo l'aggiornamento bayesiano analitico e sequenziale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaDistribution {
    alpha: f64,
    beta: f64,
}

impl BetaDistribution {
    /// Crea una nuova distribuzione $\text{Beta}(\alpha, \beta)$.
    ///
    /// # Errori
    /// Restituisce [`ProbError::InvalidParameter`] se $\alpha \le 0$ o $\beta \le 0$.
    pub fn new(alpha: f64, beta: f64) -> Result<Self, ProbError> {
        if alpha.is_nan() || alpha <= 0.0 {
            return Err(ProbError::InvalidParameter {
                parameter: "alpha",
                value: alpha,
                reason: "il parametro alpha deve essere strettamente positivo (> 0.0)",
            });
        }
        if beta.is_nan() || beta <= 0.0 {
            return Err(ProbError::InvalidParameter {
                parameter: "beta",
                value: beta,
                reason: "il parametro beta deve essere strettamente positivo (> 0.0)",
            });
        }
        Ok(Self { alpha, beta })
    }

    /// Crea un prior non-informativo uniforme $\text{Beta}(1.0, 1.0)$.
    #[must_use]
    pub fn uniform() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Crea il prior non-informativo di Jeffreys $\text{Beta}(0.5, 0.5)$.
    #[must_use]
    pub fn jeffreys() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.5,
        }
    }

    /// Restituisce il parametro di forma $\alpha$.
    #[must_use]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Restituisce il parametro di forma $\beta$.
    #[must_use]
    pub const fn beta(&self) -> f64 {
        self.beta
    }

    /// Esegue l'aggiornamento bayesiano coniugato osservando $k$ successi e $n-k$ fallimenti.
    ///
    /// La distribuzione a posteriori sarà: $\text{Beta}(\alpha + \text{successes}, \beta + \text{failures})$.
    #[must_use]
    pub fn bayesian_update(&self, successes: usize, failures: usize) -> Self {
        Self {
            alpha: self.alpha + successes as f64,
            beta: self.beta + failures as f64,
        }
    }

    /// Restituisce la moda della distribuzione se $\alpha > 1$ e $\beta > 1$.
    #[must_use]
    pub fn mode(&self) -> Option<f64> {
        if self.alpha > 1.0 && self.beta > 1.0 {
            Some((self.alpha - 1.0) / (self.alpha + self.beta - 2.0))
        } else {
            None
        }
    }
}

impl Distribution for BetaDistribution {
    fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    fn variance(&self) -> f64 {
        let sum = self.alpha + self.beta;
        (self.alpha * self.beta) / (sum * sum * (sum + 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_beta() {
        let beta = BetaDistribution::uniform();
        assert_eq!(beta.mean(), 0.5);
        assert!((beta.variance() - 1.0 / 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_bayesian_update() {
        let prior = BetaDistribution::uniform(); // Beta(1, 1) -> mean 0.5
        let posterior = prior.bayesian_update(7, 3); // Beta(8, 4)

        assert_eq!(posterior.alpha(), 8.0);
        assert_eq!(posterior.beta(), 4.0);
        // Media a posteriori: 8 / (8 + 4) = 8/12 = 0.6666...
        assert!((posterior.mean() - (8.0 / 12.0)).abs() < 1e-9);

        // Moda: (8 - 1) / (8 + 4 - 2) = 7 / 10 = 0.7
        assert_eq!(posterior.mode(), Some(0.7));
    }

    #[test]
    fn test_invalid_parameters() {
        assert!(BetaDistribution::new(0.0, 1.0).is_err());
        assert!(BetaDistribution::new(1.0, -0.5).is_err());
    }
}
