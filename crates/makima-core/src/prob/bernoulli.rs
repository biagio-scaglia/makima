//! Distribuzione di Bernoulli per eventi binari ($X \in \{0, 1\}$).

use super::error::ProbError;
use super::probability::Probability;
use super::traits::{DiscreteDistribution, Distribution};

/// Distribuzione di Bernoulli parametrizzata dalla probabilità di successo $p \in [0, 1]$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bernoulli {
    p: Probability,
}

impl Bernoulli {
    /// Crea una nuova distribuzione di Bernoulli dato il parametro $p$.
    ///
    /// # Errori
    /// Restituisce [`ProbError::InvalidProbability`] se $p \notin [0, 1]$.
    pub fn new(p: f64) -> Result<Self, ProbError> {
        let prob = Probability::new(p)?;
        Ok(Self { p: prob })
    }

    /// Crea una distribuzione di Bernoulli direttamente da un'istanza di [`Probability`].
    #[must_use]
    pub const fn from_probability(p: Probability) -> Self {
        Self { p }
    }

    /// Restituisce la probabilità di successo $p = P(X = 1)$.
    #[must_use]
    pub const fn p(&self) -> Probability {
        self.p
    }

    /// Restituisce la probabilità di insuccesso $q = 1 - p = P(X = 0)$.
    #[must_use]
    pub fn q(&self) -> Probability {
        self.p.complement()
    }

    /// Calcola l'entropia di Shannon in bit (base 2):
    /// $H(X) = -p \log_2(p) - (1-p) \log_2(1-p)$
    #[must_use]
    pub fn entropy_bits(&self) -> f64 {
        let p_val = self.p.value();
        let q_val = self.q().value();

        let term_p = if p_val > 0.0 {
            p_val * p_val.log2()
        } else {
            0.0
        };
        let term_q = if q_val > 0.0 {
            q_val * q_val.log2()
        } else {
            0.0
        };

        -(term_p + term_q)
    }
}

impl Distribution for Bernoulli {
    fn mean(&self) -> f64 {
        self.p.value()
    }

    fn variance(&self) -> f64 {
        self.p.value() * self.q().value()
    }
}

impl DiscreteDistribution for Bernoulli {
    fn pmf(&self, k: u64) -> Probability {
        match k {
            0 => self.q(),
            1 => self.p,
            _ => Probability::ZERO,
        }
    }

    fn cdf(&self, k: u64) -> Probability {
        if k == 0 {
            self.q()
        } else {
            Probability::ONE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bernoulli_moments() {
        let b = Bernoulli::new(0.7).expect("valido");
        assert!((b.mean() - 0.7).abs() < 1e-9);
        assert!((b.variance() - 0.21).abs() < 1e-9);
        assert!((b.pmf(1).value() - 0.7).abs() < 1e-9);
        assert!((b.pmf(0).value() - 0.3).abs() < 1e-9);
        assert_eq!(b.pmf(2).value(), 0.0);
    }

    #[test]
    fn test_bernoulli_entropy() {
        // Massima incertezza con p = 0.5 -> 1 bit di entropia
        let fair = Bernoulli::new(0.5).expect("valido");
        assert!((fair.entropy_bits() - 1.0).abs() < 1e-9);

        // Certezza assoluta -> 0 bit di entropia
        let certain = Bernoulli::new(1.0).expect("valido");
        assert_eq!(certain.entropy_bits(), 0.0);
    }
}
