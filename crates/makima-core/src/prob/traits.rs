//! Tratti fondamentali per definire distribuzioni probabilistiche analitiche e stocastiche.

use super::probability::Probability;

/// Tratto comune a tutte le distribuzioni di probabilità.
pub trait Distribution {
    /// Restituisce il valore atteso (media) della distribuzione: $\mathbb{E}[X]$.
    fn mean(&self) -> f64;

    /// Restituisce la varianza della distribuzione: $\text{Var}(X) = \mathbb{E}[(X - \mu)^2]$.
    fn variance(&self) -> f64;

    /// Restituisce la deviazione standard: $\sigma = \sqrt{\text{Var}(X)}$.
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Tratto per distribuzioni probabilistiche discrete.
pub trait DiscreteDistribution: Distribution {
    /// Calcola la funzione di massa di probabilità (PMF): $P(X = k)$.
    fn pmf(&self, k: u64) -> Probability;

    /// Calcola la funzione di ripartizione cumulativa discreta (CDF): $P(X \le k)$.
    fn cdf(&self, k: u64) -> Probability;
}

/// Tratto per distribuzioni probabilistiche continue.
pub trait ContinuousDistribution: Distribution {
    /// Calcola la funzione di densità di probabilità (PDF): $f(x)$.
    fn pdf(&self, x: f64) -> f64;

    /// Calcola la funzione di ripartizione cumulativa continua (CDF): $F(x) = P(X \le x)$.
    fn cdf(&self, x: f64) -> Probability;
}
