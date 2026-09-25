//! # Modulo Probabilistico di Makima
//!
//! Questo modulo fornisce le primitive matematiche e le distribuzioni fondamentali
//! per la quantificazione dell'incertezza e l'inferenza bayesiana.

pub mod bernoulli;
pub mod beta;
pub mod error;
pub mod poisson;
pub mod probability;
pub mod traits;

pub use bernoulli::Bernoulli;
pub use beta::BetaDistribution;
pub use error::ProbError;
pub use poisson::PoissonDistribution;
pub use probability::Probability;
pub use traits::{ContinuousDistribution, DiscreteDistribution, Distribution};
