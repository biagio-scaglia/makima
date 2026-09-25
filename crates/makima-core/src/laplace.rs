//! Modulo Laplace Mail: aggregazione e generazione di bollettini previsionali periodici.
//!
//! Ispirato concettualmente alla *Laplace Mail* di *Shin Megami Tensei: Devil Survivor*,
//! questo modulo sintetizza lo stato probabilistico di tutti i target monitorati,
//! categorizzando gli eventi imminenti ad alta confidenza e le aree a massima incertezza epistemica.

use crate::{MakimaEngine, Probability};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Notifica e sintesi previsionale per un singolo target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetSummary {
    /// Nome del target monitorato.
    pub target: String,
    /// Numero totale di evidenze registrate.
    pub observations_count: usize,
    /// Numero di successi storici (valore >= 0.5).
    pub success_count: usize,
    /// Numero di insuccessi storici (valore < 0.5).
    pub failure_count: usize,
    /// Previsione probabilistica corrente.
    pub probability: Probability,
    /// Incertezza epistemica (varianza del posterior).
    pub uncertainty_variance: f64,
    /// Entropia di Shannon (in bit).
    pub entropy_bits: f64,
    /// Timestamp Unix dell'ultima osservazione registrata.
    pub last_timestamp_sec: i64,
    /// Frequenza media stimata di occorrenza (eventi al giorno).
    pub estimated_daily_rate: f64,
}

/// Bollettino previsionale consolidato (*Laplace Mail*).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaplaceMail {
    /// Identificativo del bollettino emesso.
    pub issue_id: String,
    /// Timestamp di generazione del bollettino.
    pub timestamp_sec: i64,
    /// Target ad alta probabilità attesa (E[P] >= 65%).
    pub high_probability_targets: Vec<TargetSummary>,
    /// Target ad elevata incertezza epistemica o entropia (Var >= 0.02 oppure H(p) >= 0.85).
    pub uncertain_targets: Vec<TargetSummary>,
    /// Target a bassa probabilità attesa (E[P] < 50%).
    pub low_probability_targets: Vec<TargetSummary>,
    /// Elenco completo di tutti i target monitorati.
    pub all_summaries: Vec<TargetSummary>,
}

impl LaplaceMail {
    /// Genera un nuovo bollettino Laplace Mail aggregando lo stato di un [`MakimaEngine`].
    #[must_use]
    pub fn generate(engine: &MakimaEngine, timestamp_sec: i64) -> Self {
        let targets = engine.tracked_targets();
        let mut summaries = Vec::with_capacity(targets.len());

        for target in &targets {
            let summary = engine.summarize_target(target);
            summaries.push(summary);
        }

        let mut high_prob = Vec::new();
        let mut uncertain = Vec::new();
        let mut low_prob = Vec::new();

        for s in &summaries {
            let p = s.probability.value();
            if p >= 0.65 {
                high_prob.push(s.clone());
            } else if p < 0.50 {
                low_prob.push(s.clone());
            }

            if s.uncertainty_variance >= 0.02 || s.entropy_bits >= 0.85 || s.observations_count < 3
            {
                uncertain.push(s.clone());
            }
        }

        let issue_id = format!("LM-{:04}", (timestamp_sec % 10000));

        Self {
            issue_id,
            timestamp_sec,
            high_probability_targets: high_prob,
            uncertain_targets: uncertain,
            low_probability_targets: low_prob,
            all_summaries: summaries,
        }
    }
}

impl fmt::Display for LaplaceMail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "============================================================"
        )?;
        writeln!(
            f,
            "          LAPLACE MAIL — DAILY PROBABILISTIC BULLETIN       "
        )?;
        writeln!(
            f,
            "============================================================"
        )?;
        writeln!(f, "Bollettino ID:       {}", self.issue_id)?;
        writeln!(f, "Timestamp Emissione: Unix {}", self.timestamp_sec)?;
        writeln!(
            f,
            "Target Monitorati:   {} sistemi attivi",
            self.all_summaries.len()
        )?;
        writeln!(
            f,
            "------------------------------------------------------------"
        )?;

        if !self.high_probability_targets.is_empty() {
            writeln!(
                f,
                "[ EVENTI AD ALTA PROBABILITA' DI CONVERGENZA / SUCCESSO ]"
            )?;
            for t in &self.high_probability_targets {
                writeln!(
                    f,
                    "• {:<18} : E[P] = {:5.1}% | Var = {:.4} | Rate: ~{:.2} ev/gg",
                    t.target,
                    t.probability.value() * 100.0,
                    t.uncertainty_variance,
                    t.estimated_daily_rate
                )?;
            }
            writeln!(f)?;
        }

        if !self.uncertain_targets.is_empty() {
            writeln!(
                f,
                "[ ATTENZIONE: MASSIMA INCERTEZZA EPISTEMICA / ENTROPIA ]"
            )?;
            for t in &self.uncertain_targets {
                writeln!(
                    f,
                    "• {:<18} : H(p) = {:.2} bit | Var = {:.4} ({} evidenze)",
                    t.target, t.entropy_bits, t.uncertainty_variance, t.observations_count
                )?;
                writeln!(
                    f,
                    "  -> Suggerimento: acquisire nuove osservazioni su questo target."
                )?;
            }
            writeln!(f)?;
        }

        if !self.low_probability_targets.is_empty() {
            writeln!(f, "[ EVENTI CRITICI A BASSA PROBABILITA' / A RISCHIO ]")?;
            for t in &self.low_probability_targets {
                writeln!(
                    f,
                    "• {:<18} : E[P] = {:5.1}% | Fallimenti: {}/{}",
                    t.target,
                    t.probability.value() * 100.0,
                    t.failure_count,
                    t.observations_count
                )?;
            }
            writeln!(f)?;
        }

        write!(
            f,
            "============================================================"
        )
    }
}
