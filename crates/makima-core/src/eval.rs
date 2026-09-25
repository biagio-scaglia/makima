//! Modulo di valutazione, scoring probabilistico e calibrazione per Makima.
//!
//! Questo modulo implementa metriche formali per misurare la qualità empirica delle previsioni,
//! tra cui Proper Scoring Rules (Brier Score, Logarithmic Loss), Brier Skill Score contro baseline,
//! Expected Calibration Error (ECE), Maximum Calibration Error (MCE) e Reliability Diagrams.

use crate::prob::Probability;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Risultato reale (*Ground Truth*) verificatosi per un evento previsto.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Outcome {
    /// Target associato all'evento.
    pub target: String,
    /// Se l'evento previsto si è effettivamente verificato (`true` = successo, `false` = fallimento).
    pub occurred: bool,
    /// Timestamp Unix (in secondi) in cui l'esito reale è stato osservato e registrato.
    pub timestamp_sec: i64,
}

impl Outcome {
    /// Crea un nuovo esito reale.
    #[must_use]
    pub fn new(target: impl Into<String>, occurred: bool, timestamp_sec: i64) -> Self {
        Self {
            target: target.into(),
            occurred,
            timestamp_sec,
        }
    }
}

/// Raggruppamento per fascia di probabilità e frequenza empirica associata per l'analisi di calibrazione.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBin {
    /// Limite inferiore del bin (incluso).
    pub lower_bound: f64,
    /// Limite superiore del bin (escluso, tranne l'ultimo).
    pub upper_bound: f64,
    /// Numero di previsioni ricadute in questa fascia.
    pub count: usize,
    /// Media delle probabilità previste dal modello in questo bin.
    pub mean_predicted: f64,
    /// Frequenza empirica reale osservata (successi / count).
    pub observed_frequency: f64,
    /// Errore di calibrazione assoluto: |mean_predicted - observed_frequency|.
    pub calibration_error: f64,
}

/// Calcolo delle Proper Scoring Rules per singole previsioni o insiemi di previsioni.
pub struct Scoring;

impl Scoring {
    /// Calcola il **Brier Score** per una singola previsione:
    /// $\text{BS} = (p - y)^2$, dove $y \in \{0, 1\}$.
    ///
    /// Valori vicini a 0 indicano massima accuratezza; 0.25 corrisponde a una stima casuale (50%).
    #[must_use]
    pub fn brier_score(predicted: Probability, outcome: bool) -> f64 {
        let p = predicted.value();
        let y = if outcome { 1.0 } else { 0.0 };
        let diff = p - y;
        diff * diff
    }

    /// Calcola la **Logarithmic Loss** (Negative Log-Likelihood):
    /// $\text{LogLoss} = - [y \ln(p) + (1-y) \ln(1-p)]$
    ///
    /// Penalizza asintoticamente le previsioni con eccesso di confidenza errate (overconfidence).
    #[must_use]
    pub fn log_loss(predicted: Probability, outcome: bool) -> f64 {
        // Applichiamo un epsilon protettivo per evitare -infinito su stime estreme
        const EPSILON: f64 = 1e-15;
        let p = predicted.value().clamp(EPSILON, 1.0 - EPSILON);

        if outcome {
            -p.ln()
        } else {
            -(1.0 - p).ln()
        }
    }

    /// Calcola il **Brier Skill Score** rispetto a una baseline di riferimento:
    /// $\text{BSS} = 1 - \frac{\text{BS}_{\text{model}}}{\text{BS}_{\text{ref}}}$
    ///
    /// - $\text{BSS} > 0$: il modello predice meglio della baseline di riferimento.
    /// - $\text{BSS} = 0$: il modello equivale alla baseline.
    /// - $\text{BSS} < 0$: il modello predice peggio della baseline.
    #[must_use]
    pub fn brier_skill_score(model_brier: f64, baseline_brier: f64) -> f64 {
        if baseline_brier.abs() < 1e-12 {
            0.0
        } else {
            1.0 - (model_brier / baseline_brier)
        }
    }
}

/// Registro e scheda di valutazione aggregata della qualità previsionale di Makima.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// Numero di previsioni confrontate con il rispettivo esito reale.
    pub total_evaluated: usize,
    /// Media del Brier Score del modello (più basso = migliore).
    pub mean_brier_score: f64,
    /// Media della Log Loss (più basso = migliore).
    pub mean_log_loss: f64,
    /// Brier Score di una baseline ingenua costante al 50%.
    pub baseline_brier_score: f64,
    /// Brier Skill Score rispetto alla baseline (positivo = supera la baseline).
    pub brier_skill_score: f64,
    /// Expected Calibration Error (ECE pesato sul numero di campioni per bin).
    pub expected_calibration_error: f64,
    /// Maximum Calibration Error (MCE massimo scostamento rilevato).
    pub max_calibration_error: f64,
    /// Fasce di calibrazione con frequenze osservate (Reliability Data).
    pub calibration_bins: Vec<CalibrationBin>,
}

impl fmt::Display for EvaluationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "============================================================"
        )?;
        writeln!(
            f,
            "             MAKIMA PREDICTIVE EVALUATION REPORT            "
        )?;
        writeln!(
            f,
            "============================================================"
        )?;
        writeln!(f, "Previsioni Valutate:        {}", self.total_evaluated)?;
        writeln!(
            f,
            "Mean Brier Score (Modello): {:.4}  (0.0 = perfetto, 0.25 = casuale)",
            self.mean_brier_score
        )?;
        writeln!(
            f,
            "Baseline Brier (50% unif):  {:.4}",
            self.baseline_brier_score
        )?;
        writeln!(
            f,
            "Brier Skill Score (BSS):    {:+.2}%  ({})",
            self.brier_skill_score * 100.0,
            if self.brier_skill_score > 0.0 {
                "Supera la baseline"
            } else {
                "Inferiore alla baseline"
            }
        )?;
        writeln!(f, "Mean Log Loss:              {:.4}", self.mean_log_loss)?;
        writeln!(
            f,
            "Expected Calib Error (ECE): {:.2}%",
            self.expected_calibration_error * 100.0
        )?;
        writeln!(
            f,
            "Max Calib Error (MCE):      {:.2}%",
            self.max_calibration_error * 100.0
        )?;
        writeln!(
            f,
            "------------------------------------------------------------"
        )?;
        writeln!(
            f,
            "DIAGRAMMA DI CALIBRAZIONE & AFFIDABILITÀ (RELIABILITY CURVE)"
        )?;
        writeln!(
            f,
            "Fascia Pred | N. | E[P]  | Reale | Errore | Allineamento"
        )?;

        for bin in &self.calibration_bins {
            if bin.count == 0 {
                writeln!(
                    f,
                    "{:.2}-{:.2}   |  0 |  --   |  --   |   --   | [                    ]",
                    bin.lower_bound, bin.upper_bound
                )?;
            } else {
                // Rendering barra ASCII a 20 caratteri confrontando Previsto (P) e Reale (R)
                let p_pos = (bin.mean_predicted * 19.0).round().clamp(0.0, 19.0) as usize;
                let r_pos = (bin.observed_frequency * 19.0).round().clamp(0.0, 19.0) as usize;

                let mut bar = vec!['-'; 20];
                if p_pos == r_pos {
                    bar[p_pos] = '='; // Perfetta coincidenza
                } else {
                    bar[p_pos] = 'P';
                    bar[r_pos] = 'R';
                }

                let note = if bin.calibration_error <= 0.05 {
                    "Calibrato"
                } else if bin.mean_predicted > bin.observed_frequency {
                    "Overconfident"
                } else {
                    "Underconfident"
                };

                writeln!(
                    f,
                    "{:.2}-{:.2}   | {:2} | {:.2}  | {:.2}  | {:5.1}% | [{}] {}",
                    bin.lower_bound,
                    bin.upper_bound,
                    bin.count,
                    bin.mean_predicted,
                    bin.observed_frequency,
                    bin.calibration_error * 100.0,
                    bar.into_iter().collect::<String>(),
                    note
                )?;
            }
        }

        write!(
            f,
            "============================================================"
        )
    }
}

/// Valutatore aggregato che confronta coppie di (Previsione, Risultato Reale).
#[derive(Debug, Default, Clone)]
pub struct Evaluator {
    pairs: Vec<(Probability, bool)>,
}

impl Evaluator {
    /// Crea un nuovo valutatore vuoto.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra una coppia previsione / esito reale manifestatosi.
    pub fn add_prediction_outcome(&mut self, predicted: Probability, actual: bool) {
        self.pairs.push((predicted, actual));
    }

    /// Raggruppa le osservazioni in fasce di calibrazione e calcola l'affidabilità empirica.
    #[must_use]
    pub fn calibration_bins(&self, num_bins: usize) -> Vec<CalibrationBin> {
        let bins_count = num_bins.max(1);
        let bin_width = 1.0 / bins_count as f64;
        let mut bins: Vec<CalibrationBin> = Vec::with_capacity(bins_count);

        for i in 0..bins_count {
            let lower = i as f64 * bin_width;
            let upper = (i + 1) as f64 * bin_width;

            let in_bin: Vec<(Probability, bool)> = self
                .pairs
                .iter()
                .copied()
                .filter(|(p, _)| {
                    let v = p.value();
                    if i == bins_count - 1 {
                        v >= lower && v <= upper
                    } else {
                        v >= lower && v < upper
                    }
                })
                .collect();

            if in_bin.is_empty() {
                bins.push(CalibrationBin {
                    lower_bound: lower,
                    upper_bound: upper,
                    count: 0,
                    mean_predicted: 0.0,
                    observed_frequency: 0.0,
                    calibration_error: 0.0,
                });
            } else {
                let n = in_bin.len() as f64;
                let sum_p: f64 = in_bin.iter().map(|(p, _)| p.value()).sum();
                let sum_y: f64 = in_bin.iter().map(|(_, y)| if *y { 1.0 } else { 0.0 }).sum();

                let mean_p = sum_p / n;
                let obs_freq = sum_y / n;
                let error = (mean_p - obs_freq).abs();

                bins.push(CalibrationBin {
                    lower_bound: lower,
                    upper_bound: upper,
                    count: in_bin.len(),
                    mean_predicted: mean_p,
                    observed_frequency: obs_freq,
                    calibration_error: error,
                });
            }
        }

        bins
    }

    /// Calcola il report statistico completo di calibrazione e scoring.
    #[must_use]
    pub fn evaluate(&self) -> Option<EvaluationReport> {
        if self.pairs.is_empty() {
            return None;
        }

        let n = self.pairs.len() as f64;
        let mut sum_brier = 0.0;
        let mut sum_log_loss = 0.0;
        let mut sum_baseline_brier = 0.0;

        let baseline_prob = Probability::new(0.5).unwrap_or(Probability::ZERO);

        for &(p, y) in &self.pairs {
            sum_brier += Scoring::brier_score(p, y);
            sum_log_loss += Scoring::log_loss(p, y);
            sum_baseline_brier += Scoring::brier_score(baseline_prob, y);
        }

        let mean_brier = sum_brier / n;
        let mean_log_loss = sum_log_loss / n;
        let baseline_brier = sum_baseline_brier / n;
        let bss = Scoring::brier_skill_score(mean_brier, baseline_brier);

        let bins = self.calibration_bins(5);

        // Calcolo ECE ed MCE
        let mut weighted_error_sum = 0.0;
        let mut max_err = 0.0;

        for bin in &bins {
            if bin.count > 0 {
                weighted_error_sum += (bin.count as f64 / n) * bin.calibration_error;
                if bin.calibration_error > max_err {
                    max_err = bin.calibration_error;
                }
            }
        }

        Some(EvaluationReport {
            total_evaluated: self.pairs.len(),
            mean_brier_score: mean_brier,
            mean_log_loss,
            baseline_brier_score: baseline_brier,
            brier_skill_score: bss,
            expected_calibration_error: weighted_error_sum,
            max_calibration_error: max_err,
            calibration_bins: bins,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_prediction_scoring() {
        let p_true = Probability::ONE;
        let p_false = Probability::ZERO;

        assert_eq!(Scoring::brier_score(p_true, true), 0.0);
        assert_eq!(Scoring::brier_score(p_false, false), 0.0);
    }

    #[test]
    fn test_random_prediction_scoring() {
        let p_half = Probability::new(0.5).expect("valido");
        // (0.5 - 1.0)^2 = 0.25
        assert_eq!(Scoring::brier_score(p_half, true), 0.25);
        assert_eq!(Scoring::brier_score(p_half, false), 0.25);
    }

    #[test]
    fn test_evaluator_skill_score_and_calibration() {
        let mut evaluator = Evaluator::new();
        // Previsioni accurate
        evaluator.add_prediction_outcome(Probability::new(0.8).expect("v"), true);
        evaluator.add_prediction_outcome(Probability::new(0.9).expect("v"), true);
        evaluator.add_prediction_outcome(Probability::new(0.1).expect("v"), false);

        let report = evaluator.evaluate().expect("report presente");
        assert_eq!(report.total_evaluated, 3);
        assert!(report.mean_brier_score < 0.25);
        assert!(report.brier_skill_score > 0.0);
        assert!(!report.calibration_bins.is_empty());
        assert!(report.expected_calibration_error >= 0.0);
    }
}
