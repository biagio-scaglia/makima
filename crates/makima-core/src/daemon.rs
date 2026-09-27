//! # Autonomous Daemon & Cognitive Pulse Engine
//!
//! Questo modulo implementa il motore di sorveglianza e coscienza attiva di Makima.
//! Analizza flussi di telemetria, rileva derive epistemiche, anomalie statistiche
//! nei commit e genera impulsi cognitivi proattivi ("Cognitive Pulses") per il Second Brain.

use crate::prob::{Bernoulli, Probability};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Livello di urgenza di un impulso cognitivo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PulseUrgency {
    /// Informazione di routine o riflessione periodica.
    Low,
    /// Variazione stocastica notevole ma gestibile.
    Medium,
    /// Deriva epistemica elevata o anomalia significativa nei dati.
    High,
    /// Rischio critico di calibrazione o regressione evidente.
    Critical,
}

impl fmt::Display for PulseUrgency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Tipologia di evento scatenante dell'impulso.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "details")]
pub enum PulseTrigger {
    /// Nuovo commit Git rilevato e analizzato.
    GitCommit {
        hash: String,
        author: String,
        message: String,
        files_changed: usize,
    },
    /// Variazione sensibile di probabilità o entropia su un target.
    EpistemicShift {
        target: String,
        prior_prob: f64,
        new_prob: f64,
        delta_entropy: f64,
    },
    /// Anomalia statistica rilevata (Z-Score elevato).
    AnomalyDetected {
        metric: String,
        value: f64,
        z_score: f64,
        reason: String,
    },
    /// Riflessione autonoma programmata sullo stato delle previsioni aperte.
    PeriodicReflection {
        active_forecasts: usize,
        observed_targets: usize,
    },
}

/// Rappresenta un singolo impulso cognitivo generato autonomamente da Makima.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitivePulse {
    /// Identificativo univoco dell'impulso.
    pub id: String,
    /// Timestamp Unix in secondi.
    pub timestamp_sec: i64,
    /// Evento scatenante.
    pub trigger: PulseTrigger,
    /// Livello di urgenza.
    pub urgency: PulseUrgency,
    /// Monologo interiore e riflessione cognitiva proattiva.
    pub inner_thought: String,
    /// Raccomandazione o azione consigliata per l'utente/team.
    pub suggested_action: Option<String>,
}

impl CognitivePulse {
    /// Crea un nuovo impulso cognitivo.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        timestamp_sec: i64,
        trigger: PulseTrigger,
        urgency: PulseUrgency,
        inner_thought: impl Into<String>,
        suggested_action: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            timestamp_sec,
            trigger,
            urgency,
            inner_thought: inner_thought.into(),
            suggested_action,
        }
    }
}

/// Motore per il monitoraggio della telemetria e generazione di impulsi di coscienza attiva.
#[derive(Debug, Clone, Default)]
pub struct AutonomousWatcher {
    /// Ultimo hash commit Git analizzato.
    last_seen_commit: Option<String>,
    /// Storico delle dimensioni dei commit (numero file) per calcolo Z-score.
    commit_size_history: Vec<f64>,
}

impl AutonomousWatcher {
    /// Inizializza un nuovo watcher.
    #[must_use]
    pub fn new() -> Self {
        Self {
            last_seen_commit: None,
            commit_size_history: Vec::new(),
        }
    }

    /// Analizza un nuovo commit Git e produce un impulso cognitivo se presenta anomalie o pattern rilevanti.
    pub fn process_git_commit(
        &mut self,
        hash: &str,
        author: &str,
        message: &str,
        files_changed: usize,
        timestamp_sec: i64,
    ) -> Option<CognitivePulse> {
        if let Some(ref last) = self.last_seen_commit {
            if last == hash {
                return None;
            }
        }
        self.last_seen_commit = Some(hash.to_string());
        let files_f = files_changed as f64;
        self.commit_size_history.push(files_f);

        // Calcolo Z-Score se abbiamo almeno 5 campioni
        let mut z_score = 0.0;
        if self.commit_size_history.len() >= 5 {
            let n = self.commit_size_history.len() as f64;
            let mean = self.commit_size_history.iter().sum::<f64>() / n;
            let var = self
                .commit_size_history
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>()
                / n;
            let std_dev = var.sqrt();
            if std_dev > 0.001 {
                z_score = (files_f - mean) / std_dev;
            }
        }

        let is_feat = message.to_lowercase().starts_with("feat");
        let is_fix = message.to_lowercase().starts_with("fix");
        let is_test = message.to_lowercase().contains("test");

        if z_score >= 2.0 {
            let thought = format!(
                "Rilevato commit anomalo [{}] da {} con {} file modificati (Z-Score: +{:.2}). L'entità del diff potrebbe alterare significativamente la stabilità del grafo delle dipendenze.",
                &hash[..7.min(hash.len())],
                author,
                files_changed,
                z_score
            );
            return Some(CognitivePulse::new(
                format!("pulse_anomaly_{timestamp_sec}"),
                timestamp_sec,
                PulseTrigger::AnomalyDetected {
                    metric: "git_files_changed".to_string(),
                    value: files_f,
                    z_score,
                    reason: format!("Commit insolitamente grande ({files_changed} file)"),
                },
                PulseUrgency::High,
                thought,
                Some("Consiglio una verifica mirata dei test di regressione e l'aggiornamento del prior di deploy.".to_string()),
            ));
        }

        if is_feat || is_fix || is_test {
            let category = if is_feat {
                "nuova feature"
            } else if is_fix {
                "correzione bug"
            } else {
                "test coverage"
            };
            let thought = format!(
                "Registrato commit [{}] ({}) da {}: \"{}\". Integro il segnale empirico nel modello coniugato.",
                &hash[..7.min(hash.len())],
                category,
                author,
                message.lines().next().unwrap_or(message)
            );
            return Some(CognitivePulse::new(
                format!("pulse_git_{timestamp_sec}"),
                timestamp_sec,
                PulseTrigger::GitCommit {
                    hash: hash.to_string(),
                    author: author.to_string(),
                    message: message.to_string(),
                    files_changed,
                },
                PulseUrgency::Low,
                thought,
                None,
            ));
        }

        None
    }

    /// Rileva derive epistemiche sostanziali confrontando la probabilità precedente con quella aggiornata.
    #[must_use]
    pub fn check_epistemic_shift(
        &self,
        target: &str,
        prior_p: f64,
        new_p: f64,
        timestamp_sec: i64,
    ) -> Option<CognitivePulse> {
        let delta_p = (new_p - prior_p).abs();
        let b_prior = Bernoulli::from_probability(Probability::from_clamped(prior_p));
        let b_new = Bernoulli::from_probability(Probability::from_clamped(new_p));
        let delta_h = b_new.entropy_bits() - b_prior.entropy_bits();

        if delta_p >= 0.15 || delta_h.abs() >= 0.20 {
            let urgency = if delta_p >= 0.30 {
                PulseUrgency::High
            } else {
                PulseUrgency::Medium
            };
            let direction = if new_p > prior_p {
                "aumento"
            } else {
                "riduzione"
            };
            let thought = format!(
                "Attenzione: rilevata deriva epistemica su '{}'. Probabilità in {} da {:.1}% a {:.1}% (Δp: {:+.1}%, ΔEntropia: {:+.3} bit).",
                target,
                direction,
                prior_p * 100.0,
                new_p * 100.0,
                (new_p - prior_p) * 100.0,
                delta_h
            );
            return Some(CognitivePulse::new(
                format!("pulse_shift_{target}_{timestamp_sec}"),
                timestamp_sec,
                PulseTrigger::EpistemicShift {
                    target: target.to_string(),
                    prior_prob: prior_p,
                    new_prob: new_p,
                    delta_entropy: delta_h,
                },
                urgency,
                thought,
                Some(format!(
                    "Ricalibrare le previsioni dipendenti da '{}' nel Second Brain.",
                    target
                )),
            ));
        }

        None
    }

    /// Genera una riflessione periodica di sorveglianza attiva sullo stato del sistema.
    #[must_use]
    pub fn generate_periodic_reflection(
        &self,
        active_forecasts: usize,
        observed_targets: usize,
        timestamp_sec: i64,
    ) -> CognitivePulse {
        let thought = format!(
            "Sorveglianza continua attiva: {active_forecasts} previsioni pendenti sotto monitoraggio distribuite su {observed_targets} target empirici. I modelli bayesiani mantengono un flusso di calibrazione coerente."
        );
        CognitivePulse::new(
            format!("pulse_periodic_{timestamp_sec}"),
            timestamp_sec,
            PulseTrigger::PeriodicReflection {
                active_forecasts,
                observed_targets,
            },
            PulseUrgency::Low,
            thought,
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_git_commit_pulse_and_anomaly() {
        let mut watcher = AutonomousWatcher::new();
        // Normal small commits
        for i in 0..6 {
            watcher.process_git_commit(&format!("hash_{i}"), "Dev", "feat: new item", 2, 1000 + i);
        }
        // Huge anomaly commit
        let anomaly_pulse =
            watcher.process_git_commit("hash_huge", "Dev", "refactor all", 80, 2000);
        assert!(anomaly_pulse.is_some());
        let pulse = anomaly_pulse.unwrap();
        assert_eq!(pulse.urgency, PulseUrgency::High);
        assert!(pulse.inner_thought.contains("Z-Score"));
    }

    #[test]
    fn test_epistemic_shift_detection() {
        let watcher = AutonomousWatcher::new();
        let pulse = watcher.check_epistemic_shift("deploy", 0.50, 0.85, 12345);
        assert!(pulse.is_some());
        let p = pulse.unwrap();
        assert_eq!(p.urgency, PulseUrgency::High);
        assert!(p.inner_thought.contains("deriva epistemica"));
    }
}
