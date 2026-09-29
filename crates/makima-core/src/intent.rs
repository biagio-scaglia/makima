//! Contratto semantico condiviso: mirror Rust di `StructuredIntent` Python.
//!
//! Il parser NLP (`makima_lab parse-intent`) emette JSON conforme a questi tipi.
//! Il core Rust deserializza e decide se eseguire il forecast.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Intento categorico allineato a `makima_lab.nlp.schemas.intent.Intent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum IntentKind {
    Query,
    Command,
    Information,
    TemporalQuery,
    Observation,
    Status,
    Unknown,
}

impl IntentKind {
    /// True se l'intento tipicamente richiede un forecast sul core.
    #[must_use]
    pub const fn is_forecast_intent(self) -> bool {
        matches!(self, Self::Query | Self::TemporalQuery)
    }
}

impl fmt::Display for IntentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Query => "QUERY",
            Self::Command => "COMMAND",
            Self::Information => "INFORMATION",
            Self::TemporalQuery => "TEMPORAL_QUERY",
            Self::Observation => "OBSERVATION",
            Self::Status => "STATUS",
            Self::Unknown => "UNKNOWN",
        };
        write!(f, "{s}")
    }
}

/// Relazione temporale allineata a `TemporalRelation` Python.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TemporalRelation {
    Past,
    Present,
    Future,
    RelativeInterval,
    SpecificDate,
    Unknown,
}

impl fmt::Display for TemporalRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Past => "PAST",
            Self::Present => "PRESENT",
            Self::Future => "FUTURE",
            Self::RelativeInterval => "RELATIVE_INTERVAL",
            Self::SpecificDate => "SPECIFIC_DATE",
            Self::Unknown => "UNKNOWN",
        };
        write!(f, "{s}")
    }
}

/// Finestra temporale strutturata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalWindow {
    pub relation: TemporalRelation,
    #[serde(default)]
    pub raw_expression: Option<String>,
    #[serde(default)]
    pub boundary: Option<String>,
    #[serde(default)]
    pub days: Option<u32>,
}

impl TemporalWindow {
    #[must_use]
    pub fn unspecified() -> Self {
        Self {
            relation: TemporalRelation::Unknown,
            raw_expression: None,
            boundary: None,
            days: None,
        }
    }

    /// Descrizione compatta per il ledger forecast.
    #[must_use]
    pub fn window_desc(&self) -> String {
        if let Some(days) = self.days {
            format!("{days} days")
        } else {
            self.relation.to_string()
        }
    }
}

/// DTO immutabile prodotto dalla pipeline NLP e consumato dal runtime Rust.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredIntent {
    pub raw_query: String,
    pub intent: IntentKind,
    pub target: Option<String>,
    pub temporal_window: TemporalWindow,
    #[serde(default)]
    pub entities: Vec<String>,
    pub confidence: f64,
    #[serde(default)]
    pub confidence_breakdown: HashMap<String, serde_json::Value>,
    pub is_valid_for_core: bool,
    #[serde(default)]
    pub validation_notes: Vec<String>,
}

impl StructuredIntent {
    /// Deserializza da JSON compatto (stdout di `parse-intent`).
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Estrae la prima riga JSON da uno stdout potenzialmente rumoroso.
    pub fn from_parser_stdout(stdout: &str) -> Result<Self, String> {
        let json_line = stdout
            .lines()
            .map(str::trim)
            .find(|l| l.starts_with('{'))
            .ok_or_else(|| "Nessun JSON StructuredIntent nello stdout del parser.".to_string())?;
        Self::from_json_str(json_line).map_err(|e| format!("JSON StructuredIntent non valido: {e}"))
    }

    /// True se il core può eseguire un forecast su questo intento.
    #[must_use]
    pub fn admits_forecast(&self) -> bool {
        self.is_valid_for_core
            && self.intent.is_forecast_intent()
            && self
                .target
                .as_ref()
                .is_some_and(|t| !t.trim().is_empty())
    }

    /// Note di validazione unite in una stringa.
    #[must_use]
    pub fn notes_joined(&self) -> String {
        self.validation_notes.join("; ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN_RELEASE_QUERY: &str = r#"{"raw_query":"Quando rilascerò il prossimo framework?","intent":"TEMPORAL_QUERY","target":"framework_release","temporal_window":{"relation":"FUTURE","raw_expression":"future","boundary":null,"days":null},"entities":["prossimo framework"],"confidence":0.82,"confidence_breakdown":{"intent_score":0.9,"target_match_method":"descriptor_match"},"is_valid_for_core":true,"validation_notes":["Intento e target conformi ai vincoli di dominio."]}"#;

    #[test]
    fn deserializes_python_structured_intent_json() {
        let intent = StructuredIntent::from_json_str(GOLDEN_RELEASE_QUERY).expect("json valido");
        assert_eq!(intent.intent, IntentKind::TemporalQuery);
        assert_eq!(intent.target.as_deref(), Some("framework_release"));
        assert_eq!(intent.temporal_window.relation, TemporalRelation::Future);
        assert!(intent.is_valid_for_core);
        assert!(intent.admits_forecast());
        assert!((intent.confidence - 0.82).abs() < 1e-9);
    }

    #[test]
    fn rejects_unknown_for_forecast() {
        let json = r#"{"raw_query":"ciao","intent":"UNKNOWN","target":null,"temporal_window":{"relation":"UNKNOWN","raw_expression":null,"boundary":null,"days":null},"entities":[],"confidence":0.95,"confidence_breakdown":{},"is_valid_for_core":false,"validation_notes":["Intento classificato come UNKNOWN o fuori dominio."]}"#;
        let intent = StructuredIntent::from_json_str(json).expect("json");
        assert!(!intent.admits_forecast());
    }

    #[test]
    fn from_parser_stdout_skips_noise() {
        let stdout = "Loading model...\n{\"raw_query\":\"x\",\"intent\":\"STATUS\",\"target\":null,\"temporal_window\":{\"relation\":\"UNKNOWN\"},\"confidence\":0.7,\"is_valid_for_core\":true,\"validation_notes\":[]}\n";
        let intent = StructuredIntent::from_parser_stdout(stdout).expect("parse");
        assert_eq!(intent.intent, IntentKind::Status);
    }

    #[test]
    fn roundtrip_serde() {
        let intent = StructuredIntent::from_json_str(GOLDEN_RELEASE_QUERY).unwrap();
        let encoded = serde_json::to_string(&intent).unwrap();
        let decoded = StructuredIntent::from_json_str(&encoded).unwrap();
        assert_eq!(intent, decoded);
    }
}
