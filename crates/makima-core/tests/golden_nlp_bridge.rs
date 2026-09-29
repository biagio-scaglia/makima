//! Golden path: StructuredIntent (contratto NLP) → forecast Rust su evidenze empiriche.
//!
//! Non richiede Python a runtime: usa JSON golden allineato allo schema Python.

use makima_core::{IntentKind, MakimaEngine, StructuredIntent, TemporalRelation};

const GOLDEN_DEPLOY_QUERY: &str = r#"{"raw_query":"Qual è la probabilità del deploy?","intent":"QUERY","target":"deploy","temporal_window":{"relation":"UNKNOWN","raw_expression":null,"boundary":null,"days":null},"entities":["deploy"],"confidence":0.78,"confidence_breakdown":{"target_match_method":"descriptor_match"},"is_valid_for_core":true,"validation_notes":["Intento e target conformi ai vincoli di dominio."]}"#;

const GOLDEN_UNKNOWN: &str = r#"{"raw_query":"ciao makima","intent":"UNKNOWN","target":null,"temporal_window":{"relation":"UNKNOWN","raw_expression":null,"boundary":null,"days":null},"entities":[],"confidence":0.95,"confidence_breakdown":{"unknown_rejection":0.95},"is_valid_for_core":false,"validation_notes":["Intento classificato come UNKNOWN o fuori dominio."]}"#;

#[test]
fn golden_intent_to_rust_forecast_with_empirical_evidence() {
    let intent = StructuredIntent::from_json_str(GOLDEN_DEPLOY_QUERY).expect("golden JSON");
    assert_eq!(intent.intent, IntentKind::Query);
    assert_eq!(intent.target.as_deref(), Some("deploy"));
    assert!(intent.admits_forecast());

    let mut engine = MakimaEngine::new();
    engine.record_binary("deploy", true, 1_700_000_000);
    engine.record_binary("deploy", true, 1_700_086_400);
    engine.record_binary("deploy", false, 1_700_172_800);

    let target = intent.target.as_deref().expect("target");
    let forecast = engine.predict_target(target);
    assert_eq!(forecast.evidence_count, 3);
    assert!((forecast.probability.value() - 0.6).abs() < 1e-9);

    let id = engine.register_forecast_in_ledger(
        target,
        1_700_200_000,
        forecast.probability,
        intent.temporal_window.window_desc(),
        forecast.evidence_count,
        "BayesianConjugate+NLP",
    );
    assert_eq!(id.0, 1);
    assert_eq!(engine.ledger().pending_count(), 1);
}

#[test]
fn golden_unknown_intent_does_not_admit_forecast() {
    let intent = StructuredIntent::from_json_str(GOLDEN_UNKNOWN).expect("golden");
    assert_eq!(intent.intent, IntentKind::Unknown);
    assert!(!intent.admits_forecast());
}

#[test]
fn golden_temporal_release_query_contract() {
    let json = r#"{"raw_query":"Quando rilascerò il prossimo framework?","intent":"TEMPORAL_QUERY","target":"framework_release","temporal_window":{"relation":"FUTURE","raw_expression":"future","boundary":null,"days":null},"entities":["prossimo framework"],"confidence":0.82,"confidence_breakdown":{},"is_valid_for_core":true,"validation_notes":["ok"]}"#;
    let intent = StructuredIntent::from_json_str(json).unwrap();
    assert_eq!(intent.intent, IntentKind::TemporalQuery);
    assert_eq!(intent.temporal_window.relation, TemporalRelation::Future);
    assert_eq!(intent.temporal_window.window_desc(), "FUTURE");
    assert!(intent.admits_forecast());
}
