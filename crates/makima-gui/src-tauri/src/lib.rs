use makima_core::{
    BetaDistribution, Distribution, ForecastRecord, MakimaDb, MakimaEngine, MakimaStore,
    Observation,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Mutex;
use tauri::State;

/// Stato dell'applicazione gestito da Tauri in memoria condivisa thread-safe.
pub struct AppState {
    pub engine: Mutex<MakimaEngine>,
    pub db_path: PathBuf,
}

impl AppState {
    pub fn new() -> Self {
        let db_path = MakimaDb::default_path();
        let mut engine = MakimaEngine::new();

        // Carica dati da SQLite se esiste, altrimenti dal file JSON store o campione
        if db_path.exists() {
            if let Ok(db) = MakimaDb::open(&db_path) {
                let _ = db.load_into_engine(&mut engine);
            }
        } else {
            let json_path = MakimaStore::default_path();
            if let Ok(store) = MakimaStore::load_or_init(&json_path) {
                store.apply_to_engine(&mut engine);
                if let Ok(db) = MakimaDb::open(&db_path) {
                    let _ = db.sync_from_store(&store);
                }
            }
        }

        Self {
            engine: Mutex::new(engine),
            db_path,
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// DTO serializzabile per lo stato generale del motore.
#[derive(Debug, Serialize, Deserialize)]
pub struct EngineStatusDto {
    pub version: String,
    pub state: String,
    pub total_observations: usize,
    pub total_outcomes: usize,
    pub total_forecasts: usize,
    pub pending_forecasts: usize,
}

/// DTO per il riassunto statistico di un target.
#[derive(Debug, Serialize, Deserialize)]
pub struct TargetSummaryDto {
    pub target: String,
    pub observations_count: usize,
    pub success_count: usize,
    pub failure_count: usize,
    pub probability: f64,
    pub uncertainty_variance: f64,
    pub entropy_bits: f64,
    pub last_timestamp_sec: i64,
    pub estimated_daily_rate: f64,
}

/// Punto coordinato per il grafico della curva Beta di probabilità.
#[derive(Debug, Serialize, Deserialize)]
pub struct CurvePointDto {
    pub x: f64,
    pub y: f64,
}

/// DTO per la distribuzione statistica e i punti della curva.
#[derive(Debug, Serialize, Deserialize)]
pub struct DistributionDetailsDto {
    pub target: String,
    pub alpha: f64,
    pub beta: f64,
    pub mean: f64,
    pub variance: f64,
    pub entropy_bits: f64,
    pub ci_lower_95: f64,
    pub ci_upper_95: f64,
    pub curve_points: Vec<CurvePointDto>,
}

/// DTO per la risposta dell'assistente intelligente.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponseDto {
    pub response: String,
    pub thought_trace: Option<String>,
    pub confidence: Option<f64>,
    pub target: Option<String>,
    pub timestamp_sec: i64,
}

/// Approssimazione di Lanczos per il logaritmo della funzione Gamma $\ln \Gamma(x)$.
#[allow(clippy::excessive_precision, clippy::inconsistent_digit_grouping)]
fn ln_gamma(x: f64) -> f64 {
    let p = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_836_524,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let z = x - 1.0;
        let mut sum = p[0];
        for (i, &coeff) in p.iter().enumerate().skip(1) {
            sum += coeff / (z + i as f64);
        }
        let t = z + 7.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + sum.ln()
    }
}

/// Calcola la funzione di densità di probabilità Beta (PDF) esatta in un punto $x \in (0, 1)$.
fn beta_pdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }
    let ln_b = ln_gamma(alpha) + ln_gamma(beta) - ln_gamma(alpha + beta);
    let ln_pdf = (alpha - 1.0) * x.ln() + (beta - 1.0) * (1.0 - x).ln() - ln_b;
    ln_pdf.exp()
}

/// Comando IPC: Restituisce lo stato diagnostico del motore.
#[tauri::command]
fn get_engine_status(state: State<'_, AppState>) -> Result<EngineStatusDto, String> {
    let engine = state.engine.lock().map_err(|e| e.to_string())?;
    let status = engine.status();
    Ok(EngineStatusDto {
        version: status.version.to_string(),
        state: status.state.to_string(),
        total_observations: status.total_observations,
        total_outcomes: status.total_outcomes,
        total_forecasts: status.total_forecasts,
        pending_forecasts: status.pending_forecasts,
    })
}

/// Comando IPC: Restituisce i riassunti di tutti i target tracciati.
#[tauri::command]
fn get_all_target_summaries(state: State<'_, AppState>) -> Result<Vec<TargetSummaryDto>, String> {
    let engine = state.engine.lock().map_err(|e| e.to_string())?;
    let summaries = engine.target_summaries();
    let dtos = summaries
        .into_iter()
        .map(|s| TargetSummaryDto {
            target: s.target,
            observations_count: s.observations_count,
            success_count: s.success_count,
            failure_count: s.failure_count,
            probability: s.probability.value(),
            uncertainty_variance: s.uncertainty_variance,
            entropy_bits: s.entropy_bits,
            last_timestamp_sec: s.last_timestamp_sec,
            estimated_daily_rate: s.estimated_daily_rate,
        })
        .collect();
    Ok(dtos)
}

/// Comando IPC: Calcola e genera la curva di densità di probabilità Beta e l'intervallo di credibilità per un target.
#[tauri::command]
fn get_target_distribution(
    target: String,
    state: State<'_, AppState>,
) -> Result<DistributionDetailsDto, String> {
    let engine = state.engine.lock().map_err(|e| e.to_string())?;
    let summary = engine.summarize_target(&target);
    let alpha = 1.0 + summary.success_count as f64;
    let beta_param = 1.0 + summary.failure_count as f64;
    let dist = BetaDistribution::new(alpha, beta_param).map_err(|e| e.to_string())?;

    let mean = dist.mean();
    let variance = dist.variance();
    let entropy = summary.entropy_bits;

    // Calcolo approssimato intervallo di credibilità al 95% (media +- 1.96 * std_dev, bounded in [0, 1])
    let std_dev = variance.sqrt();
    let ci_lower_95 = (mean - 1.96 * std_dev).clamp(0.001, 0.999);
    let ci_upper_95 = (mean + 1.96 * std_dev).clamp(0.001, 0.999);

    // Generazione di 100 punti campionati da 0.005 a 0.995 per la curva PDF
    let mut curve_points = Vec::with_capacity(101);
    let num_points = 100;
    for i in 0..=num_points {
        let x = 0.005 + (i as f64 / num_points as f64) * 0.99;
        let y = beta_pdf(x, alpha, beta_param).clamp(0.0, 50.0);
        curve_points.push(CurvePointDto { x, y });
    }

    Ok(DistributionDetailsDto {
        target,
        alpha,
        beta: beta_param,
        mean,
        variance,
        entropy_bits: entropy,
        ci_lower_95,
        ci_upper_95,
        curve_points,
    })
}

/// Comando IPC: Restituisce tutte le osservazioni registrate.
#[tauri::command]
fn get_observations(state: State<'_, AppState>) -> Result<Vec<Observation>, String> {
    let engine = state.engine.lock().map_err(|e| e.to_string())?;
    Ok(engine.observations().to_vec())
}

/// Comando IPC: Restituisce i record storici del ledger di calibrazione.
#[tauri::command]
fn get_forecast_ledger(state: State<'_, AppState>) -> Result<Vec<ForecastRecord>, String> {
    let engine = state.engine.lock().map_err(|e| e.to_string())?;
    Ok(engine.ledger().records().to_vec())
}

/// Comando IPC: Registra una nuova evidenza/osservazione empirica (in memoria, SQLite ed Event Log).
#[tauri::command]
fn add_observation(
    target: String,
    success: bool,
    notes: Option<String>,
    state: State<'_, AppState>,
) -> Result<TargetSummaryDto, String> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    // Salva nel database SQLite
    if let Ok(db) = MakimaDb::open(&state.db_path) {
        let _ = db.insert_observation(
            &target,
            if success { 1.0 } else { 0.0 },
            now,
            notes.as_deref(),
        );
    }

    // Salva nell'engine in memoria
    let mut engine = state.engine.lock().map_err(|e| e.to_string())?;
    engine.record_binary(&target, success, now);
    let summary = engine.summarize_target(&target);

    // Salva anche nel file store JSON di fallback
    let store = MakimaStore::from_engine(&engine);
    let _ = store.save(MakimaStore::default_path());

    Ok(TargetSummaryDto {
        target: summary.target,
        observations_count: summary.observations_count,
        success_count: summary.success_count,
        failure_count: summary.failure_count,
        probability: summary.probability.value(),
        uncertainty_variance: summary.uncertainty_variance,
        entropy_bits: summary.entropy_bits,
        last_timestamp_sec: summary.last_timestamp_sec,
        estimated_daily_rate: summary.estimated_daily_rate,
    })
}

/// Comando IPC: Genera il testo completo del bollettino Laplace Mail.
#[tauri::command]
fn generate_laplace_bulletin(state: State<'_, AppState>) -> Result<String, String> {
    let engine = state.engine.lock().map_err(|e| e.to_string())?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let mail = engine.generate_laplace_mail(now);
    Ok(mail.to_string())
}

/// Comando IPC: Risponde a una query naturale integrando la base di conoscenza e il modello bayesiano.
#[tauri::command]
fn query_chat(query: String, state: State<'_, AppState>) -> Result<ChatResponseDto, String> {
    let engine = state.engine.lock().map_err(|e| e.to_string())?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let q_lower = query.to_lowercase();
    let targets = engine.tracked_targets();

    // Cerca se l'utente menziona un target specifico
    let matched_target = targets
        .into_iter()
        .find(|t| q_lower.contains(&t.to_lowercase()));

    let (response, thought_trace, confidence, target_ret) = if let Some(target) = matched_target {
        let sum = engine.summarize_target(&target);
        let prob_pct = (sum.probability.value() * 100.0).round();
        let thought = format!(
            "1. [Percezione]: L'utente si concentra sul target '{}'.\n2. [Memoria]: Recupero storico di {} evidenze ({} successi, {} fallimenti).\n3. [Analisi Bayesiana]: Aggiornamento Beta({:.1}, {:.1}) con varianza epistemica {:.5}.\n4. [Decisione]: Formulo una stima probabilistica trasparente e calibrata.",
            target, sum.observations_count, sum.success_count, sum.failure_count, 1.0 + sum.success_count as f64, 1.0 + sum.failure_count as f64, sum.uncertainty_variance
        );
        let text = format!(
            "Analisi Bayesiana per **{}**:\n\n• Probabilità a posteriori $P(p)$: **{:.1}%** (Successi: {}, Fallimenti: {})\n• Incertezza epistemica (Varianza): **{:.4}**\n• Entropia informativa: **{:.2} bit**\n• Frequenza stimata: **{:.2} eventi/giorno**\n\nIl modello applica la regola di successione di Laplace Beta({:.1}, {:.1}) aggiornata con {} evidenze storiche.",
            target, prob_pct, sum.success_count, sum.failure_count, sum.uncertainty_variance, sum.entropy_bits, sum.estimated_daily_rate, 1.0 + sum.success_count as f64, 1.0 + sum.failure_count as f64, sum.observations_count
        );
        (text, Some(thought), Some(sum.probability.value()), Some(target))
    } else if q_lower.contains("chi sei") || q_lower.contains("cosa sei") {
        let thought = "1. [Percezione]: Domanda esistenziale sull'identità di Makima.\n2. [Memoria]: Richiamo il principio fondazionale di intelligenza computazionale bayesiana.\n3. [Decisione]: Rispondo in prima persona chiarendo lo scopo analitico e probabilistico.".to_string();
        let text = "Sono **Makima**, un assistente e motore computazionale per il ragionamento bayesiano e la stima probabilistica dell'incertezza. Registro evidenze empiriche (commit, deploy, test) e calcolo distribuzioni di probabilità calibrate per prevedere esiti futuri senza allucinazioni.".to_string();
        (text, Some(thought), Some(0.99), None)
    } else if q_lower.contains("stato")
        || q_lower.contains("status")
        || q_lower.contains("riepilogo")
    {
        let status = engine.status();
        let thought = format!(
            "1. [Percezione]: Richiesta di ispezione diagnostica dello stato.\n2. [Analisi]: Motore '{}' con {} osservazioni e {} previsioni.\n3. [Decisione]: Emetto la scorecard di stato del runtime.",
            status.state, status.total_observations, status.total_forecasts
        );
        let text = format!(
            "**Stato Sistema Makima**:\n• Stato motore: `{}`\n• Totale osservazioni: `{}`\n• Totale esiti verificati: `{}`\n• Previsioni a registro: `{}` (In attesa: `{}`)",
            status.state, status.total_observations, status.total_outcomes, status.total_forecasts, status.pending_forecasts
        );
        (text, Some(thought), Some(1.0), None)
    } else {
        let summaries = engine.target_summaries();
        let thought = format!(
            "1. [Percezione]: Query generica \"{}\".\n2. [Introspezione]: Nessun target univoco menzionato, cerco nella lista dei {} target attivi.\n3. [Decisione]: Presento il quadro d'insieme invitando a una domanda specifica.",
            query, summaries.len()
        );
        let targets_list = if summaries.is_empty() {
            "Nessun target ancora registrato.".to_string()
        } else {
            summaries
                .iter()
                .map(|s| {
                    format!(
                        "- **{}**: {:.1}% ({} oss)",
                        s.target,
                        s.probability.value() * 100.0,
                        s.observations_count
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        let text = format!(
            "Ho analizzato la tua richiesta: *\"{}\"*\n\nAttualmente sto monitorando i seguenti target probabilistici:\n{}\n\nPuoi chiedermi dettagli su un target specifico (es. *\"Probabilità framework_release?\"*) o aggiungere nuove osservazioni.",
            query, targets_list
        );
        (text, Some(thought), None, None)
    };

    Ok(ChatResponseDto {
        response,
        thought_trace,
        confidence,
        target: target_ret,
        timestamp_sec: now,
    })
}

/// Comando IPC: Risolve una previsione pendente con l'esito reale verificato (Ground Truth).
#[tauri::command]
fn resolve_forecast(
    target: String,
    occurred: bool,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let mut engine = state.engine.lock().map_err(|e| e.to_string())?;
    engine.record_outcome(&target, occurred, now);

    // Salva nel database SQLite
    if let Ok(db) = MakimaDb::open(&state.db_path) {
        let _ = db.insert_outcome(&target, occurred, now, None, None);
    }

    // Salva nello store
    let store = MakimaStore::from_engine(&engine);
    let _ = store.save(MakimaStore::default_path());

    Ok(())
}

/// Comando IPC: Sincronizza la telemetria reale dai commit Git del repository.
#[tauri::command]
fn sync_git_telemetry(state: State<'_, AppState>) -> Result<usize, String> {
    // Esegue il modulo python makima_lab sync-git per popolare i target reali git:
    let mut cmd = std::process::Command::new("python");
    cmd.env("PYTHONPATH", "python");
    cmd.args(["-m", "makima_lab", "sync-git"]);
    let _ = cmd.status();

    // Ricarica i dati aggiornati in MakimaEngine dal DB SQLite
    let mut engine = state.engine.lock().map_err(|e| e.to_string())?;
    if let Ok(db) = MakimaDb::open(&state.db_path) {
        let _ = db.load_into_engine(&mut engine);
    }

    Ok(engine.observations().len())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let state = AppState::new();

    tauri::Builder::default()
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            get_engine_status,
            get_all_target_summaries,
            get_target_distribution,
            get_observations,
            get_forecast_ledger,
            add_observation,
            resolve_forecast,
            sync_git_telemetry,
            generate_laplace_bulletin,
            query_chat
        ])
        .run(tauri::generate_context!())
        .expect("errore durante l'esecuzione dell'applicazione Tauri Makima");
}
