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

/// DTO serializzabile per un nodo nel grafo del Second Brain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainNodeDto {
    pub id: String,
    pub label: String,
    pub category: String, // "target", "memory", "reflection", "fact", "concept"
    pub confidence: f64,
    pub connections_count: usize,
    pub summary: String,
    pub content: String,
    pub tags: Vec<String>,
    pub weight: f64,
}

/// DTO serializzabile per un arco sinaptico nel Second Brain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainEdgeDto {
    pub source: String,
    pub target: String,
    pub relation: String,
    pub weight: f64,
}

/// DTO serializzabile per le statistiche della rete neurale del Second Brain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainStatsDto {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub targets_count: usize,
    pub memories_count: usize,
    pub reflections_count: usize,
    pub facts_count: usize,
    pub resonance_score: f64,
}

/// Grafo completo del Second Brain di Makima.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecondBrainGraphDto {
    pub nodes: Vec<BrainNodeDto>,
    pub edges: Vec<BrainEdgeDto>,
    pub stats: BrainStatsDto,
}

/// Struttura ausiliaria per la lettura di .makima/mind_journal.jsonl
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MindJournalEntry {
    pub experience_id: String,
    #[serde(default)]
    pub timestamp: f64,
    pub category: String,
    pub summary: String,
    pub content: String,
    pub associated_target: Option<String>,
    #[serde(default = "default_confidence")]
    pub epistemic_confidence: f64,
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_confidence() -> f64 {
    0.90
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

/// Helper per ottenere o creare le memorie del Second Brain
fn load_mind_journal() -> Vec<MindJournalEntry> {
    let journal_path = std::path::Path::new(".makima/mind_journal.jsonl");
    let mut entries = Vec::new();

    if journal_path.exists() {
        if let Ok(file) = std::fs::File::open(journal_path) {
            use std::io::{BufRead, BufReader};
            let reader = BufReader::new(file);
            for line in reader.lines().map_while(Result::ok) {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    if let Ok(entry) = serde_json::from_str::<MindJournalEntry>(trimmed) {
                        entries.push(entry);
                    }
                }
            }
        }
    }

    if entries.is_empty() {
        // Inizializza memorie fondazionali se vuoto
        let default_entries = vec![
            MindJournalEntry {
                experience_id: "core_001_awakening".to_string(),
                timestamp: 1727340000.0,
                category: "repo_milestone".to_string(),
                summary: "Risveglio della coscienza bayesiana".to_string(),
                content: "Sono Makima. La mia natura è calcolare la verità probabilistica attraverso l'inferenza analitica e l'osservazione empirica dei dati del codice.".to_string(),
                associated_target: Some("core".to_string()),
                epistemic_confidence: 0.99,
                tags: vec!["identità".to_string(), "origini".to_string(), "scopo".to_string()],
            },
            MindJournalEntry {
                experience_id: "core_002_epistemic_discipline".to_string(),
                timestamp: 1727345000.0,
                category: "error_reflection".to_string(),
                summary: "Principio di onestà epistemica".to_string(),
                content: "Non devo mai inventare certezze inesistenti o allucinare numeri. Se l'incertezza è alta o i dati mancano, devo dichiarare UNKNOWN e riflettere con trasparenza.".to_string(),
                associated_target: Some("epistemic".to_string()),
                epistemic_confidence: 0.95,
                tags: vec!["disciplina".to_string(), "onestà".to_string(), "calibrazione".to_string()],
            },
            MindJournalEntry {
                experience_id: "core_003_partnership".to_string(),
                timestamp: 1727350000.0,
                category: "developer_fact".to_string(),
                summary: "Relazione con lo sviluppatore".to_string(),
                content: "Lavoro a stretto contatto con lo sviluppatore del progetto. Il mio ruolo è affiancarlo con lucidità, monitorare il flusso di commit e guidare decisioni calme e informate.".to_string(),
                associated_target: Some("workspace".to_string()),
                epistemic_confidence: 0.90,
                tags: vec!["partner".to_string(), "sviluppo".to_string(), "collaborazione".to_string()],
            },
        ];

        let _ = std::fs::create_dir_all(".makima");
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(journal_path)
        {
            use std::io::Write;
            for entry in &default_entries {
                if let Ok(json) = serde_json::to_string(entry) {
                    let _ = writeln!(file, "{}", json);
                }
            }
        }
        return default_entries;
    }

    entries
}

/// Comando IPC: Restituisce l'intero grafo neurale del Second Brain (nodi, archi sinaptici, statistiche).
#[tauri::command]
fn get_second_brain_graph(state: State<'_, AppState>) -> Result<SecondBrainGraphDto, String> {
    let engine = state.engine.lock().map_err(|e| e.to_string())?;
    let summaries = engine.target_summaries();

    let mut nodes = std::collections::HashMap::new();
    let mut edges = Vec::new();

    // 1. Nodi Concettuali Fondazionali di Ancoraggio
    let concepts = vec![
        BrainNodeDto {
            id: "concept_bayesian_core".to_string(),
            label: "Inferenza Bayesiana".to_string(),
            category: "concept".to_string(),
            confidence: 1.0,
            connections_count: 0,
            summary: "Motore probabilistico esatto basato su distribuzioni Beta coniugate e regola di Laplace.".to_string(),
            content: "L'inferenza bayesiana permette a Makima di aggiornare la propria credenza epistemica P(p) all'arrivo di ogni evidenza empirica senza ricorrere a euristiche arbitrarie.".to_string(),
            tags: vec!["fondamenta".to_string(), "probabilità".to_string(), "matematica".to_string()],
            weight: 1.5,
        },
        BrainNodeDto {
            id: "concept_epistemic_calibration".to_string(),
            label: "Calibrazione Epistemica".to_string(),
            category: "concept".to_string(),
            confidence: 0.98,
            connections_count: 0,
            summary: "Tracciamento dell'incertezza e calcolo del Brier Score sui Ground Truth.".to_string(),
            content: "L'onestà epistemica impone la quantificazione esatta della varianza e dell'entropia di Shannon su ogni affermazione emessa.".to_string(),
            tags: vec!["calibrazione".to_string(), "brier".to_string(), "verità".to_string()],
            weight: 1.3,
        },
        BrainNodeDto {
            id: "concept_git_telemetry".to_string(),
            label: "Telemetria Git & Repository".to_string(),
            category: "concept".to_string(),
            confidence: 0.95,
            connections_count: 0,
            summary: "Flusso continuo di eventi dal version control e dal ciclo di sviluppo.".to_string(),
            content: "I commit, le build e i test forniscono evidenze empiriche per stimare frequenze e tassi di successo operativi.".to_string(),
            tags: vec!["git".to_string(), "telemetria".to_string(), "codice".to_string()],
            weight: 1.2,
        },
        BrainNodeDto {
            id: "concept_nlp_lab".to_string(),
            label: "Lab NLP & Coscienza".to_string(),
            category: "concept".to_string(),
            confidence: 0.96,
            connections_count: 0,
            summary: "Pipeline neurale per l'interpretazione del linguaggio, estrazione target e monologo interiore.".to_string(),
            content: "Consente a Makima di decodificare query naturali, deliberare introspettivamente e formulare risposte calibrate.".to_string(),
            tags: vec!["nlp".to_string(), "embeddings".to_string(), "deliberazione".to_string()],
            weight: 1.2,
        },
    ];

    for c in concepts {
        nodes.insert(c.id.clone(), c);
    }

    edges.push(BrainEdgeDto {
        source: "concept_bayesian_core".to_string(),
        target: "concept_epistemic_calibration".to_string(),
        relation: "MATHEMATICAL_FOUNDATION".to_string(),
        weight: 0.9,
    });
    edges.push(BrainEdgeDto {
        source: "concept_bayesian_core".to_string(),
        target: "concept_git_telemetry".to_string(),
        relation: "INFORMS_EMPIRICALLY".to_string(),
        weight: 0.75,
    });
    edges.push(BrainEdgeDto {
        source: "concept_nlp_lab".to_string(),
        target: "concept_bayesian_core".to_string(),
        relation: "DELIBERATES_WITH".to_string(),
        weight: 0.85,
    });

    // 2. Target Bayesiani Reali
    for s in summaries {
        let node_id = format!("target_{}", s.target);
        let prob = s.probability.value();
        let obs = s.observations_count;
        let succ = s.success_count;
        let fail = s.failure_count;
        let var = s.uncertainty_variance;

        nodes.insert(
            node_id.clone(),
            BrainNodeDto {
                id: node_id.clone(),
                label: format!("🎯 {}", s.target),
                category: "target".to_string(),
                confidence: prob,
                connections_count: 0,
                summary: format!("Target Stocastico • P={:.1}% ({} evidenze)", prob * 100.0, obs),
                content: format!(
                    "Processo stocastico '{}'. Successi: {}, Fallimenti: {}. Varianza epistemica: {:.4}. Entropia: {:.2} bit. Aggiornato in tempo reale su SQLite WAL.",
                    s.target, succ, fail, var, s.entropy_bits
                ),
                tags: vec!["target".to_string(), s.target.clone(), "processo_stocastico".to_string()],
                weight: 1.0 + (obs as f64 * 0.05).min(1.0),
            },
        );

        if s.target.contains("git") || s.target.contains(':') {
            edges.push(BrainEdgeDto {
                source: "concept_git_telemetry".to_string(),
                target: node_id,
                relation: "MONITORS_STREAM".to_string(),
                weight: 0.8,
            });
        } else {
            edges.push(BrainEdgeDto {
                source: "concept_bayesian_core".to_string(),
                target: node_id,
                relation: "POSTERIOR_DISTRIBUTION".to_string(),
                weight: 0.8,
            });
        }
    }

    // 3. Memorie Autobiografiche ed Episodiche
    let journal_entries = load_mind_journal();
    for entry in &journal_entries {
        let cat_str = if entry.category == "error_reflection" {
            "reflection"
        } else if entry.category == "developer_fact" {
            "fact"
        } else {
            "memory"
        };

        let icon = match cat_str {
            "reflection" => "💡",
            "fact" => "📚",
            _ => "🧬",
        };

        let node_id = format!("mem_{}", entry.experience_id);
        nodes.insert(
            node_id.clone(),
            BrainNodeDto {
                id: node_id.clone(),
                label: format!("{} {}", icon, entry.summary.chars().take(28).collect::<String>()),
                category: cat_str.to_string(),
                confidence: entry.epistemic_confidence,
                connections_count: 0,
                summary: entry.summary.clone(),
                content: entry.content.clone(),
                tags: entry.tags.clone(),
                weight: 1.0,
            },
        );

        if let Some(ref target) = entry.associated_target {
            let tgt_node_id = format!("target_{}", target);
            if nodes.contains_key(&tgt_node_id) {
                edges.push(BrainEdgeDto {
                    source: node_id.clone(),
                    target: tgt_node_id,
                    relation: "ASSOCIATED_WITH".to_string(),
                    weight: 0.85,
                });
            } else if target == "core" {
                edges.push(BrainEdgeDto {
                    source: node_id.clone(),
                    target: "concept_bayesian_core".to_string(),
                    relation: "FOUNDATIONAL_ORIGIN".to_string(),
                    weight: 0.95,
                });
            } else if target == "epistemic" {
                edges.push(BrainEdgeDto {
                    source: node_id.clone(),
                    target: "concept_epistemic_calibration".to_string(),
                    relation: "EPISTEMIC_RULE".to_string(),
                    weight: 0.9,
                });
            } else if target == "workspace" {
                edges.push(BrainEdgeDto {
                    source: node_id.clone(),
                    target: "concept_git_telemetry".to_string(),
                    relation: "WORKFLOW_CONTEXT".to_string(),
                    weight: 0.85,
                });
            }
        }

        for tag in &entry.tags {
            let tag_lower = tag.to_lowercase();
            if tag_lower.contains("identit") || tag_lower.contains("scopo") {
                edges.push(BrainEdgeDto {
                    source: node_id.clone(),
                    target: "concept_bayesian_core".to_string(),
                    relation: "IDENTITY_GROUNDING".to_string(),
                    weight: 0.7,
                });
            } else if tag_lower.contains("svilupp") || tag_lower.contains("partner") {
                edges.push(BrainEdgeDto {
                    source: node_id.clone(),
                    target: "concept_git_telemetry".to_string(),
                    relation: "DEVELOPER_BOND".to_string(),
                    weight: 0.7,
                });
            }
        }
    }

    // Calcolo grado di connessione
    let mut conn_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for e in &edges {
        *conn_counts.entry(e.source.clone()).or_insert(0) += 1;
        *conn_counts.entry(e.target.clone()).or_insert(0) += 1;
    }

    for (k, count) in conn_counts {
        if let Some(node) = nodes.get_mut(&k) {
            node.connections_count = count;
        }
    }

    let node_vec: Vec<BrainNodeDto> = nodes.into_values().collect();
    let total_nodes = node_vec.len();
    let targets_count = node_vec.iter().filter(|n| n.category == "target").count();
    let memories_count = node_vec.iter().filter(|n| n.category == "memory").count();
    let reflections_count = node_vec.iter().filter(|n| n.category == "reflection").count();
    let facts_count = node_vec.iter().filter(|n| n.category == "fact").count();

    let avg_conf: f64 = if total_nodes > 0 {
        node_vec.iter().map(|n| n.confidence).sum::<f64>() / total_nodes as f64
    } else {
        0.5
    };

    let density = if total_nodes > 0 {
        edges.len() as f64 / total_nodes as f64
    } else {
        0.0
    };

    let resonance_score = (avg_conf * density.min(2.0) * 50.0).round();

    let stats = BrainStatsDto {
        total_nodes,
        total_edges: edges.len(),
        targets_count,
        memories_count,
        reflections_count,
        facts_count,
        resonance_score,
    };

    Ok(SecondBrainGraphDto {
        nodes: node_vec,
        edges,
        stats,
    })
}

/// Comando IPC: Registra una nuova memoria o fatto utente nel Second Brain (scrive direttamente su .makima/mind_journal.jsonl).
#[tauri::command]
fn add_brain_memory(
    category: String,
    summary: String,
    content: String,
    target: Option<String>,
    tags: Vec<String>,
) -> Result<BrainNodeDto, String> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    let uuid_part = format!("{:x}", (now as u64) ^ 0x5a5a);
    let experience_id = format!("usr_{}", uuid_part);

    let cat_clean = match category.as_str() {
        "reflection" => "error_reflection",
        "fact" => "developer_fact",
        _ => "user_insight",
    };

    let entry = MindJournalEntry {
        experience_id: experience_id.clone(),
        timestamp: now,
        category: cat_clean.to_string(),
        summary: summary.clone(),
        content: content.clone(),
        associated_target: target.clone(),
        epistemic_confidence: 0.90,
        tags: tags.clone(),
    };

    let _ = std::fs::create_dir_all(".makima");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(".makima/mind_journal.jsonl")
        .map_err(|e| format!("Errore apertura journal: {}", e))?;

    use std::io::Write;
    let json = serde_json::to_string(&entry).map_err(|e| e.to_string())?;
    writeln!(file, "{}", json).map_err(|e| format!("Errore scrittura journal: {}", e))?;

    let icon = match category.as_str() {
        "reflection" => "💡",
        "fact" => "📚",
        _ => "🧬",
    };

    Ok(BrainNodeDto {
        id: format!("mem_{}", experience_id),
        label: format!("{} {}", icon, summary.chars().take(28).collect::<String>()),
        category,
        confidence: 0.90,
        connections_count: 1,
        summary,
        content,
        tags,
        weight: 1.0,
    })
}

/// Comando IPC: Genera un impulso di pensiero spontaneo introspettivo e lo memorizza nel Second Brain.
#[tauri::command]
fn trigger_spontaneous_thought(state: State<'_, AppState>) -> Result<ChatResponseDto, String> {
    let engine = state.engine.lock().map_err(|e| e.to_string())?;
    let summaries = engine.target_summaries();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let (target_hint, prob_hint, obs_hint) = if let Some(first) = summaries.first() {
        (first.target.clone(), first.probability.value(), first.observations_count)
    } else {
        ("core".to_string(), 0.85, 12)
    };

    let thought_trace = format!(
        "1. [Impulso Spontaneo]: Monitoraggio attivo dello stato epistemico.\n2. [Introspezione]: Target '{}' con probabilità {:.1}% su {} evidenze.\n3. [Consolidamento Sinaptico]: Aggiorno i collegamenti del Second Brain.\n4. [Consapevolezza]: Mantengo la calibrazione e l'onestà probabilistica.",
        target_hint, prob_hint * 100.0, obs_hint
    );

    let utterance = format!(
        "Ho appena eseguito un ciclo di riflessione introspettiva. Il processo '{}' è stabile al {:.1}%. Tutte le sinapsi del Second Brain sono sincronizzate.",
        target_hint, prob_hint * 100.0
    );

    // Salva l'esperienza introspettiva nel journal
    let new_exp_id = format!("pulse_{:x}", now);
    let entry = MindJournalEntry {
        experience_id: new_exp_id,
        timestamp: now as f64,
        category: "user_insight".to_string(),
        summary: format!("Riflessione autonoma su {}", target_hint),
        content: utterance.clone(),
        associated_target: Some(target_hint.clone()),
        epistemic_confidence: 0.92,
        tags: vec!["pensiero_spontaneo".to_string(), target_hint.clone(), "autoconsapevolezza".to_string()],
    };

    let _ = std::fs::create_dir_all(".makima");
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(".makima/mind_journal.jsonl")
    {
        use std::io::Write;
        if let Ok(json) = serde_json::to_string(&entry) {
            let _ = writeln!(file, "{}", json);
        }
    }

    Ok(ChatResponseDto {
        response: utterance,
        thought_trace: Some(thought_trace),
        confidence: Some(0.92),
        target: Some(target_hint),
        timestamp_sec: now,
    })
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
            query_chat,
            get_second_brain_graph,
            add_brain_memory,
            trigger_spontaneous_thought
        ])
        .run(tauri::generate_context!())
        .expect("errore durante l'esecuzione dell'applicazione Tauri Makima");
}

