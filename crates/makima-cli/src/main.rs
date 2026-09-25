//! CLI di Makima per l'interazione da riga di comando con il motore probabilistico.

mod eyes;

use makima_core::{
    Bernoulli, DiscreteDistribution, Distribution, Forecast, MakimaDb, MakimaEngine, MakimaStore,
    PoissonDistribution, Scoring,
};
use std::env;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

fn current_unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn load_engine_from_store() -> MakimaEngine {
    let mut engine = MakimaEngine::new();
    let db_path = MakimaDb::default_path();
    let store_path = MakimaStore::default_path();

    // 1. Inizializza o carica lo store JSON iniziale se necessario
    let store =
        MakimaStore::load_or_init(&store_path).unwrap_or_else(|_| MakimaStore::sample_store());

    // 2. Apri e sincronizza con SQLite WAL
    match MakimaDb::open(&db_path) {
        Ok(db) => {
            let _ = db.sync_from_store(&store);
            if let Err(err) = db.load_into_engine(&mut engine) {
                eprintln!("[ATTENZIONE] Errore lettura SQLite ({err}). Caricamento da JSON.");
                store.apply_to_engine(&mut engine);
            }
        }
        Err(err) => {
            eprintln!("[ATTENZIONE] Impossibile aprire SQLite DB ({err}). Caricamento da JSON.");
            store.apply_to_engine(&mut engine);
        }
    }
    engine
}

fn save_engine_to_store(engine: &MakimaEngine) {
    let store_path = MakimaStore::default_path();
    let store = MakimaStore::from_engine(engine);
    if let Err(err) = store.save(&store_path) {
        eprintln!("[ATTENZIONE] Impossibile salvare lo stato in .makima/store.json: {err}");
    }

    let db_path = MakimaDb::default_path();
    if let Ok(db) = MakimaDb::open(&db_path) {
        let _ = db.sync_from_store(&store);
    }
}

fn print_help() {
    println!("Makima - Interpretable Probabilistic Forecasting System\n");
    println!("UTILIZZO:");
    println!("    makima <COMANDO> [OPZIONI]\n");
    println!("COMANDI PREVISIONALI & DOMINIO:");
    println!(
        "    query <frase>         Interroga il sistema in linguaggio naturale (NLP Pipeline)"
    );
    println!(
        "    tell <testo>          Confida un fatto/abitudine a Makima (Neural Mind + SQLite)"
    );
    println!("    journal <testo>       Registra una riflessione o evento nella memoria di Makima");
    println!("    predict <target>      Genera una previsione probabilistica interpretabile");
    println!("    observe <target> <v>  Registra un'osservazione storica (1/0 o true/false)");
    println!("    outcome <target> <v>  Registra l'esito reale (Ground Truth) e valuta la stima");
    println!("    targets               Mostra la dashboard di tutti i target monitorati");
    println!("    mail                  Genera il bollettino previsionale Laplace Mail");
    println!("    evaluate              Mostra il report di accuratezza e Brier Skill Score");
    println!("    status                Mostra lo stato diagnostico del motore e del sistema\n");
    println!("COMANDI COGNITIVI & NEURALI:");
    println!(
        "    neural <frase>        Ispezione della percezione neurale (Self-Attention & Priors)"
    );
    println!("    memory                Visualizza lo stato della memoria latente utente\n");
    println!("COMANDI MATEMATICI & PROBABILISTICI:");
    println!("    poisson <lambda>      Calcola distribuzione di frequenza temporale Poisson");
    println!(
        "    bernoulli <p>         Calcola momenti ed Entropia di Shannon per eventi binari\n"
    );
    println!("COMANDI TELEMETRIA REALE & DAEMON:");
    println!("    sync-git              Sincronizza cronologia Git reale nel motore Makima");
    println!("    daemon [sec]          Avvia il daemon in background per il monitoraggio Git continuo\n");
    println!("STRUMENTI & AMBIENTI:");
    println!("    eyes                  Mostra il ritratto ASCII di Makima");
    println!("    lab                   Avvia il laboratorio scientifico interattivo Python");
    println!("    help                  Mostra questa guida di supporto\n");
    println!("OPZIONI:");
    println!("    --anim                Abilita l'animazione degli occhi");
    println!("    -h, --help            Mostra la guida");
    println!("    -V, --version         Mostra la versione di Makima");
}

fn print_version() {
    println!("makima {}", env!("CARGO_PKG_VERSION"));
}

fn render_ascii_density_bar(prob: f64, width: usize) -> String {
    let pos = (prob * ((width - 1) as f64)).round() as usize;
    let mut bar = vec!['-'; width];
    if pos < width {
        bar[pos] = '*';
    }
    format!("[{}]", bar.into_iter().collect::<String>())
}

fn handle_status(_animated: bool) {
    println!();
    eyes::print_static_eyes();
    println!();

    let engine = load_engine_from_store();
    let status = engine.status();

    println!("========================================");
    println!("           Makima Engine Status         ");
    println!("========================================");
    println!("Versione Core:       {}", status.version);
    println!("Stato Operativo:     {}", status.state);
    println!("Osservazioni Totali: {}", status.total_observations);
    println!("Esiti Valutati:      {}", status.total_outcomes);
    println!("Target Tracciati:    {}", engine.tracked_targets().len());
    println!("Storage Persistente: .makima/store.json");
    println!("Architettura:        Ibrida (Rust Core + Python Lab)");
    println!("========================================");
}

fn handle_targets() {
    let engine = load_engine_from_store();
    let summaries = engine.target_summaries();

    println!("\n=========================================================================================");
    println!(
        "                           MAKIMA MULTI-TARGET DASHBOARD                                 "
    );
    println!(
        "========================================================================================="
    );
    println!(
        "Target             | Evidenze (S/F) | E[P] Prob | Varianza | Entropia | Rate Stimato"
    );
    println!(
        "-------------------+----------------+-----------+----------+----------+-----------------"
    );

    if summaries.is_empty() {
        println!("Nessun target registrato nelle evidenze. Usa 'makima observe <target> <1|0>'.");
    } else {
        for s in &summaries {
            println!(
                "{:<18} | {:2} ({:2}/{:2})    | {:5.1}%    | {:.6} | {:.4} bit| ~{:.2} ev/giorno",
                s.target,
                s.observations_count,
                s.success_count,
                s.failure_count,
                s.probability.value() * 100.0,
                s.uncertainty_variance,
                s.entropy_bits,
                s.estimated_daily_rate
            );
        }
    }
    println!("=========================================================================================\n");
}

fn handle_mail() {
    let engine = load_engine_from_store();
    let ts = current_unix_timestamp();
    let mail = engine.generate_laplace_mail(ts);
    println!("\n{mail}\n");
}

fn handle_predict(target: &str) {
    let engine = load_engine_from_store();
    let forecast: Forecast = engine.predict_target(target);

    println!("\n============================================================");
    println!("               MAKIMA PROBABILISTIC FORECAST                ");
    println!("============================================================");
    println!("Target:               {}", forecast.target);
    println!(
        "Probabilità Stimata:  {:.2}%  (E[P] = {:.4})",
        forecast.probability.value() * 100.0,
        forecast.probability.value()
    );
    println!(
        "Densità di Stima:     {} (0.0 -> 1.0)",
        render_ascii_density_bar(forecast.probability.value(), 36)
    );
    println!("Incertezza (Var):     {:.6}", forecast.uncertainty_variance);
    println!("Entropia Informativa: {:.4} bit", forecast.entropy_bits);
    println!(
        "Prior Bayesiano:      Beta(alpha={:.2}, beta={:.2})",
        forecast.prior.alpha(),
        forecast.prior.beta()
    );
    println!(
        "Posterior Aggiornato: Beta(alpha={:.2}, beta={:.2})",
        forecast.posterior.alpha(),
        forecast.posterior.beta()
    );
    println!(
        "Evidenze Rilevate:    {} osservazioni storiche",
        forecast.evidence_count
    );

    if forecast.evidence_ids.is_empty() {
        println!("Tracciamento Prove:   [Nessuna evidenza - Prior non-informativo]");
    } else {
        let ids_str: Vec<String> = forecast
            .evidence_ids
            .iter()
            .map(|id| id.0.to_string())
            .collect();
        println!("Tracciamento Prove:   [ID: {}]", ids_str.join(", "));
    }
    println!("============================================================\n");
}

fn handle_observe(target: &str, value_str: &str) {
    let is_success = match value_str.to_lowercase().as_str() {
        "1" | "true" | "t" | "success" | "ok" | "s" => true,
        "0" | "false" | "f" | "failure" | "fail" => false,
        _ => {
            eprintln!("Errore: valore non valido '{value_str}'. Usa 1/0 o true/false.");
            return;
        }
    };

    let mut engine = load_engine_from_store();
    let ts = current_unix_timestamp();
    let id = engine.record_binary(target, is_success, ts);
    save_engine_to_store(&engine);
    let forecast = engine.predict_target(target);

    println!("\n[OK] Nuova osservazione registrata con successo e salvata nello storage!");
    println!("- ID Assegnato:   {}", id.0);
    println!("- Target:         {target}");
    println!(
        "- Esito:          {}",
        if is_success {
            "Successo (1)"
        } else {
            "Insuccesso (0)"
        }
    );
    println!(
        "- Nuovo Posterior: Beta(alpha={:.2}, beta={:.2}) -> E[P] = {:.2}%",
        forecast.posterior.alpha(),
        forecast.posterior.beta(),
        forecast.probability.value() * 100.0
    );
    println!("- File Storage:   .makima/store.json");
    println!();
}

fn handle_outcome(target: &str, value_str: &str) {
    let actual_occurred = match value_str.to_lowercase().as_str() {
        "1" | "true" | "t" | "success" | "ok" | "s" => true,
        "0" | "false" | "f" | "failure" | "fail" => false,
        _ => {
            eprintln!("Errore: valore esito non valido '{value_str}'. Usa 1/0 o true/false.");
            return;
        }
    };

    let mut engine = load_engine_from_store();
    let forecast = engine.predict_target(target);
    let brier = Scoring::brier_score(forecast.probability, actual_occurred);
    let log_loss = Scoring::log_loss(forecast.probability, actual_occurred);

    let ts = current_unix_timestamp();
    engine.record_outcome(target, actual_occurred, ts);
    save_engine_to_store(&engine);

    println!("\n============================================================");
    println!("            VALUTAZIONE GROUND TRUTH (ESITO REALE)          ");
    println!("============================================================");
    println!("Target:               {target}");
    println!(
        "Previsione Emessa:    {:.2}%",
        forecast.probability.value() * 100.0
    );
    println!(
        "Esito Reale:          {}",
        if actual_occurred {
            "Si e' Verificato (1)"
        } else {
            "Non si e' Verificato (0)"
        }
    );
    println!(
        "Brier Score Singolo:  {:.4}  (0.0 = perfetto, 0.25 = baseline casuale)",
        brier
    );
    println!("Log Loss Singola:     {:.4}", log_loss);
    println!("Stato Salvato in:     .makima/store.json");
    println!("============================================================\n");
}

fn handle_evaluate() {
    let engine = load_engine_from_store();
    match engine.evaluate_performance() {
        Some(report) => println!("\n{report}\n"),
        None => println!(
            "\nNessuna previsione confrontata con esiti reali disponibile per la valutazione.\n"
        ),
    }
}

fn handle_poisson(lambda_str: &str) {
    match lambda_str.parse::<f64>() {
        Ok(lambda) => match PoissonDistribution::new(lambda) {
            Ok(dist) => {
                println!("\n============================================================");
                println!("           DISTRIBUZIONE DI POISSON (TEMPORALE)             ");
                println!("============================================================");
                println!(
                    "Parametro Tasso (lambda): {:.3} eventi attesi/intervallo",
                    dist.lambda()
                );
                println!("Media E[X]:               {:.3}", dist.mean());
                println!("Varianza Var(X):          {:.3}", dist.variance());
                println!("Deviazione Standard:      {:.3}", dist.std_dev());
                println!("------------------------------------------------------------");
                println!("Funzione di Massa PMF P(X = k):");
                for k in 0..=6 {
                    let pmf = dist.pmf(k);
                    let cdf = dist.cdf(k);
                    println!(
                        "  k = {:2} | P(X = {:2}) = {:6.2}% | CDF P(X <= {:2}) = {:6.2}%",
                        k,
                        k,
                        pmf.value() * 100.0,
                        k,
                        cdf.value() * 100.0
                    );
                }
                println!("============================================================\n");
            }
            Err(err) => eprintln!("Errore: {err}"),
        },
        Err(_) => eprintln!("Errore: '{lambda_str}' non e' un numero valido."),
    }
}

fn handle_bernoulli(p_str: &str) {
    match p_str.parse::<f64>() {
        Ok(p) => match Bernoulli::new(p) {
            Ok(dist) => {
                println!("\n============================================================");
                println!("         DISTRIBUZIONE DI BERNOULLI (EVENTO BINARIO)        ");
                println!("============================================================");
                println!(
                    "Probabilità Successo (p): {:.4} ({:.2}%)",
                    dist.p().value(),
                    dist.p().value() * 100.0
                );
                println!(
                    "Probabilità Insuccesso (q): {:.4} ({:.2}%)",
                    dist.q().value(),
                    dist.q().value() * 100.0
                );
                println!("Valore Atteso Media:      {:.4}", dist.mean());
                println!("Varianza p*(1-p):         {:.4}", dist.variance());
                println!("Entropia di Shannon H(p): {:.4} bit", dist.entropy_bits());
                println!("============================================================\n");
            }
            Err(err) => eprintln!("Errore: {err}"),
        },
        Err(_) => eprintln!("Errore: '{p_str}' non e' un numero valido."),
    }
}

fn handle_query(query_text: &str) -> ExitCode {
    let mut cmd = std::process::Command::new("python");
    cmd.env("PYTHONPATH", "python");
    cmd.args([
        "-c",
        "import sys; from makima_lab.nlp import SemanticForecastPipeline; p = SemanticForecastPipeline(); print('\\n' + p.execute(sys.argv[1]).format_report())",
        query_text,
    ]);
    match cmd.status() {
        Ok(status) if status.success() => ExitCode::SUCCESS,
        Ok(_) => ExitCode::FAILURE,
        Err(err) => {
            eprintln!("Impossibile eseguire il modulo Python NLP: {err}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_help();
        return ExitCode::SUCCESS;
    }

    match args[1].as_str() {
        "query" | "nlp" => {
            if args.len() < 3 {
                eprintln!("Uso: makima query <frase in linguaggio naturale>");
                eprintln!("Esempio: makima query Quando rilascerò il prossimo framework?");
                ExitCode::FAILURE
            } else {
                let query_text = args[2..].join(" ");
                handle_query(&query_text)
            }
        }
        "targets" | "list" => {
            handle_targets();
            ExitCode::SUCCESS
        }
        "mail" | "bulletin" | "laplace" => {
            handle_mail();
            ExitCode::SUCCESS
        }
        "status" => {
            let animated = args.iter().any(|arg| arg == "--anim");
            handle_status(animated);
            ExitCode::SUCCESS
        }
        "predict" | "forecast" => {
            if args.len() < 3 {
                eprintln!("Uso: makima predict <target>");
                eprintln!("Esempio: makima predict framework_release");
                ExitCode::FAILURE
            } else {
                handle_predict(&args[2]);
                ExitCode::SUCCESS
            }
        }
        "observe" | "record" => {
            if args.len() < 4 {
                eprintln!("Uso: makima observe <target> <1|0|true|false>");
                eprintln!("Esempio: makima observe framework_release 1");
                ExitCode::FAILURE
            } else {
                handle_observe(&args[2], &args[3]);
                ExitCode::SUCCESS
            }
        }
        "outcome" | "actual" => {
            if args.len() < 4 {
                eprintln!("Uso: makima outcome <target> <1|0|true|false>");
                eprintln!("Esempio: makima outcome framework_release 1");
                ExitCode::FAILURE
            } else {
                handle_outcome(&args[2], &args[3]);
                ExitCode::SUCCESS
            }
        }
        "evaluate" | "score" | "eval" => {
            handle_evaluate();
            ExitCode::SUCCESS
        }
        "poisson" => {
            if args.len() < 3 {
                eprintln!("Uso: makima poisson <lambda>");
                eprintln!("Esempio: makima poisson 3.0");
                ExitCode::FAILURE
            } else {
                handle_poisson(&args[2]);
                ExitCode::SUCCESS
            }
        }
        "bernoulli" => {
            if args.len() < 3 {
                eprintln!("Uso: makima bernoulli <p>");
                eprintln!("Esempio: makima bernoulli 0.75");
                ExitCode::FAILURE
            } else {
                handle_bernoulli(&args[2]);
                ExitCode::SUCCESS
            }
        }
        "tell" | "journal" => {
            if args.len() < 3 {
                eprintln!("Uso: makima tell <testo o fatto da confidare>");
                eprintln!("Esempio: makima tell \"Oggi ho completato la sessione di palestra\"");
                ExitCode::FAILURE
            } else {
                let full_text = args[2..].join(" ");
                let mut cmd = std::process::Command::new("python");
                cmd.env("PYTHONPATH", "python");
                cmd.args(["-m", "makima_lab", "tell", &full_text]);
                match cmd.status() {
                    Ok(status) if status.success() => ExitCode::SUCCESS,
                    Ok(_) => ExitCode::FAILURE,
                    Err(err) => {
                        eprintln!("Errore esecuzione pipeline neurale Python: {err}");
                        ExitCode::FAILURE
                    }
                }
            }
        }
        "neural" => {
            if args.len() < 3 {
                eprintln!("Uso: makima neural <frase da ispezionare>");
                ExitCode::FAILURE
            } else {
                let full_text = args[2..].join(" ");
                let mut cmd = std::process::Command::new("python");
                cmd.env("PYTHONPATH", "python");
                cmd.args(["-m", "makima_lab", "neural", &full_text]);
                match cmd.status() {
                    Ok(status) if status.success() => ExitCode::SUCCESS,
                    Ok(_) => ExitCode::FAILURE,
                    Err(err) => {
                        eprintln!("Errore esecuzione analisi neurale: {err}");
                        ExitCode::FAILURE
                    }
                }
            }
        }
        "memory" => {
            let mut cmd = std::process::Command::new("python");
            cmd.env("PYTHONPATH", "python");
            cmd.args(["-m", "makima_lab", "memory"]);
            match cmd.status() {
                Ok(status) if status.success() => ExitCode::SUCCESS,
                Ok(_) => ExitCode::FAILURE,
                Err(err) => {
                    eprintln!("Errore lettura memoria neurale: {err}");
                    ExitCode::FAILURE
                }
            }
        }
        "sync-git" | "git-sync" => {
            let mut cmd = std::process::Command::new("python");
            cmd.env("PYTHONPATH", "python");
            cmd.args(["-m", "makima_lab", "sync-git"]);
            match cmd.status() {
                Ok(status) if status.success() => ExitCode::SUCCESS,
                Ok(_) => ExitCode::FAILURE,
                Err(err) => {
                    eprintln!("Errore esecuzione sincronizzazione Git: {err}");
                    ExitCode::FAILURE
                }
            }
        }
        "daemon" | "watch" => {
            let interval = args.get(2).map(|s| s.as_str()).unwrap_or("15");
            let mut cmd = std::process::Command::new("python");
            cmd.env("PYTHONPATH", "python");
            cmd.args(["-m", "makima_lab", "daemon", interval]);
            match cmd.status() {
                Ok(status) if status.success() => ExitCode::SUCCESS,
                Ok(_) => ExitCode::FAILURE,
                Err(err) => {
                    eprintln!("Errore esecuzione Daemon: {err}");
                    ExitCode::FAILURE
                }
            }
        }
        "eyes" | "anim" => {
            eyes::play_eye_animation(2);
            ExitCode::SUCCESS
        }
        "lab" => {
            println!("Avvio del laboratorio scientifico Python (makima_lab)...\n");
            let mut cmd = std::process::Command::new("python");
            cmd.env("PYTHONPATH", "python");
            cmd.args(["-m", "makima_lab"]);
            match cmd.status() {
                Ok(status) if status.success() => ExitCode::SUCCESS,
                Ok(_) => ExitCode::FAILURE,
                Err(err) => {
                    eprintln!("Impossibile avviare Python: {err}");
                    eprintln!("Assicurati che Python sia installato e presente nel PATH.");
                    ExitCode::FAILURE
                }
            }
        }
        "help" | "-h" | "--help" => {
            print_help();
            ExitCode::SUCCESS
        }
        "-V" | "--version" => {
            print_version();
            ExitCode::SUCCESS
        }
        unknown => {
            eprintln!("Errore: comando sconosciuto '{}'.\n", unknown);
            eprintln!("Esegui 'makima --help' per la lista dei comandi disponibili.");
            ExitCode::FAILURE
        }
    }
}
