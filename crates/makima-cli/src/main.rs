//! CLI di Makima per l'interazione da riga di comando con il motore probabilistico.

mod eyes;

use makima_core::MakimaEngine;
use std::env;
use std::process::ExitCode;

fn print_help() {
    println!("Makima - Interpretable Probabilistic Forecasting System\n");
    println!("UTILIZZO:");
    println!("    makima <COMANDO> [OPZIONI]\n");
    println!("COMANDI:");
    println!("    status        Mostra lo stato diagnostico del motore e del sistema");
    println!("    eyes          Esegue l'animazione ASCII dello sguardo di Makima");
    println!("    help          Mostra questa guida di supporto\n");
    println!("OPZIONI:");
    println!("    --anim        Abilita l'animazione di apertura degli occhi prima dello status");
    println!("    -h, --help    Mostra la guida");
    println!("    -V, --version Mostra la versione di Makima");
}

fn print_version() {
    println!("makima {}", env!("CARGO_PKG_VERSION"));
}

fn handle_status(animated: bool) {
    if animated {
        eyes::play_eye_animation(1);
    } else {
        println!();
        eyes::print_static_eyes();
        println!();
    }

    let engine = MakimaEngine::new();
    let status = engine.status();

    println!("========================================");
    println!("           Makima Engine Status         ");
    println!("========================================");
    println!("Versione Core:       {}", status.version);
    println!("Stato Operativo:     {}", status.state);
    println!("Osservazioni Totali: {}", status.total_observations);
    println!("Architettura:        Ibrida (Rust Core + Python Lab)");
    println!("========================================");
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_help();
        return ExitCode::SUCCESS;
    }

    match args[1].as_str() {
        "status" => {
            let animated = args.iter().any(|arg| arg == "--anim");
            handle_status(animated);
            ExitCode::SUCCESS
        }
        "eyes" | "anim" => {
            eyes::play_eye_animation(2);
            ExitCode::SUCCESS
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
