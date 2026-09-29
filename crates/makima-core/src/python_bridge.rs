//! Bridge subprocess verso il laboratorio Python (`makima_lab`).
//!
//! Usato da CLI e GUI per `parse-intent` e altri sidecar. Non contiene logica
//! probabilistica: solo I/O di processo con timeout.

use crate::intent::StructuredIntent;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use std::{fmt, thread};

/// Timeout di default per le chiamate NLP (parse-intent).
pub const DEFAULT_PYTHON_TIMEOUT: Duration = Duration::from_secs(45);

/// Errori del bridge Rust → Python.
#[derive(Debug)]
pub enum PythonBridgeError {
    Spawn(std::io::Error),
    Timeout(Duration),
    NonZeroExit { code: Option<i32>, stderr: String },
    EmptyStdout,
    Intent(String),
    Io(std::io::Error),
}

impl fmt::Display for PythonBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn(e) => write!(f, "impossibile avviare Python: {e}"),
            Self::Timeout(d) => write!(f, "timeout subprocess Python dopo {d:?}"),
            Self::NonZeroExit { code, stderr } => {
                write!(
                    f,
                    "Python uscito con codice {:?}: {}",
                    code,
                    stderr.trim()
                )
            }
            Self::EmptyStdout => write!(f, "subprocess Python senza stdout"),
            Self::Intent(msg) => write!(f, "{msg}"),
            Self::Io(e) => write!(f, "I/O bridge Python: {e}"),
        }
    }
}

impl std::error::Error for PythonBridgeError {}

/// Esegue `python -m makima_lab …` con `PYTHONPATH=python` e timeout.
pub fn run_makima_lab(args: &[&str], timeout: Duration) -> Result<String, PythonBridgeError> {
    let mut cmd = Command::new("python");
    cmd.env("PYTHONPATH", "python");
    cmd.args(["-m", "makima_lab"]);
    cmd.args(args);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(PythonBridgeError::Spawn)?;
    let start = Instant::now();

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout = String::new();
                let mut stderr = String::new();
                if let Some(mut out) = child.stdout.take() {
                    out.read_to_string(&mut stdout)
                        .map_err(PythonBridgeError::Io)?;
                }
                if let Some(mut err) = child.stderr.take() {
                    err.read_to_string(&mut stderr)
                        .map_err(PythonBridgeError::Io)?;
                }

                if !status.success() {
                    return Err(PythonBridgeError::NonZeroExit {
                        code: status.code(),
                        stderr,
                    });
                }
                if stdout.trim().is_empty() {
                    return Err(PythonBridgeError::EmptyStdout);
                }
                return Ok(stdout);
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(PythonBridgeError::Timeout(timeout));
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(e) => return Err(PythonBridgeError::Io(e)),
        }
    }
}

/// Invoca `parse-intent` e restituisce lo [`StructuredIntent`] tipizzato.
pub fn parse_structured_intent(
    query: &str,
    timeout: Duration,
) -> Result<StructuredIntent, PythonBridgeError> {
    let stdout = run_makima_lab(&["parse-intent", query], timeout)?;
    StructuredIntent::from_parser_stdout(&stdout).map_err(PythonBridgeError::Intent)
}

/// Scorciatoia con timeout di default.
pub fn parse_structured_intent_default(
    query: &str,
) -> Result<StructuredIntent, PythonBridgeError> {
    parse_structured_intent(query, DEFAULT_PYTHON_TIMEOUT)
}
