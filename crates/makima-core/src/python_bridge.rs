//! Bridge subprocess verso il laboratorio Python (`makima_lab`).
//!
//! Usato da CLI e GUI per `parse-intent` e altri sidecar. Non contiene logica
//! probabilistica: solo I/O di processo con timeout e taxonomy di errori.

use crate::intent::StructuredIntent;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use std::{fmt, thread};

/// Timeout di default per le chiamate NLP (parse-intent).
pub const DEFAULT_PYTHON_TIMEOUT: Duration = Duration::from_secs(45);

/// Codici errore stabili per UX / logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeErrorCode {
    /// Impossibile avviare l'interprete Python.
    EPythonSpawn,
    /// Subprocess oltre il timeout.
    EPythonTimeout,
    /// Exit code non zero.
    EPythonExit,
    /// Stdout vuoto.
    EPythonEmpty,
    /// JSON StructuredIntent invalido / assente.
    EIntentParse,
    /// Errore I/O generico sul processo.
    EPythonIo,
}

impl BridgeErrorCode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EPythonSpawn => "E_PYTHON_SPAWN",
            Self::EPythonTimeout => "E_PYTHON_TIMEOUT",
            Self::EPythonExit => "E_PYTHON_EXIT",
            Self::EPythonEmpty => "E_PYTHON_EMPTY",
            Self::EIntentParse => "E_INTENT_PARSE",
            Self::EPythonIo => "E_PYTHON_IO",
        }
    }

    #[must_use]
    pub const fn hint(self) -> &'static str {
        match self {
            Self::EPythonSpawn => {
                "Verifica che `python` sia nel PATH e che makima_lab sia installato (`pip install -e .`)."
            }
            Self::EPythonTimeout => {
                "Il parser NLP ha superato il timeout; riprova o riduci il carico del modello embeddings."
            }
            Self::EPythonExit => "Controlla stderr del modulo makima_lab e le dipendenze Python.",
            Self::EPythonEmpty => "Il parser non ha prodotto output; usa `python -m makima_lab parse-intent ...`.",
            Self::EIntentParse => {
                "Lo stdout non contiene JSON StructuredIntent valido (contratto NLP rotto)."
            }
            Self::EPythonIo => "Errore di I/O sul subprocess Python.",
        }
    }
}

impl fmt::Display for BridgeErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

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

impl PythonBridgeError {
    #[must_use]
    pub const fn code(&self) -> BridgeErrorCode {
        match self {
            Self::Spawn(_) => BridgeErrorCode::EPythonSpawn,
            Self::Timeout(_) => BridgeErrorCode::EPythonTimeout,
            Self::NonZeroExit { .. } => BridgeErrorCode::EPythonExit,
            Self::EmptyStdout => BridgeErrorCode::EPythonEmpty,
            Self::Intent(_) => BridgeErrorCode::EIntentParse,
            Self::Io(_) => BridgeErrorCode::EPythonIo,
        }
    }
}

impl fmt::Display for PythonBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let code = self.code();
        match self {
            Self::Spawn(e) => write!(
                f,
                "[{code}] impossibile avviare Python: {e}\n  → {}",
                code.hint()
            ),
            Self::Timeout(d) => write!(
                f,
                "[{code}] timeout subprocess Python dopo {d:?}\n  → {}",
                code.hint()
            ),
            Self::NonZeroExit { code: exit, stderr } => write!(
                f,
                "[{code}] Python uscito con codice {:?}: {}\n  → {}",
                exit,
                stderr.trim(),
                code.hint()
            ),
            Self::EmptyStdout => {
                write!(
                    f,
                    "[{code}] subprocess Python senza stdout\n  → {}",
                    code.hint()
                )
            }
            Self::Intent(msg) => write!(f, "[{code}] {msg}\n  → {}", code.hint()),
            Self::Io(e) => write!(f, "[{code}] I/O bridge Python: {e}\n  → {}", code.hint()),
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
pub fn parse_structured_intent_default(query: &str) -> Result<StructuredIntent, PythonBridgeError> {
    parse_structured_intent(query, DEFAULT_PYTHON_TIMEOUT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_codes_are_stable() {
        assert_eq!(BridgeErrorCode::EPythonTimeout.as_str(), "E_PYTHON_TIMEOUT");
        let err = PythonBridgeError::EmptyStdout;
        assert_eq!(err.code(), BridgeErrorCode::EPythonEmpty);
        let msg = err.to_string();
        assert!(msg.contains("E_PYTHON_EMPTY"));
        assert!(msg.contains("→"));
    }
}
