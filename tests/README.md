# Makima Test Suite

Questa directory è dedicata ai test di integrazione end-to-end, test di regressione probabilistica e verifiche cross-language (Rust/Python).

---

## Struttura dei Test

- **Test Unitari Rust**: risiedono nei singoli crate (`crates/makima-core/src/`, `crates/makima-cli/src/`).
- **Test di Integrazione Rust**: cartelle `tests/` all'interno dei rispettivi crate.
- **Test Laboratorio Python**: verifiche sui moduli scientifici e di preprocessing in `tests/` o nel package `python/makima_lab/`.
