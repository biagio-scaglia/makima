#!/usr/bin/env bash
# Script di verifica e controllo qualità locale per Makima (POSIX/Bash)
set -euo pipefail

echo "========================================"
echo "     Makima Quality Verification Suite  "
echo "========================================"

echo ""
echo "[1/5] Controllo Formattazione Rust (rustfmt)..."
cargo fmt --all -- --check
echo "-> rustfmt OK"

echo ""
echo "[2/5] Analisi Statica Rust (clippy)..."
cargo clippy --all-targets -- -D warnings
echo "-> clippy OK"

echo ""
echo "[3/5] Esecuzione Test Unitari Rust..."
cargo test --all
echo "-> cargo test OK"

echo ""
echo "[4/5] Verifica Esecuzione CLI (makima status)..."
cargo run --bin makima -- status
echo "-> makima CLI status OK"

echo ""
echo "[5/5] Esecuzione Test Python Suite..."
python -m unittest discover -s tests -p "test_*.py"
echo "-> python tests OK"

echo ""
echo "========================================"
echo "  Tutti i controlli sono stati superati!"
echo "========================================"
