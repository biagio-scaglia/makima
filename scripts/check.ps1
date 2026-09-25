# Script di verifica e controllo qualità locale per Makima (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "     Makima Quality Verification Suite  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n[1/5] Controllo Formattazione Rust (rustfmt)..." -ForegroundColor Yellow
cargo fmt --all -- --check
Write-Host "-> rustfmt OK" -ForegroundColor Green

Write-Host "`n[2/5] Analisi Statica Rust (clippy)..." -ForegroundColor Yellow
cargo clippy --all-targets -- -D warnings
Write-Host "-> clippy OK" -ForegroundColor Green

Write-Host "`n[3/5] Esecuzione Test Unitari Rust..." -ForegroundColor Yellow
cargo test --all
Write-Host "-> cargo test OK" -ForegroundColor Green

Write-Host "`n[4/5] Verifica Esecuzione CLI (makima status)..." -ForegroundColor Yellow
cargo run --bin makima -- status
Write-Host "-> makima CLI status OK" -ForegroundColor Green

Write-Host "`n[5/5] Esecuzione Test Python Suite..." -ForegroundColor Yellow
python -m unittest discover -s tests -p "test_*.py"
Write-Host "-> python tests OK" -ForegroundColor Green

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Tutti i controlli sono stati superati! " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
