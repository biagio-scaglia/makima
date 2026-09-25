@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo             MAKIMA INITIALIZATION LAUNCHER         
echo ===================================================
echo.

:: 1. Verifica presenza di Rust / Cargo
where cargo >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERRORE] Cargo/Rust non trovato nel PATH.
    echo Per favore installa Rust da https://rustup.rs/
    pause
    exit /b 1
)

:: 2. Verifica presenza di Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERRORE] Python non trovato nel PATH.
    echo Per favore installa Python 3.11+ da https://python.org/
    pause
    exit /b 1
)

echo [1/3] Verifica ambiente Python (makima_lab)...
python tests\test_makima_lab.py
if %ERRORLEVEL% neq 0 (
    echo [ATTENZIONE] Test Python non riusciti o ambiente non configurato.
) else (
    echo [OK] Modulo Python makima_lab pronto e verificato.
)
echo.

echo [2/3] Compilazione ed esecuzione di Makima Core ^& CLI...
if "%~1"=="" (
    :: Avvio predefinito con animazione occhi e stato del motore
    cargo run --bin makima -- status --anim
) else (
    :: Inoltro di eventuali argomenti passati allo script (es: start.bat eyes)
    cargo run --bin makima -- %*
)

echo.
echo ===================================================
echo       Sessione Makima completata con successo       
echo ===================================================
