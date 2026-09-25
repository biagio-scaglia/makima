@echo off
setlocal enabledelayedexpansion

:: Imposta titolo della finestra della console
title Makima - Interpretable Probabilistic Forecasting System

:: Configura PYTHONPATH per includere la cartella python/
set "PYTHONPATH=%~dp0python;%PYTHONPATH%"

echo ===================================================
echo             MAKIMA INITIALIZATION LAUNCHER         
echo ===================================================
echo.

:: 1. Verifica presenza di Rust / Cargo
where cargo >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERRORE] Cargo/Rust non trovato nel PATH.
    echo Per favore installa Rust da https://rustup.rs/
    echo.
    pause
    exit /b 1
)

:: 2. Verifica presenza di Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERRORE] Python non trovato nel PATH.
    echo Per favore installa Python 3.11+ da https://python.org/
    echo.
    pause
    exit /b 1
)

echo [1/2] Verifica ambiente Python (makima_lab)...
python tests\test_makima_lab.py
if %ERRORLEVEL% neq 0 (
    echo [ATTENZIONE] Test Python non riusciti o ambiente non configurato.
) else (
    echo [OK] Modulo Python makima_lab verificato.
)
echo.

echo [2/2] Avvio del motore Makima...
cargo run --bin makima -- status --anim

echo.
echo ===================================================
echo               CONSOLE INTERATTIVA MAKIMA           
echo ===================================================
echo Comandi disponibili:
echo   - predict ^<target^>      : Calcola previsione probabilistica
echo   - observe ^<target^> ^<v^> : Registra nuova evidenza storica (1/0)
echo   - outcome ^<target^> ^<v^> : Registra esito reale (Ground Truth)
echo   - evaluate              : Mostra scorecard e Brier Skill Score
echo   - poisson ^<lambda^>      : Calcola distribuzione temporale Poisson
echo   - bernoulli ^<p^>         : Calcola momenti ed Entropia di Shannon
echo   - status                : Mostra lo stato del core engine Rust
echo   - eyes                  : Esegue l'animazione degli occhi
echo   - lab                   : Avvia la console Python Lab
echo   - help                  : Mostra la guida comandi
echo   - exit                  : Chiude la sessione
echo ===================================================
echo.

:INTERACTIVE_LOOP
set "USER_INPUT="
set /p USER_INPUT="makima> "

if /i "!USER_INPUT!"=="exit" goto END
if /i "!USER_INPUT!"=="quit" goto END
if /i "!USER_INPUT!"=="q" goto END
if "!USER_INPUT!"=="" goto INTERACTIVE_LOOP

if /i "!USER_INPUT!"=="lab" (
    python -m makima_lab
    echo.
    goto INTERACTIVE_LOOP
)
if /i "!USER_INPUT!"=="python" (
    python -m makima_lab
    echo.
    goto INTERACTIVE_LOOP
)
if /i "!USER_INPUT!"=="py" (
    python -m makima_lab
    echo.
    goto INTERACTIVE_LOOP
)

cargo run --bin makima -- !USER_INPUT!
echo.
goto INTERACTIVE_LOOP

:END
echo.
echo Chiusura sessione Makima.
echo Premi un tasto per chiudere la finestra...
pause >nul
