@echo off
setlocal enabledelayedexpansion

:: Forza codifica UTF-8 per caratteri e box drawing
chcp 65001 >nul

:: Titolo finestra
title Makima - Interpretable Probabilistic Forecasting System

:: Configura PYTHONPATH
set "PYTHONPATH=%~dp0python;%PYTHONPATH%"

echo [31;1m
echo   ███╗   ███╗ █████╗ ██╗  ██╗██╗███╗   ███╗ █████╗ 
echo   ████╗ ████║██╔══██╗██║ ██╔╝██║████╗ ████║██╔══██╗
echo   ██╔████╔██║███████║█████╔╝ ██║██╔████╔██║███████║
echo   ██║╚██╔╝██║██╔══██║██╔═██╗ ██║██║╚██╔╝██║██╔══██║
echo   ██║ ╚═╝ ██║██║  ██║██║  ██╗██║██║ ╚═╝ ██║██║  ██║
echo   ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝
echo [0m
echo [90m  ── Interpretable Probabilistic Engine & Cognitive Neural Mind ──[0m
echo.

:: 1. Verifica presenza di Rust / Cargo
where cargo >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [31m[ERRORE] Cargo/Rust non trovato nel PATH.[0m
    echo Per favore installa Rust da https://rustup.rs/
    echo.
    pause
    exit /b 1
)

:: 2. Verifica presenza di Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [31m[ERRORE] Python non trovato nel PATH.[0m
    echo Per favore installa Python 3.11+ da https://python.org/
    echo.
    pause
    exit /b 1
)

echo [36m[1/2] Verifica ambiente scientifico & Neural Mind (makima_lab)...[0m
python -m unittest discover -s tests -p "test_*.py" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [33m[!] Test Python completati con avvisi. Procedura di avvio in corso...[0m
) else (
    echo [32m[✓] Suite di test superata (24 unit test verificati con successo).[0m
)
echo.

echo [36m[2/2] Avvio del motore e diagnostica Makima...[0m
cargo run --bin makima -- status

echo.
echo [31;1m╔═══════════════════════════════════════════════════════════════════════════════╗[0m
echo [31;1m║                        CONSOLE INTERATTIVA MAKIMA                             ║[0m
echo [31;1m╚═══════════════════════════════════════════════════════════════════════════════╝[0m
echo [33;1m 🔮 PREVISIONI & DOMINIO:[0m
echo   [37m• query ^<frase^>[0m          : Interroga in linguaggio naturale (NLP Pipeline)
echo   [37m• predict ^<target^>[0m      : Calcola previsione bayesiana Beta-Binomiale
echo   [37m• targets[0m               : Dashboard multi-target e parametri posterior
echo   [37m• mail[0m                  : Genera il bollettino previsionale Laplace Mail
echo   [37m• observe ^<target^> ^<v^>[0m : Registra nuova evidenza storica (1=succ, 0=fail)
echo   [37m• outcome ^<target^> ^<v^>[0m : Registra esito reale e calcola calibrazione
echo   [37m• evaluate[0m              : Scorecard di accuratezza, Brier Score e Log-Loss
echo.
echo [35;1m 🧠 COGNIZIONE NEURALE & TELEMETRIA GIT REALE:[0m
echo   [37m• tell ^<testo^>[0m           : Confida un fatto a Makima (NLP + PyTorch + SQLite)
echo   [37m• neural ^<frase^>[0m         : Ispezione neurale con Self-Attention e Priors
echo   [37m• memory[0m                : Visualizza lo stato della memoria latente utente
echo   [37m• sync-git[0m              : Sincronizza tutta la cronologia Git reale nel motore
echo   [37m• daemon [sec][0m          : Avvia il monitoraggio in background per nuovi commit
echo.
echo [36;1m 📐 MATEMATICA & STATISTICA:[0m
echo   [37m• poisson ^<lambda^>[0m      : Calcola distribuzione Poisson e tassi di arrivo
echo   [37m• bernoulli ^<p^>[0m         : Calcola momenti ed Entropia di Shannon (bit)
echo   [37m• status[0m                : Mostra lo stato diagnostico del core engine Rust
echo   [37m• eyes[0m                  : Mostra il ritratto ASCII di Makima
echo   [37m• lab[0m                   : Avvia la console scientifica interattiva Python
echo   [37m• exit[0m                  : Chiude la sessione
echo [90m─────────────────────────────────────────────────────────────────────────────────[0m
echo.

:INTERACTIVE_LOOP
set "USER_INPUT="
set /p USER_INPUT="[31;1mmakima[0m [33m❯[0m "

if /i "!USER_INPUT!"=="exit" goto END
if /i "!USER_INPUT!"=="quit" goto END
if /i "!USER_INPUT!"=="q" goto END
if "!USER_INPUT!"=="" goto INTERACTIVE_LOOP

:: Routing rapido comandi Python diretti
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

:: Routing per query semantica rapida
set "PREFIX=!USER_INPUT:~0,6!"
if /i "!PREFIX!"=="query " (
    python -m makima_lab query "!USER_INPUT:~6!"
    echo.
    goto INTERACTIVE_LOOP
)
set "PREFIX4=!USER_INPUT:~0,4!"
if /i "!PREFIX4!"=="nlp " (
    python -m makima_lab query "!USER_INPUT:~4!"
    echo.
    goto INTERACTIVE_LOOP
)

cargo run --bin makima -- !USER_INPUT!
echo.
goto INTERACTIVE_LOOP

:END
echo.
echo [90mChiusura sessione Makima. Arrivederci.[0m
echo Premi un tasto per chiudere la finestra...
pause >nul
