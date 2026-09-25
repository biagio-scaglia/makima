@echo off
setlocal enabledelayedexpansion

chcp 65001 >nul
title Makima - Interpretable Probabilistic Forecasting System (Avvio Rapido)
set "PYTHONPATH=%~dp0python;%PYTHONPATH%"

echo.
echo ========================================================================
echo   M A K I M A   -   A V V I O   R A P I D O   I S T A N T A N E O
echo   Interpretable Probabilistic Engine and Cognitive Neural Mind
echo ========================================================================
echo.

where python >nul 2>nul
if errorlevel 1 goto PYTHON_ERROR

rem Rilevamento eseguibile precompilato per avvio istantaneo (<200ms)
set "MAKIMA_BIN="
if exist "%~dp0target\release\makima.exe" (
    set "MAKIMA_BIN=%~dp0target\release\makima.exe"
) else if exist "%~dp0target\debug\makima.exe" (
    set "MAKIMA_BIN=%~dp0target\debug\makima.exe"
) else (
    where cargo >nul 2>nul
    if errorlevel 1 goto CARGO_ERROR
    echo [*] Compilazione iniziale del binario Makima in corso...
    cargo build --bin makima
    if exist "%~dp0target\debug\makima.exe" (
        set "MAKIMA_BIN=%~dp0target\debug\makima.exe"
    ) else (
        set "MAKIMA_BIN=cargo run --bin makima --"
    )
)

echo [OK] Motore pronto. Avvio immediato della diagnostica Makima...
"%MAKIMA_BIN%" status

echo.
echo ========================================================================
echo                       CONSOLE INTERATTIVA MAKIMA                        
echo ========================================================================
echo  * PREVISIONI E DOMINIO:
echo    - query ^<frase^>          : Interroga in linguaggio naturale (NLP Pipeline)
echo    - predict ^<target^>      : Calcola previsione bayesiana Beta-Binomiale
echo    - forecasts             : Registro storico del ciclo di vita delle previsioni
echo    - targets               : Dashboard multi-target e parametri posterior
echo    - mail                  : Genera il bollettino previsionale Laplace Mail
echo    - observe ^<target^> ^<v^> : Registra nuova evidenza storica (1=succ, 0=fail)
echo    - outcome ^<target^> ^<v^> : Registra esito reale e calcola calibrazione
echo    - evaluate              : Scorecard di accuratezza, Brier Score ed ECE
echo.
echo  * RAGIONAMENTO COGNITIVO ED SLM (QWEN 2.5):
echo    - explain [target]      : Spiegazione in linguaggio naturale del forecast (Qwen 2.5)
echo    - digest                : Bollettino esecutivo Laplace Digest generato da SLM
echo    - chat ^<messaggio^>      : Conversazione analitica diretta con Makima
echo.
echo  * VALIDAZIONE SCIENTIFICA E BENCHMARK:
echo    - benchmark             : Esegue il benchmark comparativo completo su 5,000 campioni
echo    - ablation              : Esegue l'Ablation Study su tutti i sottosistemi
echo    - test                  : Esegue la suite completa di unit test (su richiesta)
echo.
echo  * COGNIZIONE NEURALE E TELEMETRIA GIT:
echo    - tell ^<testo^>           : Confida un fatto a Makima (NLP + PyTorch + SQLite)
echo    - neural ^<frase^>         : Ispezione neurale con Self-Attention e Priors
echo    - memory                : Visualizza lo stato della memoria latente utente
echo    - sync-git              : Sincronizza tutta la cronologia Git reale nel motore
echo    - daemon [sec]          : Avvia il monitoraggio in background per nuovi commit
echo.
echo  * MATEMATICA E STATISTICA:
echo    - poisson ^<lambda^>      : Calcola distribuzione Poisson e tassi di arrivo
echo    - bernoulli ^<p^>         : Calcola momenti ed Entropia di Shannon (bit)
echo    - status                : Mostra lo stato diagnostico del core engine Rust
echo    - eyes                  : Mostra il ritratto ASCII di Makima
echo    - lab                   : Avvia la console scientifica interattiva Python
echo    - exit                  : Chiude la sessione
echo ========================================================================
echo.

set "EMPTY_COUNT=0"

:INTERACTIVE_LOOP
set "USER_INPUT="
set /p USER_INPUT="makima > "

if not defined USER_INPUT (
    set /a EMPTY_COUNT+=1
    if !EMPTY_COUNT! geq 2 goto END
    goto INTERACTIVE_LOOP
)
set "EMPTY_COUNT=0"

rem Riconoscimento flessibile comandi di uscita (anche con spazi finali)
if /i "!USER_INPUT:~0,4!"=="exit" goto END
if /i "!USER_INPUT:~0,4!"=="quit" goto END
if /i "!USER_INPUT!"=="q" goto END
if /i "!USER_INPUT:~0,2!"=="q " goto END

if /i "!USER_INPUT!"=="test" (
    echo [*] Esecuzione suite di test Python...
    python -m unittest discover -s tests
    echo.
    goto INTERACTIVE_LOOP
)

if /i "!USER_INPUT!"=="benchmark" (
    python -m makima_lab benchmark
    echo.
    goto INTERACTIVE_LOOP
)
if /i "!USER_INPUT!"=="ablation" (
    python -m makima_lab ablation
    echo.
    goto INTERACTIVE_LOOP
)
if /i "!USER_INPUT!"=="digest" (
    python -m makima_lab digest
    echo.
    goto INTERACTIVE_LOOP
)

set "PREFIX8=!USER_INPUT:~0,8!"
if /i "!PREFIX8!"=="explain " (
    python -m makima_lab explain "!USER_INPUT:~8!"
    echo.
    goto INTERACTIVE_LOOP
)
if /i "!USER_INPUT!"=="explain" (
    python -m makima_lab explain deploy
    echo.
    goto INTERACTIVE_LOOP
)

set "PREFIX5=!USER_INPUT:~0,5!"
if /i "!PREFIX5!"=="chat " (
    python -m makima_lab chat "!USER_INPUT:~5!"
    echo.
    goto INTERACTIVE_LOOP
)
if /i "!USER_INPUT!"=="chat" (
    python -m makima_lab chat
    echo.
    goto INTERACTIVE_LOOP
)

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

"%MAKIMA_BIN%" !USER_INPUT!
echo.
goto INTERACTIVE_LOOP

:CARGO_ERROR
echo [ERRORE] Cargo/Rust non trovato nel PATH.
echo Per favore installa Rust da https://rustup.rs/
pause
exit /b 1

:PYTHON_ERROR
echo [ERRORE] Python non trovato nel PATH.
echo Per favore installa Python 3.11+ da https://python.org/
pause
exit /b 1

:END
echo.
echo Chiusura sessione Makima. Arrivederci.
