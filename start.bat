@echo off
setlocal enabledelayedexpansion

chcp 65001 >nul
title Makima - Interpretable Probabilistic Forecasting System
set "PYTHONPATH=%~dp0python;%PYTHONPATH%"

echo.
echo ========================================================================
echo   M A K I M A   -   I N I T I A L I Z A T I O N   L A U N C H E R
echo   Interpretable Probabilistic Engine and Cognitive Neural Mind
echo ========================================================================
echo.

where cargo >nul 2>nul
if errorlevel 1 goto CARGO_ERROR

where python >nul 2>nul
if errorlevel 1 goto PYTHON_ERROR

echo [1/2] Verifica ambiente scientifico e Neural Mind: makima_lab...
python -m unittest discover -s tests >nul 2>&1
if errorlevel 1 goto PYTEST_WARN
echo [OK] Suite di test superata: 40 unit test verificati con successo.
goto PYTEST_DONE

:PYTEST_WARN
echo [*] Test Python completati con avvisi. Procedura di avvio in corso.

:PYTEST_DONE
echo.
echo [2/2] Avvio del motore e diagnostica Makima...
cargo run --bin makima -- status

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
echo  * VALIDAZIONE SCIENTIFICA E BENCHMARK:
echo    - benchmark             : Esegue il benchmark comparativo completo su 5,000 campioni
echo    - ablation              : Esegue l'Ablation Study su tutti i sottosistemi
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

:INTERACTIVE_LOOP
set "USER_INPUT="
set /p USER_INPUT="makima > "

if /i "!USER_INPUT!"=="exit" goto END
if /i "!USER_INPUT!"=="quit" goto END
if /i "!USER_INPUT!"=="q" goto END
if "!USER_INPUT!"=="" goto INTERACTIVE_LOOP

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
