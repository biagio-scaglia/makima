@echo off
setlocal EnableDelayedExpansion
title Makima - Desktop GUI (Tauri)

echo ===============================================================================
echo                MAKIMA -- Bayesian Intelligence Desktop Interface
echo ===============================================================================
echo.

if not exist "%~dp0crates\makima-gui\node_modules" (
    echo [INFO] Installazione iniziale dipendenze UI in corso...
    call npm --prefix "%~dp0crates\makima-gui" install
    if errorlevel 1 (
        echo [ERRORE] Impossibile installare le dipendenze npm.
        pause
        exit /b 1
    )
)

echo [INFO] Avvio interfaccia Desktop Makima (Tauri v2)...
echo.

cd /d "%~dp0crates\makima-gui"
call npx tauri dev

if errorlevel 1 (
    echo.
    echo [AVVISO] Chiusura o errore nell'applicazione Desktop.
)
pause
