
@echo off
echo [INFO] Avvio procedura combinata Build & Run...

:: Chiama lo script di build
call .\conda\build\build.bat

:: Controlla se c'è stato un errore nella build
if %errorlevel% neq 0 (
    echo [ERROR] La creazione dell'environment ha fallito. Impossibile eseguire il run.
    pause
    exit /b %errorlevel%
)

echo [INFO] Build completata con successo. Avvio applicazione...
echo ---------------------------------------------------------

:: Chiama lo script di run
call conda\run\run.bat