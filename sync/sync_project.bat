@echo off
setlocal

:: --- CONFIGURAZIONE ---
set "TARGET_DIR=H:\My Drive\Uni\Magistrale\colab\2\AV\Project"
set "REPO_URL=https://github.com/LeonardoLuongo/ArtificialVisionGroup17"
set "BRANCH=main"
:: ----------------------

echo Controllo della directory: %TARGET_DIR%

:: Crea la cartella se non esiste
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"
cd /d "%TARGET_DIR%"

:: Rimuove eventuali lock di git precedenti se qualcosa si è bloccato
if exist ".git\index.lock" del ".git\index.lock"

:: Controllo se è già un repo git valido
if exist ".git" (
    echo [OK] Repo rilevata.
    echo [ACTION] Aggiorno i file locali con GIT PULL...
    git pull origin %BRANCH%
) else (
    echo [INFO] Inizializzo nuovo repo nella cartella esistente...
    
    git init
    git remote add origin %REPO_URL%
    git fetch origin
    
    echo [ACTION] Scarico i file da GitHub (sovrascrivo solo i file di codice)...
    git reset --hard origin/%BRANCH%
    git branch --set-upstream-to=origin/%BRANCH% %BRANCH%
)

echo.
if %ERRORLEVEL% NEQ 0 (
    echo [ERRORE] Qualcosa e' andato storto. Controlla la connessione internet o le credenziali.
) else (
    echo [SUCCESS] Operazione completata.
)
pause