#!/bin/bash

# Imposta il percorso (adatta /h/ se il tuo mount point è diverso)
TARGET_DIR="/h/My Drive/Uni/Magistrale/colab/2/AV/Project"
REPO_URL="https://github.com/LeonardoLuongo/ArtificialVisionGroup17"

echo "Target Directory: $TARGET_DIR"

# Crea la directory se non esiste
mkdir -p "$TARGET_DIR"

# Entra nella directory
cd "$TARGET_DIR" || exit

# Controlla se è già un repo git
if [ -d ".git" ]; then
    echo "Repo esistente. Eseguo git pull..."
    git pull
else
    echo "Repo non trovata. Eseguo git clone..."
    git clone "$REPO_URL" .
fi

echo "Fatto."