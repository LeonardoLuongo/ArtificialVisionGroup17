
#!/bin/bash

echo "[INFO] Avvio procedura combinata Build & Run..."

# Assicura che gli script interni abbiano i permessi di esecuzione
chmod +x conda/build/build.sh
chmod +x conda/run/run.sh

# Esegue la build. 
# L'operatore '&&' assicura che il 'run' parta solo se la 'build' finisce senza errori.
./conda/build/build.sh && \
echo "[INFO] Build completata. Avvio applicazione..." && \
echo "---------------------------------------------------------" && \
./conda/run/run.sh