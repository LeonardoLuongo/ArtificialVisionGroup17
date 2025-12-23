# =================================================
# FILE: src/BlurFineTuning/train_blur_final.py
# =================================================
from ultralytics import YOLO
from pathlib import Path
import sys
from config import *
from ultralytics.utils import metrics

def custom_fitness(x):
    """
    x = [P, R, mAP@50, mAP@50-95]
    Di default YOLO usa: 0.1 * mAP@50 + 0.9 * mAP@50-95
    Noi usiamo: 1.0 * mAP@50 + 0.0 * mAP@50-95
    """
    return x[2] # Restituisce solo mAP@50

# Sovrascriviamo la funzione della libreria
metrics.fitness = custom_fitness

def train_blur_final():
    blur_yaml = ROOT_DIR / "dataset/blur_dataset/dataset.yaml"
    
    # Usa sempre il tuo modello migliore di partenza
    prev_model_path = MODEL_PATH / "training_runs/soccernet_high_precision/weights/best.pt"
    
    model = YOLO(str(prev_model_path))
    
    print(f"--- Avvio Training FINALE (Unfrozen + Augmentation) ---")
    
    results = model.train(
        data=str(blur_yaml),
        
        # --- STRATEGIA "NO MERCY" ---
        epochs=25,              # Diamo tempo al backbone di adattarsi
        imgsz=960,             # Fondamentale per vedere i giocatori piccoli
        batch=2,                
        
        # 1. SCONGELIAMO TUTTO
        # Il modello deve modificare il modo in cui "vede" i pixel, non solo come decide.
        freeze=0,               
        
        # 2. LEARNING RATE BASSO MA NON TROPPO
        # 1e-5 era troppo basso per muovere il backbone. 
        # Usiamo 5e-5 o 1e-4 per permettere adattamento.
        lr0=5e-5,               
        optimizer='AdamW',
        
        # 3. AUGMENTATION ESTREMA (La chiave per generalizzare)
        # Forza YOLO a vedere sfocato anche dove non c'è
        # blur=1.0,               # 1.0 = Probabilità massima di applicare blur artificiale
        # median_blur=0.5,        # Aggiunge median blur (rimuove dettagli fini)
        
        # Aiuta a riconoscere oggetti parziali
        erasing=0.4,            # Cancella pezzi di giocatore (simula occlusione)
        
        # Mosaic leggero per contesto
        mosaic=0.5,             
        
        # 4. BILANCIAMENTO LOSS (High Recall)
        # Vogliamo che trovi TUTTI, anche a costo di qualche falso positivo
        # (Il tracker filtrerà i falsi positivi, ma non può inventarsi i mancanti)
        box=7.5,
        cls=2.0,                # Aumentato da 1.5 -> Penalizza di più se manca un giocatore
        dfl=1.5,
        
        # Hardware
        device=0,
        workers=4,
        project=str(MODEL_PATH / "training_runs"),
        name='soccernet_blur_final',
        exist_ok=True,
        save=True,
        val=True
    )

if __name__ == "__main__":
    train_blur_final()