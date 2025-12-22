import os
import json
import random
from pathlib import Path
from config import *

def is_overlapping(rect1, rect2):
    """
    Controlla se due rettangoli si sovrappongono.
    Formato rect: {'x': val, 'y': val, 'width': val, 'height': val}
    """
    # Definiamo i bordi del rettangolo 1
    l1, r1 = rect1['x'], rect1['x'] + rect1['width']
    t1, b1 = rect1['y'], rect1['y'] + rect1['height']
    
    # Definiamo i bordi del rettangolo 2
    l2, r2 = rect2['x'], rect2['x'] + rect2['width']
    t2, b2 = rect2['y'], rect2['y'] + rect2['height']

    # Se uno è a sinistra dell'altro
    if l1 >= r2 or l2 >= r1:
        return False
    # Se uno è sopra l'altro
    if t1 >= b2 or t2 >= b1:
        return False

    return True # Si sovrappongono

def generate_random_roi_config_no_overlap():
    """Genera 2 ROI valide e NON sovrapposte."""
    config = {}
    
    # --- Genera ROI 1 ---
    x1 = round(random.uniform(0.0, 0.5), 2)
    w1 = round(random.uniform(0.1, 1.0 - x1), 2)
    y1 = round(random.uniform(0.0, 0.5), 2)
    h1 = round(random.uniform(0.1, 1.0 - y1), 2)
    
    rect1 = {"x": x1, "y": y1, "width": w1, "height": h1}
    config["roi1"] = rect1

    # --- Genera ROI 2 (con controllo overlap) ---
    max_attempts = 100
    for _ in range(max_attempts):
        x2 = round(random.uniform(0.0, 0.5), 2)
        w2 = round(random.uniform(0.1, 1.0 - x2), 2)
        y2 = round(random.uniform(0.0, 0.5), 2)
        h2 = round(random.uniform(0.1, 1.0 - y2), 2)
        
        rect2 = {"x": x2, "y": y2, "width": w2, "height": h2}
        
        # Se NON si sovrappongono, accettiamo e usciamo
        if not is_overlapping(rect1, rect2):
            config["roi2"] = rect2
            return config
    
    # Fallback raro: se non riesce dopo 100 tentativi, 
    # forza una ROI2 molto piccola nell'angolo opposto per evitare loop infiniti
    if x1 < 0.5:
        fallback_x = 0.8
    else:
        fallback_x = 0.1
        
    config["roi2"] = {"x": fallback_x, "y": 0.1, "width": 0.1, "height": 0.1}
    return config

def create_rois_in_dataset(dataset_root):
    data_dir = Path(dataset_root)
    
    if not data_dir.exists():
        print(f"Errore: Il percorso {dataset_root} non esiste.")
        return

    # Trova le cartelle numeriche
    video_folders = [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    if not video_folders:
        print("Nessuna cartella numerica trovata.")
        return

    for folder in video_folders:
        roi_path = folder / "roi.json"
        
        # Usa la nuova funzione senza overlap
        roi_data = generate_random_roi_config_no_overlap()
        
        with open(roi_path, 'w') as f:
            json.dump(roi_data, f, indent=4)
        
        print(f"Generato NO-OVERLAP {roi_path.name} in {folder.name}")

if __name__ == "__main__":
    PATH_DATASET = DATASET_TR23_TEST
    create_rois_in_dataset(PATH_DATASET)