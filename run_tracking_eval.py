import os
import cv2
import glob
import sys
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Configurazione Percorsi
# Adatta questo percorso se la tua cartella dataset è in una posizione diversa
DATASET_ROOT = Path("dataset/tracking-2023/test")
SEQUENCE_NAME = "SNMOT-116" # Il nome della sequenza che vedo nello screenshot
OUTPUT_DIR = Path("inference_results")

# Percorsi specifici costruiti automaticamente
SEQ_DIR = DATASET_ROOT / SEQUENCE_NAME
IMG_DIR = SEQ_DIR / "img1"
GT_FILE = SEQ_DIR / "gt" / "gt.txt"
RESULT_FILE = OUTPUT_DIR / f"{SEQUENCE_NAME}.txt"

# Assicuriamoci che la cartella di output esista
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_inference():
    print(f"--- Avvio Inference Zero-Shot con YOLOv8n su {SEQUENCE_NAME} ---")
    
    # Caricamento Modello
    model = YOLO('yolov8n.pt')
    
    # Recupero lista immagini ordinate
    # È cruciale che siano ordinate numericamente (000001.jpg, 000002.jpg...)
    img_files = sorted(glob.glob(str(IMG_DIR / "*.jpg")))
    
    if not img_files:
        print(f"Errore: Nessuna immagine trovata in {IMG_DIR}")
        return

    print(f"Trovate {len(img_files)} immagini. Inizio tracking...")
    
    # Apriamo il file di output per scrivere i risultati
    with open(RESULT_FILE, 'w') as f_out:
        # Loop sulle immagini
        # Usiamo stream=True per efficienza con lunghe sequenze
        # persist=True è OBBLIGATORIO per il tracking tra frame
        # classes=[0] filtra solo le persone
        results = model.track(
            source=img_files, 
            persist=True, 
            classes=[0], 
            conf=0.3, 
            iou=0.5,
            verbose=False,
            device=0 # Usa la GPU (device 0)
        )

        for i, result in enumerate(results):
            frame_id = i + 1 # MOT format parte da 1
            
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for box, track_id, conf in zip(boxes, track_ids, confs):
                    # Conversione da (x1, y1, x2, y2) a (x, y, w, h)
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Formato standard MOT/SoccerNet:
                    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                    # Le ultime 3 sono coordinate 3D o world, mettiamo -1 o 1 come default
                    line = f"{frame_id},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1"
                    f_out.write(line + "\n")
            
            if frame_id % 50 == 0:
                print(f"Processati {frame_id}/{len(img_files)} frame")

    print(f"Inference completata. Risultati salvati in {RESULT_FILE}")

def run_evaluation():
    """
    Usa TrackEval per calcolare HOTA confrontando il file generato con il GT.
    TrackEval è molto rigido sui percorsi, quindi usiamo un wrapper.
    """
    print("\n--- Avvio Valutazione HOTA ---")
    
    if not GT_FILE.exists():
        print(f"ATTENZIONE: File GT non trovato in {GT_FILE}. Impossibile calcolare lo score.")
        return

    try:
        import trackeval
    except ImportError:
        print("Errore: TrackEval non installato. Esegui 'pip install git+https://github.com/JonathonLuiten/TrackEval.git'")
        return

    # Configurazione per TrackEval
    # Dobbiamo dire a TrackEval dove sono i nostri file custom
    eval_config = trackeval.Evaluator.get_default_eval_config()
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}

    # Override delle configurazioni per adattarsi alla tua struttura senza spostare file
    # Questo è un trucco per usare TrackEval su file singoli sparsi
    class CustomDataset(trackeval.datasets.MotChallenge2DBox):
        def __init__(self, config):
            super().__init__(config)
        
        def _load_raw_file(self, tracker, seq, is_gt):
            # Sovrascriviamo la funzione di caricamento file
            if is_gt:
                file_path = str(GT_FILE)
            else:
                file_path = str(RESULT_FILE)
            
            # Lettura e parsing manuale del file MOT
            # TrackEval si aspetta di leggere file da percorsi specifici, noi gli passiamo i dati grezzi
            try:
                data = np.loadtxt(file_path, delimiter=',')
            except:
                return {}, 0 # File vuoto o errore
                
            if len(data.shape) < 2:
                data = data.reshape(-1, 10) # Gestione caso 1 sola riga

            # Convertiamo in dizionario frame -> oggetti
            raw_data = {}
            num_timesteps = 0
            for row in data:
                frame = int(row[0])
                if frame > num_timesteps: num_timesteps = frame
                
                if frame not in raw_data:
                    raw_data[frame] = []
                
                # Format: [id] + [x,y,w,h] + [conf]
                # TrackEval vuole: id, x, y, w, h, conf, class_id, visibility
                # Il nostro file ha: frame, id, x, y, w, h, conf, -1, -1, -1
                p_id = int(row[1])
                x, y, w, h = row[2:6]
                conf = row[6]
                raw_data[frame].append([p_id, x, y, w, h, conf, 1, 1]) # 1=class_id, 1=visibility
            
            return raw_data, num_timesteps

        def get_seq_info(self):
            # Definiamo manualmente la sequenza
            return [SEQUENCE_NAME]

    # Esecuzione valutazione
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [CustomDataset(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    
    if len(dataset_list) > 0:
        return evaluator.evaluate(dataset_list, metrics_list)
    return None

if __name__ == "__main__":
    # 1. Eseguiamo il tracking
    run_inference()
    
    # 2. Calcoliamo lo score
    run_evaluation()