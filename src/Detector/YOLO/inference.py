import os
from ultralytics import YOLO
import torch
from config import *
import shutil
import sys
from src.utils.get_best_device import get_best_device

def run_detection_and_save_txt():
    # Carica il modello
    DEVICE = get_best_device()
    model_name = 'best.pt'  # Puoi cambiare con il percorso del tuo modello
    model = YOLO(MODEL_PATH / "training_runs" / "soccernet_high_precision" / "weights" / model_name) 
    model.to(DEVICE)
    print(f"Modello caricato su: {model.device}")

    base_path = DATASET_TR23_TEST
    output_folder = DETECTOR_OUTPUT_DIR / model_name.split(".")[0] # Cartella dove salveremo i txt per TrackEval
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # Se la cartella esiste già, controlliamo se è vuota
        if len(os.listdir(output_folder)) > 0:
            print(f"\nATTENZIONE: La cartella di output '{output_folder}' contiene già dei file.")
            conferma = input("Vuoi eliminare TUTTI i file presenti e continuare? (y/n): ").strip().lower()

            if conferma in ['s', 'si', 'y', 'yes']:
                print("Eliminazione dei file in corso...")
                for filename in os.listdir(output_folder):
                    file_path = os.path.join(output_folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path) # Elimina file o link simbolici
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path) # Elimina sottocartelle
                    except Exception as e:
                        print(f"Impossibile eliminare {file_path}. Motivo: {e}")
                print("Cartella svuotata con successo.\n")
            else:
                print("Operazione annullata. Esecuzione interrotta.")
                sys.exit() # Esce dallo script


    if not os.path.exists(base_path):
        print(f"Errore: Percorso {base_path} non trovato.")
        return

    sequences = os.listdir(base_path)

    for seq_name in sequences:
        seq_path = os.path.join(base_path, seq_name)
        img_source = os.path.join(seq_path, 'img1')

        if os.path.isdir(seq_path) and os.path.exists(img_source):
            print(f"--> Elaborazione sequenza: {seq_name}")
            
            # File di output per questa sequenza
            res_file_path = os.path.join(output_folder, f"{seq_name}.txt")
            
            # Eseguiamo l'inferenza usando lo stream per non saturare la RAM
            # stream=True restituisce un generatore
            results = model.predict(source=img_source, conf=0.35, imgsz=1280, verbose=False, stream=True, device=0, classes=[0])
            
            with open(res_file_path, 'w') as f:
                for result in results:
                    # Otteniamo il numero del frame dal percorso del file
                    # Es: '.../000001.jpg' -> frame_id = 1
                    path = result.path
                    frame_id = int(os.path.basename(path).split('.')[0])
                    
                    # I box sono in formato xywh (x_center, y_center, width, height) o xyxy
                    # Per MOT serve: Top-Left X, Top-Left Y, Width, Height
                    boxes = result.boxes.cpu().numpy()
                    
                    for box in boxes:
                        # Estrai coordinate xywh (top-left)
                        # box.xywh restituisce centro; box.xyxy restituisce angoli.
                        # box.xywhn normalizzato. 
                        # Ultralytics ha xywh come (center_x, center_y, w, h).
                        # Noi vogliamo (top_left_x, top_left_y, w, h).
                        
                        x1, y1, x2, y2 = box.xyxy[0]
                        w = x2 - x1
                        h = y2 - y1
                        conf = box.conf[0]
                        cls = box.cls[0]

                        # FILTRO: Assicuriamoci di salvare solo la classe "persona" (id 0 in COCO)
                        if int(cls) == 0: 
                            # Scriviamo nel formato MOT:
                            # frame, id, x, y, w, h, conf, -1, -1, -1
                            # Dato che è detection pura, ID mettiamo -1 o 1, non importa per DetA
                            line = f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,1,-1\n"
                            f.write(line)
            
            print(f"Salvati risultati in: {res_file_path}")

            with open(SEQMAP_FILE, 'a') as seqmap_file:
                seqmap_file.write(f"{seq_name}\n")

if __name__ == "__main__":
    run_detection_and_save_txt()