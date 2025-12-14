import os
from ultralytics import YOLO
import torch
from config import *

def run_detection_and_save_txt():
    # Carica il modello
    model_name = 'yolo12m.pt'  # Puoi cambiare con il percorso del tuo modello
    model = YOLO(model_name) 
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Modello caricato su: {model.device}")

    base_path = DATASET_TR23_TEST
    output_folder = DETECTOR_OUTPUT_DIR / model_name.split(".")[0] # Cartella dove salveremo i txt per TrackEval
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
            results = model.predict(source=img_source, conf=0.25, verbose=False, stream=True, device=0, classes=[0])
            
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