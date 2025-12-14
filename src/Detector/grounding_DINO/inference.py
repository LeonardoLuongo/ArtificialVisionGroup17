import os
import shutil
import sys
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Importa le tue config
from config import * 

def run_grounding_dino_detection():
    # --- CONFIGURAZIONE MODELLO ---
    # Usiamo "grounding-dino-base" (più preciso) o "grounding-dino-tiny" (più veloce)
    model_id = "IDEA-Research/grounding-dino-base"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Caricamento Grounding DINO ({model_id}) su: {device}...")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # IL PROMPT: Cosa cercare. 
    # È importante mettere il punto "." alla fine per Grounding DINO.
    text_prompt = "person." 
    
    # Soglie
    BOX_THRESHOLD = 0.30  # Confidenza minima per considerare un box
    TEXT_THRESHOLD = 0.25 # Confidenza minima per associare la parola "person" al box

    base_path = DATASET_TR23_TEST
    output_folder = DETECTOR_OUTPUT_DIR / "grounding_dino"
    
    # --- GESTIONE CARTELLE (Identica al tuo script) ---
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        if len(os.listdir(output_folder)) > 0:
            print(f"\nATTENZIONE: La cartella '{output_folder}' non è vuota.")
            conferma = input("Vuoi eliminare TUTTI i file e continuare? (y/n): ").strip().lower()
            if conferma in ['s', 'si', 'y', 'yes']:
                for filename in os.listdir(output_folder):
                    file_path = os.path.join(output_folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Errore eliminazione {file_path}: {e}")
            else:
                sys.exit()

    if not os.path.exists(base_path):
        print(f"Errore: Percorso {base_path} non trovato.")
        return

    sequences = os.listdir(base_path)

    # --- CICLO SEQUENZE ---
    for seq_name in sequences:
        seq_path = os.path.join(base_path, seq_name)
        img_source = os.path.join(seq_path, 'img1')

        if os.path.isdir(seq_path) and os.path.exists(img_source):
            print(f"--> Elaborazione sequenza: {seq_name}")
            res_file_path = os.path.join(output_folder, f"{seq_name}.txt")
            
            # Ordina le immagini per sicurezza
            image_files = sorted([f for f in os.listdir(img_source) if f.endswith(('.jpg', '.png', '.jpeg'))])
            
            with open(res_file_path, 'w') as f:
                for img_name in image_files:
                    img_path = os.path.join(img_source, img_name)
                    frame_id = int(img_name.split('.')[0])
                    
                    # 1. Carica immagine con PIL (richiesto da Transformers)
                    try:
                        image = Image.open(img_path).convert("RGB")
                    except Exception as e:
                        print(f"Errore lettura {img_name}: {e}")
                        continue

                    # 2. Pre-process e Inferenza
                    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)

                    # 3. Post-process (converte output grezzi in coordinate pixel)
                    # target_sizes serve per scalare i box normalizzati alle dimensioni reali dell'immagine
                    width, height = image.size
                    results = processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        box_threshold=BOX_THRESHOLD,
                        text_threshold=TEXT_THRESHOLD,
                        target_sizes=[(height, width)]
                    )[0]

                    # 4. Scrittura risultati
                    # results['boxes'] è in formato xyxy (x_min, y_min, x_max, y_max)
                    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                        
                        # Il label restituito è la stringa del prompt (es. "person")
                        # Se il prompt è semplice "person.", controlliamo solo che la box sia valida
                        
                        # Conversione da xyxy a xywh (Top-Left X, Top-Left Y, Width, Height)
                        x1, y1, x2, y2 = box.cpu().numpy()
                        
                        w = x2 - x1
                        h = y2 - y1
                        # x1 e y1 sono già le coordinate Top-Left
                        
                        conf = score.item()
                        
                        # Filtro aggiuntivo: a volte DINO rileva label parziali se il prompt è complesso,
                        # ma qui con "person." il filtro è implicito nel modello.
                        if label == "person":
                            line = f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
                            f.write(line)

            print(f"Salvati risultati in: {res_file_path}")

            with open(SEQMAP_FILE, 'a') as seqmap_file:
                seqmap_file.write(f"{seq_name}\n")

if __name__ == "__main__":
    run_grounding_dino_detection()