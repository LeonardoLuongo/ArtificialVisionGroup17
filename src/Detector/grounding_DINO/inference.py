import os
import sys
import shutil
import torch
import warnings
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from pathlib import Path

# --- SETUP PATH ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

try:
    from config import *
except ImportError:
    print("ERRORE CRITICO: Non riesco a trovare 'config.py'.")
    sys.exit(1)

def run_grounding_dino_detection():
    # Setup del modello
    model_id = "IDEA-Research/grounding-dino-base"
    
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Uso accelerazione Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("Uso accelerazione CUDA")
    else:
        device = 'cpu'
        print("Uso CPU (Lento)")

    print(f"Caricamento Grounding DINO ({model_id})...")
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    TEXT_PROMPT = "person."  
    BOX_THRESHOLD = 0.35   
    TEXT_THRESHOLD = 0.25  

    base_path = os.path.join(DATASET_TR23_TEST, "test")
    output_folder = Path(DETECTOR_OUTPUT_DIR) / "grounding_dino_base"
    
    print(f"\n--- DEBUG PATH ---")
    print(f"Path Dataset: {base_path}")
    print(f"Path Output:  {output_folder}")
    
    if not os.path.exists(base_path):
        print(f"ERRORE: La cartella del dataset NON esiste: {base_path}")
        return

    sequences = os.listdir(base_path)
    print(f"Ho trovato {len(sequences)} elementi nella cartella del dataset.")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files_created = 0

    for seq_name in sequences:
        seq_path = os.path.join(base_path, seq_name)
        img_source = os.path.join(seq_path, 'img1')

        if not os.path.isdir(seq_path): continue
        if not os.path.exists(img_source):
            print(f"  -> SALTATA {seq_name}: Manca cartella img1")
            continue
        
        # Filtra solo immagini vere
        image_files = sorted([f for f in os.listdir(img_source) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        num_imgs = len(image_files)
        
        if num_imgs == 0:
            print(f"  -> SALTATA {seq_name}: Cartella vuota")
            continue

        print(f"\nAnalisi sequenza: {seq_name} ({num_imgs} immagini)")
        
        res_file_path = os.path.join(output_folder, f"{seq_name}.txt")
        
        with open(res_file_path, 'w') as f:
            lines_written = 0
            
            for i, img_name in enumerate(image_files):
                if i % 10 == 0: 
                    print(f"     Elaborazione frame {i}/{num_imgs}...", end='\r')

                img_path = os.path.join(img_source, img_name)
                
                try:
                    frame_id = int(img_name.split('.')[0])
                except ValueError:
                    continue 

                try:
                    image_pil = Image.open(img_path).convert("RGB")
                    
                    inputs = processor(images=image_pil, text=TEXT_PROMPT, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)

                    width, height = image_pil.size
                    
                    results = processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        target_sizes=[(height, width)]
                    )[0]

                    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                        
                        if score.item() < BOX_THRESHOLD:
                            continue
                        
                        if "person" in label:
                            x1, y1, x2, y2 = box.cpu().numpy()
                            w = x2 - x1
                            h = y2 - y1
                            conf = score.item()

                            # Scrittura formato MOT
                            # frame_id, id_obj(-1), x1, y1, w, h, conf, -1, -1, -1
                            line = f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,1,-1\n"
                            f.write(line)
                            lines_written += 1
                            
                except Exception as e:
                    print(f"\nErrore nel frame {img_name}: {e}")
                    continue

            print(f"     COMPLETATO! Scritte {lines_written} righe.        ")
            files_created += 1

    print(f"\n--- FINE ---")
    print(f"File salvati in: {output_folder}")

if __name__ == "__main__":
    run_grounding_dino_detection()