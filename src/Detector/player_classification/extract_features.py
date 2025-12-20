import os
import sys
import cv2
import numpy as np
import configparser
from pathlib import Path
from tqdm import tqdm

# Setup path per importare config (saliamo di 2 livelli da src/TeamAssignment)
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

# Importiamo i path precisi dal tuo config.py
from config import DATASET_TR23_TRAIN, DATASET_TR23_TEST, SPLIT_DIR, MODEL_PATH

# Parametri di campionamento
KNN_SUBSAMPLE_RATE = 10 

class SoccerAdvancedExtractor:
    def __init__(self):
        self.label_map = {
            'player team left': 0, 'player team right': 1,
            'referee': 2, 'goalkeeper team left': 3,
            'goalkeepers team left': 3, 'goalkeeper team right': 4,
            'goalkeepers team right': 4
        }

    def get_label(self, ini_path, tracklet_id):
        config = configparser.ConfigParser()
        config.read(ini_path)
        key = f"trackletID_{tracklet_id}"
        if 'Sequence' in config and key in config['Sequence']:
            val = config['Sequence'][key].lower().split(';')[0].strip()
            for k, v in self.label_map.items():
                if k in val: return v
        return -1

    def extract_spatial_hsv(self, image, bbox):
        x, y, w, h = map(int, bbox)
        crop = image[max(0, y):y+h, max(0, x):x+w]
        if crop.size == 0 or h < 10: return None
        
        mid = h // 2
        top_part = crop[0:mid, :]
        bottom_part = crop[mid:, :]
        
        def get_hist(img_part):
            if img_part.size == 0: return np.zeros(64)
            hsv = cv2.cvtColor(img_part, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()

        return np.concatenate([get_hist(top_part), get_hist(bottom_part)])

def run_extraction(split_name):
    print(f"\n--- Avvio Estrazione: {split_name.upper()} ---")
    
    manifest_path = SPLIT_DIR / f"{split_name}.txt"
    if not manifest_path.exists():
        print(f"[ERRORE] Manifest {manifest_path} non trovato!")
        return

    # Usiamo le tue costanti del config
    base_source = DATASET_TR23_TRAIN if split_name == 'train' else DATASET_TR23_TEST

    with open(manifest_path, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]

    processor = SoccerAdvancedExtractor()
    features, labels = [], []

    for seq_name in tqdm(sequences, desc=f"Split {split_name}"):
        seq_path = base_source / seq_name
        gt_path = seq_path / "gt" / "gt.txt"
        img_dir = seq_path / "img1"
        ini_path = seq_path / "gameinfo.ini"

        if not all([gt_path.exists(), ini_path.exists(), img_dir.exists()]):
            continue

        gt_data = np.loadtxt(gt_path, delimiter=',')
        unique_frames = np.unique(gt_data[:, 0])
        
        for f_idx in unique_frames:
            if (int(f_idx) - 1) % KNN_SUBSAMPLE_RATE != 0: continue

            img = cv2.imread(str(img_dir / f"{int(f_idx):06d}.jpg"))
            if img is None: continue

            rows = gt_data[gt_data[:, 0] == f_idx]
            for r in rows:
                lbl = processor.get_label(ini_path, int(r[1]))
                if lbl != -1:
                    feat = processor.extract_spatial_hsv(img, r[2:6])
                    if feat is not None:
                        features.append(feat)
                        labels.append(lbl)

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    X_out = MODEL_PATH / f"knn_X_{split_name}.npy"
    y_out = MODEL_PATH / f"knn_y_{split_name}.npy"
    np.save(X_out, np.array(features))
    np.save(y_out, np.array(labels))
    print(f"Salvati {len(features)} campioni in {X_out}")

if __name__ == "__main__":
    run_extraction('train')
    run_extraction('test')