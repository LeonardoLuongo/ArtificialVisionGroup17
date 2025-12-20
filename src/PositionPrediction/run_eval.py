import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Setup path per importare config e modelli
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from config import DATASET_TR23_TEST
from src.PositionPrediction.mlp import MLPPredictor
from src.PositionPrediction.lstm import LSTMPredictor
from src.PositionPrediction.full_kalman import SoccerKalmanFilter

# --- CONFIGURAZIONE ---
WINDOW_SIZE = 10
PREDICTION_STEPS = 15  # Testiamo una "scomparsa" di 0.6 secondi (a 25fps)
IMG_W, IMG_H = 1920, 1080
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MC_ITERATIONS = 10     # Per calcolare l'incertezza via Dropout

def get_center(box):
    # Ritorna il centro [cx, cy] in pixel
    return np.array([(box[0] + box[2]/2) * IMG_W, (box[1] + box[3]/2) * IMG_H])

def run_test_set_evaluation():
    # 1. Caricamento Modelli
    mlp = MLPPredictor(window_size=WINDOW_SIZE).to(DEVICE)
    mlp.load_state_dict(torch.load("best_mlp_soccer.pt", map_location=DEVICE))
    
    lstm = LSTMPredictor(input_dim=4, hidden_dim=64).to(DEVICE)
    lstm.load_state_dict(torch.load("best_lstm_soccer.pt", map_location=DEVICE))

    # Dizionari per accumulare errori e incertezze
    metrics = {
        "Kalman": {"err": [], "unc": []},
        "MLP":    {"err": [], "unc": []},
        "LSTM":   {"err": [], "unc": []}
    }

    # 2. Scansione cartella TEST
    test_sequences = [d for d in os.listdir(DATASET_TR23_TEST) if os.path.isdir(os.path.join(DATASET_TR23_TEST, d))]
    
    for seq in tqdm(test_sequences, desc="Valutazione su Test Set"):
        gt_path = os.path.join(DATASET_TR23_TEST, seq, 'gt', 'gt.txt')
        if not os.path.exists(gt_path): continue
        
        data = np.loadtxt(gt_path, delimiter=',')
        unique_ids = np.unique(data[:, 1])

        for obj_id in unique_ids:
            # Estraggo traiettoria del giocatore
            p_data = data[data[:, 1] == obj_id]
            p_data = p_data[p_data[:, 0].argsort()]
            if len(p_data) < (WINDOW_SIZE + PREDICTION_STEPS): continue

            coords = p_data[:, 2:6].copy()
            coords[:, [0, 2]] /= IMG_W
            coords[:, [1, 3]] /= IMG_H

            # Finestra di input e GT futura
            history = coords[:WINDOW_SIZE]
            future_gt = coords[WINDOW_SIZE : WINDOW_SIZE + PREDICTION_STEPS]

            # --- A. KALMAN ---
            kf = SoccerKalmanFilter(coords[WINDOW_SIZE-1])
            for i in range(PREDICTION_STEPS):
                pred = kf.predict()
                metrics["Kalman"]["err"].append(np.linalg.norm(get_center(pred) - get_center(future_gt[i])))
                metrics["Kalman"]["unc"].append(kf.get_uncertainty())

            # --- B. MLP (con Monte Carlo Dropout) ---
            mlp.train() # Attivo Dropout per incertezza
            curr_hist = history.flatten().tolist()
            for i in range(PREDICTION_STEPS):
                preds = []
                for _ in range(MC_ITERATIONS):
                    with torch.no_grad():
                        p = mlp(torch.FloatTensor(curr_hist).view(1,-1).to(DEVICE)).cpu().numpy()[0]
                        preds.append(p)
                
                mean_p = np.mean(preds, axis=0)
                metrics["MLP"]["err"].append(np.linalg.norm(get_center(mean_p) - get_center(future_gt[i])))
                metrics["MLP"]["unc"].append(np.var(preds, axis=0).sum() * 1e5) # Scalato per leggibilità
                
                # Aggiorno storia per predizione ricorsiva
                curr_hist = curr_hist[4:] + mean_p.tolist()

            # --- C. LSTM (con Monte Carlo Dropout) ---
            lstm.train()
            curr_hist = history.tolist()
            for i in range(PREDICTION_STEPS):
                preds = []
                for _ in range(MC_ITERATIONS):
                    with torch.no_grad():
                        p = lstm(torch.FloatTensor(curr_hist).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
                        preds.append(p)
                
                mean_p = np.mean(preds, axis=0)
                metrics["LSTM"]["err"].append(np.linalg.norm(get_center(mean_p) - get_center(future_gt[i])))
                metrics["LSTM"]["unc"].append(np.var(preds, axis=0).sum() * 1e5)
                
                curr_hist = curr_hist[1:] + [mean_p.tolist()]

    # 3. Risultati Finali
    print("\n" + "═"*60)
    print(f"{'Modello':<12} | {'Errore Medio (ADE)':<18} | {'Incertezza Media':<15}")
    print("-" * 60)
    for m in metrics:
        avg_e = np.mean(metrics[m]["err"])
        avg_u = np.mean(metrics[m]["unc"])
        print(f"{m:<12} | {avg_e:>14.2f} px | {avg_u:>16.4f}")
    print("═"*60)

if __name__ == "__main__":
    run_test_set_evaluation()