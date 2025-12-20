import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Setup path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from config import DATASET_TR23_TEST
from src.PositionPrediction.mlp import MLPPredictor
from src.PositionPrediction.lstm import LSTMPredictor
from src.PositionPrediction.full_kalman import SoccerKalmanFilter

# --- CONFIGURAZIONE ---
WINDOW_SIZE = 10
PREDICTION_STEPS = 15
IMG_W, IMG_H = 1920, 1080
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PESO IBRIDO: 0.6 significa 60% Kalman e 40% LSTM
# È un ottimo punto di partenza visto che Kalman è più stabile
ALPHA = 0.8 

# Cartella per i grafici
PLOT_DIR = Path(__file__).parent / "eval_plots"
PLOT_DIR.mkdir(exist_ok=True)
MAX_PLOTS = 20 

def plot_and_save(history_px, gt_future_px, kalman_px, mlp_px, lstm_px, hybrid_px, filename):
    plt.figure(figsize=(12, 8))
    
    # Plot Storia (Osservata)
    plt.plot(history_px[:,0], history_px[:,1], 'k.-', linewidth=2, label='Storia (Osservata)')
    plt.plot(history_px[-1,0], history_px[-1,1], 'ko', markersize=10)

    # Plot Ground Truth (Reale)
    plt.plot(gt_future_px[:,0], gt_future_px[:,1], 'g--', linewidth=2, label='Ground Truth (Reale)', alpha=0.5)
    plt.plot(gt_future_px[-1,0], gt_future_px[-1,1], 'g*', markersize=15)

    # Plot Modelli Individuali
    plt.plot(kalman_px[:,0], kalman_px[:,1], 'b.-', alpha=0.4, label='Kalman Filter')
    plt.plot(mlp_px[:,0], mlp_px[:,1], 'r.-', alpha=0.3, label='MLP')
    plt.plot(lstm_px[:,0], lstm_px[:,1], 'm.-', alpha=0.4, label='LSTM')

    # Plot HYBRID (Il protagonista)
    plt.plot(hybrid_px[:,0], hybrid_px[:,1], 'c.-', linewidth=3, label=f'Hybrid (α={ALPHA})')

    plt.title(f"Confronto Modelli: {filename.stem}", fontsize=12)
    plt.legend(loc='best', ncol=2)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Zoom intelligente
    all_pts = np.concatenate([history_px, gt_future_px, hybrid_px])
    margin = 200
    plt.xlim(min(all_pts[:,0]) - margin, max(all_pts[:,0]) + margin)
    plt.ylim(max(all_pts[:,1]) + margin, min(all_pts[:,1]) - margin)

    plt.savefig(filename)
    plt.close()

def get_center_px(box_norm):
    cx = (box_norm[0] + box_norm[2]/2) * IMG_W
    cy = (box_norm[1] + box_norm[3]/2) * IMG_H
    return np.array([cx, cy])

def run_final_comparison():
    print(f"Avvio Valutazione Finale con Modello Hybrid (Alpha: {ALPHA})")
    
    # Caricamento Modelli
    mlp = MLPPredictor(window_size=WINDOW_SIZE).to(DEVICE)
    mlp.load_state_dict(torch.load(Path(__file__).parent / "best_mlp.pt", map_location=DEVICE))
    mlp.eval()

    lstm = LSTMPredictor(input_dim=4, hidden_dim=64).to(DEVICE)
    lstm.load_state_dict(torch.load(Path(__file__).parent / "best_lstm.pt", map_location=DEVICE))
    lstm.eval()

    stats = {
        "Kalman": {"ade": [], "fde": []}, 
        "MLP":    {"ade": [], "fde": []}, 
        "LSTM":   {"ade": [], "fde": []},
        "Hybrid": {"ade": [], "fde": []}
    }

    test_sequences = sorted([d for d in os.listdir(DATASET_TR23_TEST) if os.path.isdir(os.path.join(DATASET_TR23_TEST, d))])
    plot_count = 0 

    for seq in tqdm(test_sequences, desc="Valutazione"):
        gt_path = os.path.join(DATASET_TR23_TEST, seq, 'gt', 'gt.txt')
        if not os.path.exists(gt_path): continue
        data = np.loadtxt(gt_path, delimiter=',')
        unique_ids = np.unique(data[:, 1])

        for tid in unique_ids:
            p_traj = data[data[:, 1] == tid]
            p_traj = p_traj[p_traj[:, 0].argsort()]
            if len(p_traj) < (WINDOW_SIZE + PREDICTION_STEPS): continue

            coords = p_traj[:, 2:6].copy()
            coords[:, [0, 2]] /= IMG_W
            coords[:, [1, 3]] /= IMG_H
            history = coords[:WINDOW_SIZE]
            future_gt = coords[WINDOW_SIZE : WINDOW_SIZE + PREDICTION_STEPS]

            # 1. PREDIZIONE KALMAN (con Warm-up)
            kf = SoccerKalmanFilter(coords[0])
            for i in range(1, WINDOW_SIZE):
                kf.predict(); kf.update(coords[i])
            
            k_preds_px = []
            for i in range(PREDICTION_STEPS):
                p = kf.predict()
                k_preds_px.append(get_center_px(p))

            # 2. PREDIZIONE MLP
            m_preds_px = []
            curr_h_mlp = history.flatten().tolist()
            for i in range(PREDICTION_STEPS):
                with torch.no_grad():
                    p = mlp(torch.FloatTensor(curr_h_mlp).view(1, -1).to(DEVICE)).cpu().numpy()[0]
                m_preds_px.append(get_center_px(p))
                curr_h_mlp = curr_h_mlp[4:] + p.tolist()

            # 3. PREDIZIONE LSTM
            l_preds_px = []
            curr_h_lstm = history.tolist()
            for i in range(PREDICTION_STEPS):
                with torch.no_grad():
                    p = lstm(torch.FloatTensor(curr_h_lstm).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
                l_preds_px.append(get_center_px(p))
                curr_h_lstm = curr_h_lstm[1:] + [p.tolist()]

            # 4. PREDIZIONE HYBRID (Fusione)
            h_preds_px = []
            for i in range(PREDICTION_STEPS):
                # Fusione pesata tra Kalman e LSTM
                hybrid_pt = (ALPHA * k_preds_px[i]) + ((1 - ALPHA) * l_preds_px[i])
                h_preds_px.append(hybrid_pt)

            # --- CALCOLO METRICHE ---
            gt_centers_px = [get_center_px(f) for f in future_gt]
            
            for name, preds in zip(["Kalman", "MLP", "LSTM", "Hybrid"], 
                                   [k_preds_px, m_preds_px, l_preds_px, h_preds_px]):
                errs = [np.linalg.norm(preds[i] - gt_centers_px[i]) for i in range(PREDICTION_STEPS)]
                stats[name]["ade"].append(np.mean(errs))
                stats[name]["fde"].append(errs[-1])

            # Salva Grafico
            if plot_count < MAX_PLOTS:
                fname = PLOT_DIR / f"hybrid_plot_{seq}_id{int(tid)}.png"
                plot_and_save(
                    np.array([get_center_px(h) for h in history]),
                    np.array(gt_centers_px),
                    np.array(k_preds_px), np.array(m_preds_px), 
                    np.array(l_preds_px), np.array(h_preds_px),
                    fname
                )
                plot_count += 1

    # --- REPORT FINALE ---
    print("\n" + "═"*75)
    print(f"{'MODELLO':<12} | {'ADE (Errore Medio)':<20} | {'FDE (Ultimo Frame)':<18}")
    print("-" * 75)
    for m in ["Kalman", "MLP", "LSTM", "Hybrid"]:
        ade = np.mean(stats[m]["ade"])
        fde = np.mean(stats[m]["fde"])
        # Evidenziamo l'Hybrid per trovarlo subito
        prefix = ">> " if m == "Hybrid" else "   "
        print(f"{prefix}{m:<9} | {ade:>16.2f} px | {fde:>14.2f} px")
    print("═"*75)
    print(f"Alpha utilizzato: {ALPHA} (Kalman) / {1-ALPHA:.1f} (LSTM)")

if __name__ == "__main__":
    run_final_comparison()