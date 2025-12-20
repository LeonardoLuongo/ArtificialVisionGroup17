import os
import sys
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

# Setup path per importare config
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from config import DATASET_TR23_TRAIN
from mlp import MLPPredictor
from lstm import LSTMPredictor

# --- CONFIGURAZIONE ---
WINDOW_SIZE = 10    # Frame passati usati per predire
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
VAL_RATIO = 0.2     # 20% delle sequenze usate per validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Risoluzione standard SoccerNet per normalizzazione
IMG_W, IMG_H = 1920, 1080

class SoccerTrajectoryDataset(Dataset):
    def __init__(self, sequences, window_size=10):
        self.x = []
        self.y = []
        
        for seq_path in tqdm(sequences, desc="Caricamento sequenze"):
            gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
            if not os.path.exists(gt_file): continue
            
            # Carica GT: frame, id, x, y, w, h, conf, class, vis
            data = np.loadtxt(gt_file, delimiter=',')
            
            # Raggruppa per ID giocatore
            unique_ids = np.unique(data[:, 1])
            for obj_id in unique_ids:
                # Prendi solo i record di questo giocatore ordinati per frame
                player_data = data[data[:, 1] == obj_id]
                player_data = player_data[player_data[:, 0].argsort()]
                
                # Estrai solo x, y, w, h e normalizza
                coords = player_data[:, 2:6]
                coords[:, [0, 2]] /= IMG_W
                coords[:, [1, 3]] /= IMG_H
                
                # Crea finestre mobili (Sliding Window)
                if len(coords) > window_size:
                    for i in range(len(coords) - window_size):
                        self.x.append(coords[i : i + window_size]) # Input: 10 frame
                        self.y.append(coords[i + window_size])     # Target: frame 11
        
        self.x = torch.FloatTensor(np.array(self.x))
        self.y = torch.FloatTensor(np.array(self.y))

    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

def train_one_epoch(model, loader, optimizer, criterion, is_lstm=False):
    model.train()
    total_loss = 0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        
        # Se MLP, dobbiamo appiattire l'input (batch, window*4)
        if not is_lstm:
            batch_x = batch_x.view(batch_x.size(0), -1)
            
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, is_lstm=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            if not is_lstm: batch_x = batch_x.view(batch_x.size(0), -1)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            total_loss += loss.item()
    return total_loss / len(loader)

def run_training():
    # 1. Preparazione split sequenze
    all_seq_dirs = [os.path.join(DATASET_TR23_TRAIN, d) for d in os.listdir(DATASET_TR23_TRAIN) 
                    if os.path.isdir(os.path.join(DATASET_TR23_TRAIN, d))]
    np.random.shuffle(all_seq_dirs)
    
    val_split = int(len(all_seq_dirs) * VAL_RATIO)
    train_dirs = all_seq_dirs[val_split:]
    val_dirs = all_seq_dirs[:val_split]
    
    print(f"Train su {len(train_dirs)} video, Val su {len(val_dirs)} video")

    # 2. Creazione Dataset e Loader
    train_ds = SoccerTrajectoryDataset(train_dirs, WINDOW_SIZE)
    val_ds = SoccerTrajectoryDataset(val_dirs, WINDOW_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 3. Inizializzazione Modelli
    models = {
        "MLP": MLPPredictor(window_size=WINDOW_SIZE).to(DEVICE),
        "LSTM": LSTMPredictor(input_dim=4, hidden_dim=64).to(DEVICE)
    }
    
    criterion = nn.MSELoss()

    # 4. Loop di addestramento
    for name, model in models.items():
        print(f"\n--- Inizio Training modello: {name} ---")
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        is_lstm = (name == "LSTM")
        best_val_loss = float('inf')
        
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, is_lstm)
            val_loss = validate(model, val_loader, criterion, is_lstm)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"best_{name.lower()}_soccer.pt")
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

if __name__ == "__main__":
    run_training()