import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
import numpy as np

from mlp import MLPPredictor
from lstm import LSTMPredictor

# --- CONFIGURAZIONE ---
BATCH_SIZE = 1024  # Batch grande per velocità massima (i dati sono già in RAM)
EPOCHS = 100
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_W, IMG_H = 1920, 1080

def get_pixel_error(output, target):
    """Calcola l'errore medio in pixel tra i centri predetto e reale"""
    # Denormalizzazione centri
    out_cx = (output[:, 0] + output[:, 2]/2) * IMG_W
    out_cy = (output[:, 1] + output[:, 3]/2) * IMG_H
    tgt_cx = (target[:, 0] + target[:, 2]/2) * IMG_W
    tgt_cy = (target[:, 1] + target[:, 3]/2) * IMG_H
    
    dist = torch.sqrt((out_cx - tgt_cx)**2 + (out_cy - tgt_cy)**2)
    return dist.mean().item()

def train_engine(name, model, train_loader, val_loader):
    print(f"\n>>> Allenamento {name} su {DEVICE}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    is_lstm = (name == "LSTM")
    best_pixel_err = float('inf')
    
    # Per il grafico finale o log
    history = {'train_loss': [], 'val_px_err': []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{EPOCHS}", leave=False)
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            if not is_lstm: 
                batch_x = batch_x.view(batch_x.size(0), -1)
                
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        # Validazione e calcolo errore in Pixel
        model.eval()
        val_px_err = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                if not is_lstm: vx = vx.view(vx.size(0), -1)
                vo = model(vx)
                val_px_err += get_pixel_error(vo, vy)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_px = val_px_err / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_px_err'].append(avg_val_px)

        # Salvataggio del modello migliore
        if avg_val_px < best_pixel_err:
            best_pixel_err = avg_val_px
            torch.save(model.state_dict(), Path(__file__).parent / f"best_{name.lower()}.pt")
            
        print(f"Epoch {epoch+1:03d} | Loss: {avg_train_loss:.6f} | Errore Val: {avg_val_px:.2f} px {'(MIGLIORE)' if avg_val_px == best_pixel_err else ''}")

    print(f"\nFine training {name}. Miglior errore: {best_pixel_err:.2f} px")

def main():
    base_path = Path(__file__).parent / "dataset"
    
    # 1. Caricamento istantaneo dei tensori
    print("Caricamento dati...")
    train_x = torch.load(base_path / "train" / "x.pth")
    train_y = torch.load(base_path / "train" / "y.pth")
    val_x = torch.load(base_path / "val" / "x.pth")
    val_y = torch.load(base_path / "val" / "y.pth")

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=BATCH_SIZE)

    # 2. Inizializzazione Modelli
    # WINDOW_SIZE = 10 (come impostato in build_dataset.py)
    mlp = MLPPredictor(window_size=10).to(DEVICE)
    lstm = LSTMPredictor(input_dim=4, hidden_dim=64).to(DEVICE)

    # 3. Training
    train_engine("MLP", mlp, train_loader, val_loader)
    train_engine("LSTM", lstm, train_loader, val_loader)

if __name__ == "__main__":
    main()