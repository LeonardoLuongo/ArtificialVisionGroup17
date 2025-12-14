# =================================================
# FILE: src/Dataset/create_smart_splits.py
# =================================================
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from sklearn.cluster import KMeans
from collections import defaultdict
import shutil

# Setup path
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(ROOT_DIR))
from config import DATASET_TR23_TRAIN, SPLIT_DIR

# Parametri
VAL_RATIO = 0.2
N_CLUSTERS = 5  # Quanti "tipi" di video diversi ci aspettiamo (es. stadi diversi)
SEED = 42

def extract_features(dataset_path):
    """
    Usa una ResNet18 pre-addestrata per estrarre un vettore di caratteristiche
    dalla prima immagine di ogni sequenza.
    """
    print("Caricamento modello per estrazione feature (ResNet18)...")
    # Usiamo ResNet18: leggera e veloce, sufficiente per distinguere stadi/luci
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Rimuoviamo l'ultimo layer (classificazione) per avere solo le feature
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    sequences = []
    
    seq_dirs = sorted([s for s in os.listdir(dataset_path) if (dataset_path / s).is_dir()])
    
    print(f"Estrazione feature da {len(seq_dirs)} sequenze...")
    
    with torch.no_grad():
        for seq in seq_dirs:
            img_dir = dataset_path / seq / "img1"
            # Prendi la prima immagine disponibile
            img_path = next(img_dir.glob("*.jpg"), None)
            
            if img_path:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_t = preprocess(img).unsqueeze(0)
                    emb = model(img_t).flatten().numpy()
                    
                    features.append(emb)
                    sequences.append(seq)
                except Exception as e:
                    print(f"Errore lettura {seq}: {e}")

    return sequences, np.array(features)

def create_smart_splits():
    if not DATASET_TR23_TRAIN.exists():
        print("Dataset non trovato.")
        return

    # 1. Estrai Features Visive
    sequences, features = extract_features(DATASET_TR23_TRAIN)
    
    # 2. Clustering (K-Means)
    # Raggruppa i video in N gruppi basandosi sulla somiglianza visiva
    print(f"Esecuzione K-Means Clustering con {N_CLUSTERS} cluster...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    clusters = kmeans.fit_predict(features)
    
    # Organizza sequenze per cluster
    clustered_seqs = defaultdict(list)
    for seq, cluster_id in zip(sequences, clusters):
        clustered_seqs[cluster_id].append(seq)
        
    # 3. Stratified Split
    # Prendi il 20% da OGNI cluster per metterlo nel validation
    train_final = []
    val_final = []
    
    print("\n--- Analisi Distribuzione Cluster ---")
    for cid in sorted(clustered_seqs.keys()):
        seqs_in_cluster = clustered_seqs[cid]
        # Mischia all'interno del cluster
        random.seed(SEED)
        random.shuffle(seqs_in_cluster)
        
        n_val = int(len(seqs_in_cluster) * VAL_RATIO)
        # Assicuriamoci di avere almeno 1 val se il cluster è piccolo ma > 1
        if n_val == 0 and len(seqs_in_cluster) > 1:
            n_val = 1
            
        val_subset = seqs_in_cluster[:n_val]
        train_subset = seqs_in_cluster[n_val:]
        
        val_final.extend(val_subset)
        train_final.extend(train_subset)
        
        print(f"Cluster {cid}: Totale {len(seqs_in_cluster)} -> Train: {len(train_subset)}, Val: {len(val_subset)}")
        # (Opzionale) Stampa qualche nome per capire cosa c'è nel cluster
        # print(f"   Esempi: {seqs_in_cluster[:3]}")

    # 4. Salva i Manifest
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(SPLIT_DIR / "train.txt", "w") as f:
        f.write("\n".join(sorted(train_final)))
        
    with open(SPLIT_DIR / "val.txt", "w") as f:
        f.write("\n".join(sorted(val_final)))
        
    print("\n------------------------------------------------")
    print(f"Smart Split Completato.")
    print(f"Train: {len(train_final)} sequenze")
    print(f"Val:   {len(val_final)} sequenze")
    print(f"I set sono bilanciati visivamente per coprire {N_CLUSTERS} tipologie di scenari.")
    print("------------------------------------------------")

if __name__ == "__main__":
    create_smart_splits()