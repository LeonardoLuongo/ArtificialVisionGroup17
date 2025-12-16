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
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Setup path
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(ROOT_DIR))
from config import DATASET_TR23_TRAIN, SPLIT_DIR, OUTPUT_DIR
from src.utils.get_best_device import get_best_device

# Parametri
VAL_RATIO = 0.2
SEED = 42
FRAMES_PER_SEQ = 3  
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10   

def extract_features(dataset_path):
    device = get_best_device()
    print(f"Caricamento modello ResNet18 su {device}...")
    
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1]) 
    model.to(device)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features_dict = {} # Mappa seq_name -> feature_vector
    
    seq_dirs = sorted([s for s in os.listdir(dataset_path) if (dataset_path / s).is_dir()])
    
    print(f"Estrazione feature da {len(seq_dirs)} sequenze...")
    
    with torch.no_grad():
        for seq in tqdm(seq_dirs):
            img_dir = dataset_path / seq / "img1"
            all_images = sorted(list(img_dir.glob("*.jpg")))
            
            if not all_images: continue
            
            indices = np.linspace(0, len(all_images) - 1, FRAMES_PER_SEQ, dtype=int)
            selected_imgs = [all_images[i] for i in indices]
            
            seq_feats = []
            for img_path in selected_imgs:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_t = preprocess(img).unsqueeze(0).to(device)
                    emb = model(img_t).flatten().cpu().numpy()
                    seq_feats.append(emb)
                except Exception:
                    pass

            if seq_feats:
                avg_feat = np.mean(seq_feats, axis=0)
                features_dict[seq] = avg_feat

    return features_dict

def find_best_k(features, min_k=2, max_k=10):
    print("\n--- Ricerca Automatica K (Clustering) ---")
    best_k = min_k
    best_score = -1
    best_labels = None
    
    limit_k = min(len(features), max_k)
    if len(features) < 2: return 1, np.zeros(len(features))

    for k in range(min_k, limit_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            
    print(f"--> Miglior K trovato: {best_k} (Silhouette Score: {best_score:.4f})")
    return best_k, best_labels

def visualize_split_quality(features_dict, train_seqs, val_seqs):
    """
    Genera un grafico 2D per mostrare quanto Train e Val si sovrappongono.
    """
    print("\nGenerazione grafico di qualità dello split...")
    
    all_seqs = train_seqs + val_seqs
    X = np.array([features_dict[s] for s in all_seqs])
    
    # Riduciamo a 2 dimensioni con PCA per poter disegnare
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Creiamo due liste di coordinate
    train_indices = [i for i, s in enumerate(all_seqs) if s in train_seqs]
    val_indices = [i for i, s in enumerate(all_seqs) if s in val_seqs]
    
    plt.figure(figsize=(10, 8))
    
    # Disegna Train (Blu)
    plt.scatter(X_pca[train_indices, 0], X_pca[train_indices, 1], 
                c='blue', alpha=0.5, label='Train Set', s=50)
    
    # Disegna Val (Rosso)
    plt.scatter(X_pca[val_indices, 0], X_pca[val_indices, 1], 
                c='red', marker='x', alpha=0.9, label='Val Set', s=70)
    
    plt.title(f"Distribuzione Train vs Val (PCA)\nObiettivo: I punti rossi devono essere ben distribuiti tra i blu")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_plot = OUTPUT_DIR / "split_quality_check.png"
    if not OUTPUT_DIR.exists(): OUTPUT_DIR.mkdir(parents=True)
    
    plt.savefig(output_plot)
    print(f"--> Grafico salvato in: {output_plot}")
    print("    APRILO! Se i punti rossi coprono bene l'area dei blu, lo split è ottimo.")

def create_smart_splits():
    if not DATASET_TR23_TRAIN.exists():
        print(f"Dataset non trovato in {DATASET_TR23_TRAIN}")
        return

    # 1. Estrazione Features
    features_map = extract_features(DATASET_TR23_TRAIN)
    sequences = sorted(list(features_map.keys()))
    features_matrix = np.array([features_map[s] for s in sequences])
    
    if len(sequences) == 0: return

    # 2. Clustering
    n_clusters, clusters = find_best_k(features_matrix, MIN_CLUSTERS, MAX_CLUSTERS)
    
    clustered_seqs = defaultdict(list)
    for seq, cluster_id in zip(sequences, clusters):
        clustered_seqs[cluster_id].append(seq)
        
    # 3. Stratified Split
    train_final = []
    val_final = []
    
    print("\n--- Generazione Split ---")
    for cid in sorted(clustered_seqs.keys()):
        seqs_in_cluster = clustered_seqs[cid]
        random.seed(SEED)
        random.shuffle(seqs_in_cluster)
        
        n_val = int(len(seqs_in_cluster) * VAL_RATIO)
        if n_val == 0 and len(seqs_in_cluster) > 1: n_val = 1
            
        val_subset = seqs_in_cluster[:n_val]
        train_subset = seqs_in_cluster[n_val:]
        
        val_final.extend(val_subset)
        train_final.extend(train_subset)
        print(f"Cluster {cid}: {len(train_subset)} Train, {len(val_subset)} Val")

    # 4. Salva i file
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_DIR / "train.txt", "w") as f: f.write("\n".join(sorted(train_final)))
    with open(SPLIT_DIR / "val.txt", "w") as f: f.write("\n".join(sorted(val_final)))
    
    # 5. VERIFICA VISIVA (Nuovo!)
    visualize_split_quality(features_map, train_final, val_final)

if __name__ == "__main__":
    create_smart_splits()