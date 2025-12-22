import torch
from config import *

# 1. Imposta i percorsi
INPUT_PATH = MODEL_PATH / "re-ids" / "model_soup.deit_l.pth"  # Il file che hai rinominato
OUTPUT_PATH = MODEL_PATH / "re-ids" / "ViT-L_16.pt" # Il file pulito che useremo

print(f"Caricamento da {INPUT_PATH}...")
try:
    # Carica il checkpoint sulla CPU
    checkpoint = torch.load(INPUT_PATH, map_location='cpu')
    
    # 2. Estrai i pesi reali (state_dict)
    # A volte i pesi sono dentro una chiave 'state_dict', a volte sono diretti.
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        print("Trovata chiave 'state_dict'. Estrazione...")
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 3. Pulizia dei prefissi (es. 'module.')
    # Se il modello è stato trainato su più GPU, i nomi hanno 'module.' davanti.
    # Dobbiamo rimuoverlo perché noi usiamo una GPU sola o CPU.
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if k.startswith('module.'):
            name = k[7:] # rimuove 'module.'
        new_state_dict[name] = v
        
    # 4. Salvataggio
    torch.save(new_state_dict, OUTPUT_PATH)
    print(f"Fatto! File salvato come: {OUTPUT_PATH}")
    print("Ora usa questo file nel codice del tracker.")

except Exception as e:
    print(f"Errore: {e}")