import os
from pathlib import Path

from config import DATASET_TR23_TEST

# --- Qui inserisci o assicurati di avere le tue definizioni ---
# DATASET_TR23 = Path("percorso/della/tua/cartella") 
# DATASET_TR23_TEST = DATASET_TR23 / "test"
# -----------------------------------------------------------

def rinomina_dataset(target_dir):
    if not target_dir.exists():
        print(f"Errore: Il percorso {target_dir} non esiste.")
        return

    print(f"Cartella target: {target_dir}")
    print("Scegli l'operazione:")
    print("1: SNMOT-XXX -> XXX")
    print("2: XXX -> SNMOT-XXX")
    
    scelta = input("Inserisci 1 o 2: ")
    
    if scelta not in ['1', '2']:
        print("Scelta non valida.")
        return

    count = 0
    # Iteriamo solo sulle directory dentro DATASET_TR23_TEST
    for cartella in target_dir.iterdir():
        if not cartella.is_dir():
            continue
            
        vecchio_nome = cartella.name
        nuovo_nome = None

        if scelta == '1':
            # Da SNMOT-116 a 116
            if vecchio_nome.startswith("SNMOT-"):
                num_str = vecchio_nome.replace("SNMOT-", "")
                if num_str.isdigit() and 116 <= int(num_str) <= 200:
                    nuovo_nome = num_str
        
        elif scelta == '2':
            # Da 116 a SNMOT-116
            if vecchio_nome.isdigit() and 116 <= int(vecchio_nome) <= 200:
                nuovo_nome = f"SNMOT-{vecchio_nome}"

        if nuovo_nome:
            nuovo_path = cartella.parent / nuovo_nome
            
            if not nuovo_path.exists():
                cartella.rename(nuovo_path)
                print(f"Rinominata: {vecchio_nome} -> {nuovo_nome}")
                count += 1
            else:
                print(f"Salto {vecchio_nome}: {nuovo_nome} esiste già.")

    print(f"\nFatto! Cartelle rinominate: {count}")

# Esecuzione
if __name__ == "__main__":
    # Assicurati che DATASET_TR23_TEST sia definita prima di chiamare la funzione
    # Esempio: rinomina_dataset(DATASET_TR23_TEST)
    try:
        rinomina_dataset(DATASET_TR23_TEST)
    except NameError:
        print("Errore: La variabile DATASET_TR23_TEST non è definita in questo script.")