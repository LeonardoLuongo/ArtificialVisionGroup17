import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# 1. INSERISCI QUI I TUOI DATI DI VALUTAZIONE (HOTA)
# -----------------------------------------------------------------------------
# Esempio: Confronto tra diversi Detector mantenendo fisso BoostTrack
# Oppure: Confronto tra diversi Tracker (SORT vs BoostTrack) mantenendo fisso YOLO
# Sotto metto dati IPOTETICI verosimili. Sostituiscili con i tuoi!

models = ['YOLOv5mu + SORT', 'YOLOv8m + BoostTrack', 'YOLO12m + BoostTrack']

# Metriche HOTA (Valori 0-100 o 0-1)
# HOTA è la media geometrica di DetA e AssA (più o meno)
hota_score = [48.5, 62.1, 65.4]  # Higher Order Tracking Accuracy
deta_score = [60.2, 75.5, 78.2]  # Detection Accuracy
assa_score = [39.5, 51.0, 54.8]  # Association Accuracy

# -----------------------------------------------------------------------------
# 2. CONFIGURAZIONE GRAFICO
# -----------------------------------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
width = 0.25 

# Barre raggruppate
rects1 = ax.bar(x - width, hota_score, width, label='HOTA', color='#27AE60', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x, deta_score, width, label='DetA (Detection)', color='#2980B9', edgecolor='black', alpha=0.9)
rects3 = ax.bar(x + width, assa_score, width, label='AssA (Association)', color='#F39C12', edgecolor='black', alpha=0.9)

# -----------------------------------------------------------------------------
# 3. FORMATTAZIONE
# -----------------------------------------------------------------------------
ax.set_ylabel('Score (%)', fontweight='bold')
ax.set_title('Impatto dei componenti sulla metrica HOTA', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='bold')
ax.set_ylim(0, 100) # Assumendo scala 0-100 (se hai 0-1, cambia a 1.1)

ax.legend(loc='upper left', ncol=3, frameon=True, edgecolor='black')
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Annotazioni
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# -----------------------------------------------------------------------------
# 4. TABELLA SOTTOSTANTE
# -----------------------------------------------------------------------------
table_data = [
    [f"{v:.1f}" for v in hota_score],
    [f"{v:.1f}" for v in deta_score],
    [f"{v:.1f}" for v in assa_score]
]
rows = ['HOTA', 'DetA', 'AssA']

the_table = plt.table(cellText=table_data, rowLabels=rows, colLabels=models,
                      loc='bottom', bbox=[0.0, -0.35, 1.0, 0.25], cellLoc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(1, 1.3)

plt.subplots_adjust(left=0.1, bottom=0.30)

# Salvataggio
plt.savefig('confronto_HOTA.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()