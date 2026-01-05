import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# 1. INSERISCI I TUOI DATI QUI
# -----------------------------------------------------------------------------
# Se non hai i dati esatti di SORT/DeepSORT sul tuo test set, 
# puoi lanciare una validazione veloce su 1 solo video per ottenerli.
# Altrimenti, questi sotto sono valori tipici per questo dataset (placeholder).

trackers = ['SORT', 'DeepSORT', 'BoostTrack (Ours)']

# HOTA: Media geometrica tra Detection e Association
hota_scores = [42.5, 51.2, 63.8] 

# DetA: Dovrebbe essere quasi identica se usi lo stesso YOLO per tutti
deta_scores = [75.0, 75.2, 76.5] 

# AssA: QUI SI VEDE LA DIFFERENZA (Il vero guadagno del tuo lavoro)
assa_scores = [24.1, 35.5, 53.4] 

# -----------------------------------------------------------------------------
# 2. CONFIGURAZIONE GRAFICO
# -----------------------------------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(trackers))
width = 0.25 

# Barre
rects1 = ax.bar(x - width, hota_scores, width, label='HOTA', color='#27AE60', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x, deta_scores, width, label='DetA (Detection)', color='#2980B9', edgecolor='black', alpha=0.9)
# AssA in evidenza (Arancione/Rosso) perché è il punto chiave
rects3 = ax.bar(x + width, assa_scores, width, label='AssA (Association)', color='#C0392B', edgecolor='black', alpha=0.9)

# -----------------------------------------------------------------------------
# 3. FORMATTAZIONE
# -----------------------------------------------------------------------------
ax.set_ylabel('Score (%)', fontweight='bold')
ax.set_title('Confronto Algoritmi di Tracking (Backbone: YOLOv12m)', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(trackers, fontweight='bold')
ax.set_ylim(0, 100) # Scala 0-100

ax.legend(loc='upper left', ncol=3, frameon=True, edgecolor='black')
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Annotazioni sopra le barre
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# -----------------------------------------------------------------------------
# 4. TABELLA SOTTOSTANTE
# -----------------------------------------------------------------------------
table_data = [
    [f"{v:.1f}" for v in hota_scores],
    [f"{v:.1f}" for v in deta_scores],
    [f"{v:.1f}" for v in assa_scores]
]
rows = ['HOTA', 'DetA', 'AssA']

the_table = plt.table(cellText=table_data, rowLabels=rows, colLabels=trackers,
                      loc='bottom', bbox=[0.0, -0.35, 1.0, 0.25], cellLoc='center')

the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(1, 1.3)

plt.subplots_adjust(left=0.1, bottom=0.30)

# Salvataggio
plt.savefig('confronto_trackers.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()