import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# 1. DATI REALI 
# -----------------------------------------------------------------------------
models = ['YOLOv5mu', 'YOLOv8m', 'YOLO11m', 'YOLO12m']

# Valori estratti dai tuoi log
map_50    = [0.98097, 0.98106, 0.98083, 0.98354]
recall    = [0.95117, 0.95613, 0.95240, 0.95865]
precision = [0.97150, 0.97598, 0.97090, 0.97890]

# -----------------------------------------------------------------------------
# 2. CONFIGURAZIONE GRAFICO
# -----------------------------------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(models))
width = 0.25 

# Barre
rects1 = ax.bar(x - width, map_50, width, label='mAP@0.50', color='#2E86C1', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x, recall, width, label='Recall', color='#E74C3C', edgecolor='black', alpha=0.9)
rects3 = ax.bar(x + width, precision, width, label='Precision', color='#95A5A6', edgecolor='black', alpha=0.9)

# -----------------------------------------------------------------------------
# 3. FORMATTAZIONE
# -----------------------------------------------------------------------------
ax.set_ylabel('Score (0-1)', fontweight='bold')
ax.set_title('Confronto Detection', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='bold')

# Zoom su Y per vedere le differenze
ax.set_ylim(0.90, 1.02) 
ax.legend(loc='upper center', ncol=3, frameon=True, edgecolor='black')
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Annotazioni valori
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', rotation=90, fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# -----------------------------------------------------------------------------
# 4. TABELLA SOTTOSTANTE
# -----------------------------------------------------------------------------
table_data = [
    [f"{v:.5f}" for v in map_50],
    [f"{v:.5f}" for v in recall],
    [f"{v:.5f}" for v in precision]
]
rows = ['mAP@0.50', 'Recall', 'Precision']

the_table = plt.table(cellText=table_data, rowLabels=rows, colLabels=models,
                      loc='bottom', bbox=[0.0, -0.40, 1.0, 0.25], cellLoc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(9)
the_table.scale(1, 1.3)

plt.subplots_adjust(left=0.1, bottom=0.30)

plt.savefig('confronto_detection.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()