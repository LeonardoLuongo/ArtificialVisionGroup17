import matplotlib.pyplot as plt
import numpy as np

# Dati forniti
labels = [
    'ByteTrack\n(Baseline)', 
    'BotSORT\n+ ECC', 
    'BoostTrack\n+ OSNet', 
    'RobustMapper\n(Final)'
]
values = [0.5484, 0.5897, 0.6756, 0.753982]

# Configurazione stile accademico
plt.style.use('seaborn-v0_8-whitegrid') # Stile pulito
fig, ax = plt.subplots(figsize=(10, 6))

# Creazione delle barre con un gradiente di blu
colors = ['#ced4da', '#a8dadc', '#457b9d', '#1d3557']
bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2, width=0.6)

# Aggiunta dei valori sopra le barre
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 5 punti di offset verticale
                textcoords="offset points",
                ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color='#1d3557')

# Configurazione assi e titoli
ax.set_ylabel('Valore AssA (Association Accuracy)', fontsize=13, fontweight='bold')
ax.set_title('Evoluzione delle Performance di Associazione nel MOT', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0.4, 0.85) # Range ottimizzato per visualizzare i balzi

# Aggiunta di una freccia per evidenziare il miglioramento totale
improvement = ((values[-1] - values[0]) / values[0]) * 100
ax.annotate(f'Miglioramento Totale: +{improvement:.1f}%', 
            xy=(0, values[0]), xycoords='data',
            xytext=(0.5, 0.82), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
            fontsize=12, fontweight='bold', bbox=dict(boxstyle="round", fc="none", ec="black"))

# Pulizia finale
plt.xticks(fontsize=11, fontweight='bold')
plt.yticks(fontsize=11)
plt.tight_layout()

# Salvataggio
plt.savefig('evoluzione_assa.png', dpi=300)
plt.show()