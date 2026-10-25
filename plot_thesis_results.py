import os
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION DES CHEMINS ---
OUTPUT_DIR = r"D:\PHD\TACS\Code 16 Oct 2026"
FIGURE_PATH = os.path.join(OUTPUT_DIR, "validation_benchmark_plots.png")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- DATASHEET INITIALIZATION ---
methods = [
    'PageRank\n(PR - Real)', 
    'Machine Learning\n(TF-IDF)', 
    'MarsRank\n(MRM - Influence)', 
    'Node2Vec /\nDeepWalk', 
    'MarsRank\n(MR - Specificity)'
]

f1_scores = [7.6, 7.6, 22.2, 26.5, 28.9]
kendall_concordance = [48.3, 50.0, 58.8, 60.1, 62.3]

# --- MATPLOTLIB CANVAS SETUP ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
plt.subplots_adjust(wspace=0.35)

# ----------------=====================================================
# CHART 1: Discrete Classification Evaluation (F1-Score Metrics)
# ----------------=====================================================
colors1 = ['#e74c3c', '#95a5a6', '#34495e', '#3498db', '#2ecc71']
bars1 = ax1.barh(methods, f1_scores, color=colors1, edgecolor='none', height=0.55)
ax1.set_xlabel('F1-Score (%)', fontsize=11, fontweight='bold', labelpad=10)
ax1.set_title('Discrete Classification Evaluation\n(Capability Profiling F1 Accuracy)', fontsize=12, fontweight='bold', pad=15)
ax1.set_xlim(0, 35)
ax1.grid(axis='x', linestyle='--', alpha=0.5)

# Value annotations for Chart 1
for bar in bars1:
    width = bar.get_width()
    ax1.text(width + 0.6, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
             va='center', ha='left', fontsize=10, fontweight='bold')

# ----------------=====================================================
# CHART 2: Continuous Ranking Evaluation (Kendall Concordance)
# ----------------=====================================================
colors2 = ['#c0392b', '#7f8c8d', '#2c3e50', '#2980b9', '#27ae60']
bars2 = ax2.barh(methods, kendall_concordance, color=colors2, edgecolor='none', height=0.55)
ax2.set_xlabel('Kendall Concordance Rate (%)', fontsize=11, fontweight='bold', labelpad=10)
ax2.set_title('Continuous Ranking Evaluation\n(Concordance Pairs vs Software Token Cost)', fontsize=12, fontweight='bold', pad=15)
ax2.set_xlim(40, 70)
ax2.grid(axis='x', linestyle='--', alpha=0.5)

# Value annotations for Chart 2
for bar in bars2:
    width = bar.get_width()
    ax2.text(width + 0.6, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
             va='center', ha='left', fontsize=10, fontweight='bold')

# --- VISUAL STYLING AND SPINES CLEANUP (Academic Publication Standards) ---
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    ax.tick_params(axis='both', which='major', labelsize=10)

# --- EXPORT IMAGE ---
plt.savefig(FIGURE_PATH, format='png', bbox_inches='tight', dpi=300)
plt.close()

print(f"🥇 Publication-ready benchmark figure saved cleanly to:\n   -> {FIGURE_PATH}")
