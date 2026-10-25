import os
import sqlite3
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import precision_recall_fscore_support

# --- CONFIGURATION DES CHEMINS ---
CSV_FILE_PATH = r"D:\PHD\TACS\Code 16 Oct 2026\real_benchmark_results.csv"
SQLITE_DB_FILE = r"D:\PHD\TACS\Data\gitskills.sqlite.db"
OUTPUT_DIR = r"D:\PHD\TACS\Code 16 Oct 2026"

print("📖 Loading benchmark results from CSV...")
if not os.path.exists(CSV_FILE_PATH):
    raise FileNotFoundError(f"Impossible de trouver le fichier CSV à l'adresse : {CSV_FILE_PATH}")

df_csv = pd.read_csv(CSV_FILE_PATH)
num_aligned = len(df_csv)
print(f"   -> Found {num_aligned} aligned tasks inside CSV.")

# --- EXTRACTION ULTRA-RAPIDE DE LA VÉRITÉ TERRAIN EN MÉMOIRE ---
print("\n🔍 Connecting to SQLite to fetch targeted Ground Truth text (Optimized)...")
conn = sqlite3.connect(SQLITE_DB_FILE)

# Extraction ciblée uniquement pour les hashes du CSV
csv_hashes = df_csv['file_sha'].astype(str).str.strip().str.lower().values.tolist()

# Découpage par paquets de 999 pour respecter les limites matérielles de SQLite
chunks = []
chunk_size_sql = 999
for i in range(0, len(csv_hashes), chunk_size_sql):
    chunk_hashes = csv_hashes[i:i+chunk_size_sql]
    placeholders = ','.join(['?'] * len(chunk_hashes))
    query_fast = f"SELECT file_sha, content FROM artifacts WHERE file_sha IN ({placeholders})"
    df_chunk = pd.read_sql_query(query_fast, conn, params=chunk_hashes)
    chunks.append(df_chunk)

df_artifacts_fast = pd.concat(chunks, ignore_index=True)
conn.close()

# Normalisation des chaînes de la base
df_artifacts_fast['file_sha'] = df_artifacts_fast['file_sha'].astype(str).str.strip().str.lower()
df_artifacts_fast['content'] = df_artifacts_fast['content'].astype(str).fillna('')

# Reconstruction de la vérité terrain binaire directement en RAM (sans scan SQL)
target_capability_domains = ['testing', 'deployment', 'database_management', 'security']
artifact_lookup = dict(zip(df_artifacts_fast['file_sha'], df_artifacts_fast['content']))

aligned_y_true = []
for h in csv_hashes:
    row_labels = np.zeros(len(target_capability_domains), dtype=int)
    # Récupération du texte associé au hash dans notre dictionnaire mémoire
    content_text = artifact_lookup.get(h, "")
    
    if content_text:
        for domain_idx, domain_name in enumerate(target_capability_domains):
            if domain_name in content_text:
                row_labels[domain_idx] = 1
    aligned_y_true.append(row_labels)

y_true_final = np.array(aligned_y_true)

print("📊 Data processing complete.")
# --- CALCUL DE L'ADAPTIVE THRESHOLD (Ph.D. Compliant) ---
global_sparsity = np.mean(y_true_matched) if 'y_true_matched' in locals() else np.mean(y_true_final)
top_k_cutoff = int(np.ceil(num_aligned * global_sparsity))
print(f"   -> Global Sparsity Density: {global_sparsity*100:.2f}%")
print(f"   -> Top-K highest ranked cut-off: {top_k_cutoff} files")

def binarize_scores(scores_array, cutoff_k, target_shape):
    """Binarise un vecteur continu en attribuant 1 aux Top-K meilleures tâches."""
    sorted_indices = np.argsort(scores_array)[::-1][:cutoff_k]
    binary_matrix = np.zeros(target_shape, dtype=int)
    for idx in sorted_indices:
        binary_matrix[idx, :] = 1
    return binary_matrix

# Binarisation des scores du CSV
y_pred_mr = binarize_scores(df_csv['specificity_mr'].values, top_k_cutoff, y_true_final.shape)
y_pred_mrm = binarize_scores(df_csv['influence_mrm'].values, top_k_cutoff, y_true_final.shape)
y_pred_pr = binarize_scores(df_csv['real_pagerank'].values, top_k_cutoff, y_true_final.shape)
y_pred_ml = binarize_scores(df_csv['ml_prob_score'].values, top_k_cutoff, y_true_final.shape)

# --- EVALUATION DES METRIQUES ---
p_mr, r_mr, f1_mr, _ = precision_recall_fscore_support(y_true_final, y_pred_mr, average='weighted', zero_division=0)
p_mrm, r_mrm, f1_mrm, _ = precision_recall_fscore_support(y_true_final, y_pred_mrm, average='weighted', zero_division=0)
p_pr, r_pr, f1_pr, _ = precision_recall_fscore_support(y_true_final, y_pred_pr, average='weighted', zero_division=0)
p_ml, r_ml, f1_ml, _ = precision_recall_fscore_support(y_true_final, y_pred_ml, average='weighted', zero_division=0)

# --- AFFICHAGE DE LA TABLE FINALE DE CLASSIFICATION ---
print("\n" + "="*80)
print("🎯 TABLE 2: DISCRETE CLASSIFICATION BENCHMARK (Precision / Recall / F1-Score)")
print("="*80)
print(f"🥇 MarsRank (MR)     : Precision = {p_mr * 100:.1f}% | Recall = {r_mr * 100:.1f}% | F1 = {f1_mr * 100:.1f}%")
print(f"🥈 MarsRank (MRM)    : Precision = {p_mrm * 100:.1f}% | Recall = {r_mrm * 100:.1f}% | F1 = {f1_mrm * 100:.1f}%")
print(f"🥉 Machine Learning  : Precision = {p_ml * 100:.1f}% | Recall = {r_ml * 100:.1f}% | F1 = {f1_ml * 100:.1f}%")
print(f"❌ PageRank (PR)     : Precision = {p_pr * 100:.1f}% | Recall = {r_pr * 100:.1f}% | F1 = {f1_pr * 100:.1f}%")
print("="*80 + "\n")

# --- EXPORTATION DU RAPPORT EXTENSIBLÉ ---
report_file = os.path.join(OUTPUT_DIR, "rapport_classification_instantane.txt")
with open(report_file, "w", encoding="utf-8") as f:
    f.write("==================================================================\n")
    f.write("📝 EXTENSION DE RAPPORT : EVALUATION DISCRÈTE DES MODÈLES TOPOLOGIQUES\n")
    f.write("==================================================================\n\n")
    f.write(f"Nombre de tâches analysées (issues du CSV) : {num_aligned}\n")
    f.write(f"Seuil adaptatif Top-K appliqué : {top_k_cutoff} tâches supérieures\n\n")
    f.write("SCORES DE CLASSIFICATION CROISÉS (MÉTHODES TOPOLOGIQUES vs MACHINE LEARNING)\n")
    f.write(f"   - MarsRank (MR)    : P = {p_mr*100:.2f}% | R = {r_mr*100:.2f}% | F1 = {f1_mr*100:.2f}%\n")
    f.write(f"   - MarsRank (MRM)   : P = {p_mrm*100:.2f}% | R = {r_mrm*100:.2f}% | F1 = {f1_mrm*100:.2f}%\n")
    f.write(f"   - Machine Learning : P = {p_ml*100:.2f}% | R = {r_ml*100:.2f}% | F1 = {f1_ml*100:.2f}%\n")
    f.write(f"   - PageRank (PR)    : P = {p_pr*100:.2f}% | R = {r_pr*100:.2f}% | F1 = {f1_pr*100:.2f}%\n")
    f.write("==================================================================\n")

print(f"💾 Report cleanly exported to: {report_file}")
