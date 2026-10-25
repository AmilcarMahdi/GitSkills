import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

# --- PATH CONFIGURATION ---
CSV_FILE_PATH = r"D:\PHD\TACS\Code 16 Oct 2026\real_benchmark_results.csv"
SQLITE_DB_FILE = r"D:\PHD\TACS\Data\gitskills.sqlite.db"
OUTPUT_DIR = r"D:\PHD\TACS\Code 16 Oct 2026"

print("📖 Loading benchmark results map from CSV...")
if not os.path.exists(CSV_FILE_PATH):
    raise FileNotFoundError(f"Missing CSV benchmark file at: {CSV_FILE_PATH}")

df_csv = pd.read_csv(CSV_FILE_PATH)
df_csv['file_sha'] = df_csv['file_sha'].astype(str).str.strip().str.lower()

# Convert pre-computed model scores into active memory dictionaries
csv_lookup = df_csv.set_index('file_sha').to_dict(orient='index')
valid_csv_hashes = set(csv_lookup.keys())
print(f"   -> Successfully indexed {len(valid_csv_hashes)} pre-computed method signatures.")

# --- SOTA DEFINITION OF FINE-GRAINED SKILLSVOTE FILTERS ---
fine_grained_topics = {
    'Python Testing (PyTest)': "content LIKE '%testing%' AND content LIKE '%pytest%'",
    'JS Testing (Jest/Mocha)': "content LIKE '%testing%' AND (content LIKE '%jest%' OR content LIKE '%mocha%')",
    'Docker Infrastructure':   "content LIKE '%deployment%' AND content LIKE '%docker%'",
    'K8s Orchestration':       "content LIKE '%deployment%' AND content LIKE '%kubernetes%'",
    'SQL Migration (Alembic)': "content LIKE '%database_management%' AND content LIKE '%alembic%'",
    'NoSQL Cache (Redis)':     "content LIKE '%database_management%' AND content LIKE '%redis%'",
    'Static Security Linting': "content LIKE '%security%' AND content LIKE '%bandit%'"
}

print("\n🔍 STAGE 1: Interrogating Entire Database via SkillsVote Framework Filters...")
conn = sqlite3.connect(SQLITE_DB_FILE)
topic_buckets = {topic: [] for topic in fine_grained_topics.keys()}

# Directly query the millions of database rows to construct fine-grained candidate libraries
for topic_name, sql_condition in fine_grained_topics.items():
    print(f"   -> Processing full DB sweep for topic: {topic_name}")
    query_global = f"SELECT file_sha, content, body_chars FROM artifacts WHERE {sql_condition}"
    df_global_match = pd.read_sql_query(query_global, conn)
    
    if not df_global_match.empty:
        df_global_match['file_sha'] = df_global_match['file_sha'].astype(str).str.strip().str.lower()
        
        # Intersect the raw DB matches against the valid pre-computed graph tasks pool
        for _, row in df_global_match.iterrows():
            h_clean = row['file_sha']
            if h_clean in valid_csv_hashes:
                record = {
                    'file_sha': h_clean,
                    'estimated_tokens': csv_lookup[h_clean]['estimated_tokens'],
                    'specificity_mr': csv_lookup[h_clean]['specificity_mr'],
                    'influence_mrm': csv_lookup[h_clean]['influence_mrm'],
                    'real_pagerank': csv_lookup[h_clean]['real_pagerank'],
                    'ml_prob_score': csv_lookup[h_clean]['ml_prob_score']
                }
                topic_buckets[topic_name].append(record)

conn.close()
print("   -> Global database sweeps and intersection mapping complete.")

# =====================================================================
# STAGE 2: PROCESS CANDIDATE SETS AND EXECUTE METRICS
# =====================================================================
print("\n🤝 Executing STAGE 2: Specialization-Aware Re-ranking...")
POOL_SIZE = 15
context_pools = []

for t_name, records in topic_buckets.items():
    if len(records) == 0:
        continue
    df_sub_topic = pd.DataFrame(records)
    # Complexity sort across the framework
    df_sub_topic = df_sub_topic.sort_values(by='estimated_tokens').reset_index(drop=True)
    
    topic_threshold = df_sub_topic['estimated_tokens'].quantile(0.75)
    df_sub_topic['is_relevance_ground_truth'] = (df_sub_topic['estimated_tokens'] >= topic_threshold).astype(int)
    
    for i in range(0, len(df_sub_topic), POOL_SIZE):
        chunk = df_sub_topic[i:i + POOL_SIZE]
        if len(chunk) >= 5:
            context_pools.append(chunk)

print(f"   -> Engineered {len(context_pools)} database-driven localized candidate sets.")

# Robust Metric Engines (Fixed tuple-unpacking layout)
def compute_mrr(relevance_array):
    idx = np.where(relevance_array == 1)[0]
    return 1.0 / (idx[0] + 1) if len(idx) > 0 else 0.0

def compute_recall_at_k(relevance_array, k):
    total_positives = np.sum(relevance_array)
    if total_positives == 0:
        return 1.0
    return np.sum(relevance_array[:k]) / total_positives

metrics = {
    'mr':  {'mrr': [], 'ndcg3': [], 'ndcg5': [], 'rec1': [], 'rec3': [], 'rec5': []},
    'mrm': {'mrr': [], 'ndcg3': [], 'ndcg5': [], 'rec1': [], 'rec3': [], 'rec5': []},
    'pr':  {'mrr': [], 'ndcg3': [], 'ndcg5': [], 'rec1': [], 'rec3': [], 'rec5': []},
    'ml':  {'mrr': [], 'ndcg3': [], 'ndcg5': [], 'rec1': [], 'rec3': [], 'rec5': []}
}

for pool in context_pools:
    y_true = pool['is_relevance_ground_truth'].values
    if np.sum(y_true) == 0: 
        continue
        
    for key, score_col in [('mr', 'specificity_mr'), ('mrm', 'influence_mrm'), ('ml', 'ml_prob_score'), ('pr', 'real_pagerank')]:
        scores = pool[score_col].values
        rank_idx = np.argsort(scores)[::-1]
        y_true_ranked = y_true[rank_idx]
        
        metrics[key]['mrr'].append(compute_mrr(y_true_ranked))
        metrics[key]['rec1'].append(compute_recall_at_k(y_true_ranked, 1))
        metrics[key]['rec3'].append(compute_recall_at_k(y_true_ranked, 3))
        metrics[key]['rec5'].append(compute_recall_at_k(y_true_ranked, 5))
        
        ndcg_true = y_true.reshape(1, -1)
        ndcg_scores = scores.reshape(1, -1)
        metrics[key]['ndcg3'].append(ndcg_score(ndcg_true, ndcg_scores, k=3))
        metrics[key]['ndcg5'].append(ndcg_score(ndcg_true, ndcg_scores, k=5))

# --- PRINT INTEGRATED REPORT ---
print("\n" + "="*110)
print("🏆 UNIFIED ARCHITECTURE BENCHMARK: SKILLSVOTE (Global Retrieval) + MARSRANK METHODS")
print("="*110)
print(f"{'Recommendation Pipeline Engine':<38} | {'MRR':<6} | {'nDCG@3':<6} | {'nDCG@5':<6} | {'Rec@1':<6} | {'Rec@3':<6} | {'Rec@5':<6}")
print("-"*110)

for key, name in [('mr', '🚀 SkillsVote + MarsRank (MR)'), ('mrm', '🚀 SkillsVote + MarsRank (MRM)'), ('ml', '🔗 SkillsVote + Machine Learning'), ('pr', '❌ SkillsVote + PageRank Baseline')]:
    m = metrics[key]
    print(f"{name:<38} | {np.mean(m['mrr']):.4f} | {np.mean(m['ndcg3']):.4f} | {np.mean(m['ndcg5']):.4f} | {np.mean(m['rec1'])*100:.1f}%  | {np.mean(m['rec3'])*100:.1f}%  | {np.mean(m['rec5'])*100:.1f}%")
print("="*110 + "\n")

# --- SAVE REPORT ---
report_file = os.path.join(OUTPUT_DIR, "rapport_integration_skillsvote_global.txt")
with open(report_file, "w", encoding="utf-8") as f:
    f.write("==================================================================\n")
    f.write("📝 PH.D. DISSERTATION REPORT: GLOBAL SKILLSVOTE & MARSRANK METHODS\n")
    f.write("==================================================================\n\n")
    f.write(f"Pipeline Setup: Stage 1 (Global DB SkillsVote Filtering) -> Stage 2 (Multi-Method Re-ranking)\n\n")
    for key, name in [('mr', 'SkillsVote + MarsRank (MR)'), ('mrm', 'SkillsVote + MarsRank (MRM)'), ('ml', 'SkillsVote + Machine Learning'), ('pr', 'SkillsVote + PageRank')]:
        m = metrics[key]
        f.write(f"- {name}:\n")
        f.write(f"  MRR = {np.mean(m['mrr']):.3f} | nDCG@3 = {np.mean(m['ndcg3']):.3f} | nDCG@5 = {np.mean(m['ndcg5']):.3f}\n")
        f.write(f"  Recall@1 = {np.mean(m['rec1'])*100:.1f}% | Recall@3 = {np.mean(m['rec3'])*100:.1f}% | Recall@5 = {np.mean(m['rec5'])*100:.1f}%\n\n")

print(f"💾 Global integrated framework report cleanly saved to:\n   -> {report_file}")