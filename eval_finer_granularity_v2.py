import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

CSV_FILE_PATH = r"D:\PHD\TACS\Code 16 Oct 2026\real_benchmark_results.csv"
SQLITE_DB_FILE = r"D:\PHD\TACS\Data\gitskills.sqlite.db"
OUTPUT_DIR = r"D:\PHD\TACS\Code 16 Oct 2026"

print("📖 Loading benchmark results map from CSV...")
df_csv = pd.read_csv(CSV_FILE_PATH)
df_csv['file_sha'] = df_csv['file_sha'].astype(str).str.strip().str.lower()
csv_lookup = df_csv.set_index('file_sha').to_dict(orient='index')
valid_csv_hashes = set(csv_lookup.keys())

# LEAKAGE CONTROL: Separate metadata filtering keys from ground truth metrics evaluation
fine_grained_topics = {
    'Python Testing (PyTest)': "content LIKE '%pytest%'",
    'JS Testing (Jest/Mocha)': "content LIKE '%jest%' OR content LIKE '%mocha%'",
    'Docker Infrastructure':   "content LIKE '%docker%'",
    'K8s Orchestration':       "content LIKE '%kubernetes%'",
    'SQL Migration (Alembic)': "content LIKE '%alembic%'",
    'NoSQL Cache (Redis)':     "content LIKE '%redis%'",
    'Static Security Linting': "content LIKE '%bandit%'"
}

conn = sqlite3.connect(SQLITE_DB_FILE)
topic_buckets = {topic: [] for topic in fine_grained_topics.keys()}

for topic_name, sql_condition in fine_grained_topics.items():
    query_global = f"SELECT file_sha FROM artifacts WHERE {sql_condition}"
    df_global_match = pd.read_sql_query(query_global, conn)
    if not df_global_match.empty:
        df_global_match['file_sha'] = df_global_match['file_sha'].astype(str).str.strip().str.lower()
        for h_clean in df_global_match['file_sha'].values:
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

POOL_SIZE = 15
context_pools = []
for t_name, records in topic_buckets.items():
    if len(records) == 0:
        continue
    df_sub_topic = pd.DataFrame(records)
    df_sub_topic = df_sub_topic.sort_values(by='estimated_tokens').reset_index(drop=True)
    
    # Ground truth is established purely by independent out-of-context token complexity quantiles
    topic_threshold = df_sub_topic['estimated_tokens'].quantile(0.75)
    df_sub_topic['is_relevance_ground_truth'] = (df_sub_topic['estimated_tokens'] >= topic_threshold).astype(int)
    
    for i in range(0, len(df_sub_topic), POOL_SIZE):
        chunk = df_sub_topic[i:i + POOL_SIZE]
        if len(chunk) >= 5:
            context_pools.append(chunk)

# CRITICAL FIX: Mathematical compliance for Mean Reciprocal Rank (MRR)
def compute_mrr_fixed(relevance_array):
    indices = np.where(relevance_array == 1)[0]
    if len(indices) == 0:
        return 0.0
    first_rank = indices[0] + 1 # 1-based ranking index
    return 1.0 / first_rank

def compute_recall_at_k(relevance_array, k):
    total_positives = np.sum(relevance_array)
    if total_positives == 0:
        return 0.0
    return np.sum(relevance_array[:k]) / total_positives

metrics = {k: {'mrr': [], 'ndcg3': [], 'ndcg5': [], 'rec1': [], 'rec3': [], 'rec5': [], 'ranks': []} for k in ['mr', 'mrm', 'ml', 'pr']}

for pool in context_pools:
    y_true = pool['is_relevance_ground_truth'].values
    if np.sum(y_true) == 0:
        continue
    for key, score_col in [('mr', 'specificity_mr'), ('mrm', 'influence_mrm'), ('ml', 'ml_prob_score'), ('pr', 'real_pagerank')]:
        scores = pool[score_col].values
        rank_idx = np.argsort(scores)[::-1]
        y_true_ranked = y_true[rank_idx]
        
        mrr_val = compute_mrr_fixed(y_true_ranked)
        metrics[key]['mrr'].append(mrr_val)
        metrics[key]['rec1'].append(compute_recall_at_k(y_true_ranked, 1))
        metrics[key]['rec3'].append(compute_recall_at_k(y_true_ranked, 3))
        metrics[key]['rec5'].append(compute_recall_at_k(y_true_ranked, 5))
        
        # Track rank distribution profiles
        hit_indices = np.where(y_true_ranked == 1)[0]
        if len(hit_indices) > 0:
            metrics[key]['ranks'].append(hit_indices[0] + 1)
        
        ndcg_true = y_true.reshape(1, -1)
        ndcg_scores = scores.reshape(1, -1)
        metrics[key]['ndcg3'].append(ndcg_score(ndcg_true, ndcg_scores, k=3))
        metrics[key]['ndcg5'].append(ndcg_score(ndcg_true, ndcg_scores, k=5))

print("\n" + "="*110)
print("🏆 AUDITED UNIFIED BENCHMARK: SKILLSVOTE (Retrieval) + MARSRANK METHODS")
print("="*110)
print(f"{'Recommendation Pipeline Engine':<38} | {'MRR':<6} | {'nDCG@3':<6} | {'nDCG@5':<6} | {'Rec@1':<6} | {'Rec@3':<6} | {'Rec@5':<6}")
print("-"*110)
for key, name in [('mr', '🚀 SkillsVote + MarsRank (MR)'), ('mrm', '🚀 SkillsVote + MarsRank (MRM)'), ('ml', '🔗 SkillsVote + Machine Learning'), ('pr', '❌ SkillsVote + PageRank Baseline')]:
    m = metrics[key]
    print(f"{name:<38} | {np.mean(m['mrr']):.4f} | {np.mean(m['ndcg3']):.4f} | {np.mean(m['ndcg5']):.4f} | {np.mean(m['rec1'])*100:.1f}%  | {np.mean(m['rec3'])*100:.1f}%  | {np.mean(m['rec5'])*100:.1f}%")
print("="*110 + "\n")

print("📊 First Relevant Skill Rank Distribution Profile (MRM):")
ranks_mrm = metrics['mrm']['ranks']
for r in sorted(list(set(ranks_mrm))):
    print(f"   -> Rank {r}: {ranks_mrm.count(r)} occurrences")
