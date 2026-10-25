import os
import sqlite3
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import ndcg_score

# --- PATH CONFIGURATION ---
CSV_FILE_PATH = r"D:\PHD\TACS\Code 16 Oct 2026\real_benchmark_results.csv"
SQLITE_DB_FILE = r"D:\PHD\TACS\Data\gitskills.sqlite.db"
OUTPUT_DIR = r"D:\PHD\TACS\Code 16 Oct 2026"

print("📖 Loading aligned benchmark results from CSV...")
if not os.path.exists(CSV_FILE_PATH):
    raise FileNotFoundError(f"Missing CSV benchmark file at: {CSV_FILE_PATH}")

df_csv = pd.read_csv(CSV_FILE_PATH)
num_samples = len(df_csv)
print(f"   -> Found {num_samples} aligned tasks inside CSV.")

# --- SOTA DEFINITION OF FINE-GRAINED MICRO-TOPICS ---
fine_grained_topics = {
    'Python Testing (PyTest)': "content LIKE '%testing%' AND content LIKE '%pytest%'",
    'JS Testing (Jest/Mocha)': "content LIKE '%testing%' AND (content LIKE '%jest%' OR content LIKE '%mocha%')",
    'Docker Infrastructure':   "content LIKE '%deployment%' AND content LIKE '%docker%'",
    'K8s Orchestration':       "content LIKE '%deployment%' AND content LIKE '%kubernetes%'",
    'SQL Migration (Alembic)': "content LIKE '%database_management%' AND content LIKE '%alembic%'",
    'NoSQL Cache (Redis)':     "content LIKE '%database_management%' AND content LIKE '%redis%'",
    'Static Security Linting': "content LIKE '%security%' AND content LIKE '%bandit%'"
}

print("\n🔍 Extracting fine-grained architectural tool subsets (CSV Hashes Only)...")
conn = sqlite3.connect(SQLITE_DB_FILE)
csv_hashes = df_csv['file_sha'].astype(str).str.strip().str.lower().values.tolist()

# Initialize lookup dictionary mapping each hash to an array of active fine-grained topics
# Shape: length of topics dict
num_topics = len(fine_grained_topics)
topic_names_list = list(fine_grained_topics.keys())
hash_topic_lookup = {h: np.zeros(num_topics, dtype=int) for h in csv_hashes}

# Query specific text blocks matching the CSV keys using SQL chunking boundaries
chunk_size_sql = 999
for i in range(0, len(csv_hashes), chunk_size_sql):
    chunk_hashes = csv_hashes[i:i+chunk_size_sql]
    placeholders = ','.join(['?'] * len(chunk_hashes))
    
    query_fast = f"SELECT file_sha, content FROM artifacts WHERE file_sha IN ({placeholders})"
    df_chunk = pd.read_sql_query(query_fast, conn, params=chunk_hashes)
    df_chunk['content'] = df_chunk['content'].fillna('').astype(str)
    
    # Process targeted framework token intersection directly in RAM
    for _, row in df_chunk.iterrows():
        h_clean = str(row['file_sha']).strip().lower()
        content_text = row['content']
        
        if h_clean in hash_topic_lookup:
            # We apply the specific tool token constraints evaluated by the dictionary conditions
            for t_idx, (t_name, _) in enumerate(fine_grained_topics.items()):
                # Simulating SQL inner text matches cleanly in Python
                if t_name == 'Python Testing (PyTest)' and 'testing' in content_text and 'pytest' in content_text:
                    hash_topic_lookup[h_clean][t_idx] = 1
                elif t_name == 'JS Testing (Jest/Mocha)' and 'testing' in content_text and ('jest' in content_text or 'mocha' in content_text):
                    hash_topic_lookup[h_clean][t_idx] = 1
                elif t_name == 'Docker Infrastructure' and 'deployment' in content_text and 'docker' in content_text:
                    hash_topic_lookup[h_clean][t_idx] = 1
                elif t_name == 'K8s Orchestration' and 'deployment' in content_text and 'kubernetes' in content_text:
                    hash_topic_lookup[h_clean][t_idx] = 1
                elif t_name == 'SQL Migration (Alembic)' and 'database_management' in content_text and 'alembic' in content_text:
                    hash_topic_lookup[h_clean][t_idx] = 1
                elif t_name == 'NoSQL Cache (Redis)' and 'database_management' in content_text and 'redis' in content_text:
                    hash_topic_lookup[h_clean][t_idx] = 1
                elif t_name == 'Static Security Linting' and 'security' in content_text and 'bandit' in content_text:
                    hash_topic_lookup[h_clean][t_idx] = 1

conn.close()
print("   -> Fine-grained database text indexing complete.")

# --- GROUPING ARTIFACTS BY EXPLICIT ARCHITECTURAL TOOL BUCKETS ---
print("🔮 Grouping CSV records into fine-grained tool subsets...")
topic_buckets = {topic: [] for topic in topic_names_list}

for _, row in df_csv.iterrows():
    h = str(row['file_sha']).strip().lower()
    if h in hash_topic_lookup:
        topic_flags = hash_topic_lookup[h]
        for t_idx, t_name in enumerate(topic_names_list):
            if topic_flags[t_idx] == 1:
                topic_buckets[t_name].append(row.to_dict())

# --- CHUNKING SUBSETS INTO LOCAL CANDIDATE RECOMMENDATION SETS (N=15) ---
POOL_SIZE = 15
context_pools = []

for t_name, records in topic_buckets.items():
    if len(records) == 0:
        continue
    df_sub_topic = pd.DataFrame(records)
    
    # Sort by token footprint to map development stages within the micro-tool stack
    df_sub_topic = df_sub_topic.sort_values(by='estimated_tokens').reset_index(drop=True)
    
    # Define specialized ground truth targets (Upper quartile of token weight within this framework)
    topic_threshold = df_sub_topic['estimated_tokens'].quantile(0.75)
    df_sub_topic['is_relevance_ground_truth'] = (df_sub_topic['estimated_tokens'] >= topic_threshold).astype(int)
    
    # Slice into context candidate pools of size N=15
    for i in range(0, len(df_sub_topic), POOL_SIZE):
        chunk = df_sub_topic[i:i + POOL_SIZE]
        if len(chunk) >= 5:  # Ensure statistical significance for ranking stability
            context_pools.append(chunk)

print(f"   -> Engineered {len(context_pools)} isolated, FINE-GRAINED tool candidate pools.")

# --- RANK-AWARE INFORMATION RETRIEVAL ENGINES ---
def compute_mrr(relevance_array):
    idx = np.where(relevance_array == 1)[0]
    return 1.0 / (idx[0] + 1) if len(idx) > 0 else 0.0

def compute_recall_at_k(relevance_array, k):
    total_positives = np.sum(relevance_array)
    if total_positives == 0:
        return 1.0
    return np.sum(relevance_array[:k]) / total_positives

metrics_accumulator = {
    'mr':  {'mrr': [], 'ndcg3': [], 'ndcg5': [], 'rec1': [], 'rec3': [], 'rec5': []},
    'mrm': {'mrr': [], 'ndcg3': [], 'ndcg5': [], 'rec1': [], 'rec3': [], 'rec5': []},
    'pr':  {'mrr': [], 'ndcg3': [], 'ndcg5': [], 'rec1': [], 'rec3': [], 'rec5': []},
    'ml':  {'mrr': [], 'ndcg3': [], 'ndcg5': [], 'rec1': [], 'rec3': [], 'rec5': []}
}

for pool in context_pools:
    y_true = pool['is_relevance_ground_truth'].values
    if np.sum(y_true) == 0: 
        continue
        
    for key, score_col in [('mr', 'specificity_mr'), ('mrm', 'influence_mrm'), ('pr', 'real_pagerank'), ('ml', 'ml_prob_score')]:
        scores = pool[score_col].values
        
        # Sort current context pool descending based on algorithm rank scores
        rank_idx = np.argsort(scores)[::-1]
        y_true_ranked = y_true[rank_idx]
        
        metrics_accumulator[key]['mrr'].append(compute_mrr(y_true_ranked))
        metrics_accumulator[key]['rec1'].append(compute_recall_at_k(y_true_ranked, 1))
        metrics_accumulator[key]['rec3'].append(compute_recall_at_k(y_true_ranked, 3))
        metrics_accumulator[key]['rec5'].append(compute_recall_at_k(y_true_ranked, 5))
        
        ndcg_true = y_true.reshape(1, -1)
        ndcg_scores = scores.reshape(1, -1)
        metrics_accumulator[key]['ndcg3'].append(ndcg_score(ndcg_true, ndcg_scores, k=3))
        metrics_accumulator[key]['ndcg5'].append(ndcg_score(ndcg_true, ndcg_scores, k=5))

# --- OUTPUT FINAL SCIENTIFIC TABLES ---
print("\n" + "="*95)
print("📊 Dr. Mkaouer's FINE-GRAINED EVALUATION BENCHMARK (Retrieve & Re-rank Framework)")
print("="*95)
print(f"{'Method/Baseline':<25} | {'MRR':<8} | {'nDCG@3':<8} | {'nDCG@5':<8} | {'Recall@1':<8} | {'Recall@3':<8} | {'Recall@5':<8}")
print("-"*95)

for key, name in [('mr', 'MarsRank (MR)'), ('mrm', 'MarsRank (MRM)'), ('ml', 'Machine Learning'), ('pr', 'PageRank (PR)')]:
    m = metrics_accumulator[key]
    print(f"{name:<25} | {np.mean(m['mrr']):.4f} | {np.mean(m['ndcg3']):.4f} | {np.mean(m['ndcg5']):.4f} | {np.mean(m['rec1'])*100:.1f}%  | {np.mean(m['rec3'])*100:.1f}%  | {np.mean(m['rec5'])*100:.1f}%")
print("="*95 + "\n")

# --- FILE EXPORT TRACK ---
report_file = os.path.join(OUTPUT_DIR, "rapport_fine_granularity_mkaouer.txt")
with open(report_file, "w", encoding="utf-8") as f:
    f.write("==================================================================\n")
    f.write("📝 EXPERIMENT 2: FINE-GRAINED TOOL SKILL RECOMMENDATION BENCHMARK\n")
    f.write("==================================================================\n\n")
    f.write(f"Candidate Set Pool Width : N = {POOL_SIZE} structural components\n")
    f.write(f"Total Evaluated Contexts : {len(context_pools)} framework-driven blocks\n\n")
    f.write(f"{'Method':<20} | {'MRR':<6} | {'nDCG@3':<7} | {'nDCG@5':<7} | {'Rec@1':<7} | {'Rec@3':<7} | {'Rec@5':<7}\n")
    f.write("-"*75 + "\n")
    for key, name in [('mr', 'MarsRank (MR)'), ('mrm', 'MarsRank (MRM)'), ('ml', 'Machine Learning'), ('pr', 'PageRank (PR)')]:
        m = metrics_accumulator[key]
        f.write(f"{name:<20} | {np.mean(m['mrr']):.3f} | {np.mean(m['ndcg3']):.3f} | {np.mean(m['ndcg5']):.3f} | {np.mean(m['rec1'])*100:.1f}% | {np.mean(m['rec3'])*100:.1f}% | {np.mean(m['rec5'])*100:.1f}%\n")
    f.write("==================================================================\n")

print(f"💾 Fine-grained verification report written cleanly to:\n   -> {report_file}")
