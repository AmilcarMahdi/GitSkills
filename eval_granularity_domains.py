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

# --- HIGH-SPEED TARGETED METADATA EXTRACTION ---
print("\n🔍 Connecting to SQLite to fetch targeted Topic Profiles (CSV Hashes Only)...")
conn = sqlite3.connect(SQLITE_DB_FILE)

csv_hashes = df_csv['file_sha'].astype(str).str.strip().str.lower().values.tolist()
target_capability_domains = ['testing', 'deployment', 'database_management', 'security']

# Initializing in-memory lookup dictionary for matching topics
# Key: file_sha -> Value: list of active binary domain flags [testing, deployment, db, security]
hash_topic_lookup = {h: np.zeros(len(target_capability_domains), dtype=int) for h in csv_hashes}

# Query only for rows matching the CSV keys using SQL chunking boundaries
chunk_size_sql = 999
for i in range(0, len(csv_hashes), chunk_size_sql):
    chunk_hashes = csv_hashes[i:i+chunk_size_sql]
    placeholders = ','.join(['?'] * len(chunk_hashes))
    
    # Extract only file_sha and raw content columns for the targeted slice
    query_fast = f"SELECT file_sha, content FROM artifacts WHERE file_sha IN ({placeholders})"
    df_chunk = pd.read_sql_query(query_fast, conn, params=chunk_hashes)
    
    # FIX: Handle database Null values inside Pandas before entering string matching
    df_chunk['content'] = df_chunk['content'].fillna('').astype(str)
    
    # Process text matching immediately in RAM
    for _, row in df_chunk.iterrows():
        h_clean = str(row['file_sha']).strip().lower()
        content_text = row['content']
        
        if h_clean in hash_topic_lookup:
            for domain_idx, domain_name in enumerate(target_capability_domains):
                if domain_name in content_text:
                    hash_topic_lookup[h_clean][domain_idx] = 1

conn.close()
print("   -> Targeted database text indexing complete.")

# --- GROUPING ARTIFACTS BY EXPLICIT FUNCTIONAL DOMAINS ---
print("🔮 Grouping CSV records into explicit Domain Pools...")
topic_buckets = {domain: [] for domain in target_capability_domains}

for _, row in df_csv.iterrows():
    h = str(row['file_sha']).strip().lower()
    if h in hash_topic_lookup:
        domain_flags = hash_topic_lookup[h]
        for domain_idx, domain_name in enumerate(target_capability_domains):
            if domain_flags[domain_idx] == 1:
                topic_buckets[domain_name].append(row.to_dict())

# --- CHUNKING DOMAINS INTO REALISTIC CANDIDATE POOLS (N=15) ---
POOL_SIZE = 15
context_pools = []

for domain_name, records in topic_buckets.items():
    if len(records) == 0:
        continue
    df_topic = pd.DataFrame(records)
    
    # Within each domain library, sort tasks by token consumption to identify specialists
    df_topic = df_topic.sort_values(by='estimated_tokens').reset_index(drop=True)
    
    # Define specialized ground truth targets (Upper quartile of token weight inside the domain)
    topic_threshold = df_topic['estimated_tokens'].quantile(0.75)
    df_topic['is_relevance_ground_truth'] = (df_topic['estimated_tokens'] >= topic_threshold).astype(int)
    
    # Slice this specific domain space into isolated context candidate pools
    for i in range(0, len(df_topic), POOL_SIZE):
        chunk = df_topic[i:i + POOL_SIZE]
        if len(chunk) >= 5:  # Secure minimum sample height for rank verification stability
            context_pools.append(chunk)

print(f"   -> Engineered {len(context_pools)} isolated, DOMAIN-DRIVEN candidate pools.")

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
print("\n" + "="*90)
print("📊 Dr. Mkaouer's TOPIC-ISOLATED EVALUATION BENCHMARK (Retrieve & Re-rank Framework)")
print("="*90)
print(f"{'Method/Baseline':<25} | {'MRR':<8} | {'nDCG@3':<8} | {'nDCG@5':<8} | {'Recall@1':<8} | {'Recall@3':<8} | {'Recall@5':<8}")
print("-"*90)

for key, name in [('mr', 'MarsRank (MR)'), ('mrm', 'MarsRank (MRM)'), ('ml', 'Machine Learning'), ('pr', 'PageRank (PR)')]:
    m = metrics_accumulator[key]
    print(f"{name:<25} | {np.mean(m['mrr']):.4f} | {np.mean(m['ndcg3']):.4f} | {np.mean(m['ndcg5']):.4f} | {np.mean(m['rec1'])*100:.1f}%  | {np.mean(m['rec3'])*100:.1f}%  | {np.mean(m['rec5'])*100:.1f}%")
print("="*90 + "\n")

# --- FILE EXPORT TRACK ---
report_file = os.path.join(OUTPUT_DIR, "rapport_topic_aware_mkaouer.txt")
with open(report_file, "w", encoding="utf-8") as f:
    f.write("==================================================================\n")
    f.write("📝 EXPERIMENT 2: DOMAIN-ISOLATED SKILL RECOMMENDATION BENCHMARK\n")
    f.write("==================================================================\n\n")
    f.write(f"Candidate Set Pool Width : N = {POOL_SIZE} structural components\n")
    f.write(f"Total Evaluated Contexts : {len(context_pools)} domain-driven blocks\n\n")
    f.write(f"{'Method':<20} | {'MRR':<6} | {'nDCG@3':<7} | {'nDCG@5':<7} | {'Rec@1':<7} | {'Rec@3':<7} | {'Rec@5':<7}\n")
    f.write("-"*75 + "\n")
    for key, name in [('mr', 'MarsRank (MR)'), ('mrm', 'MarsRank (MRM)'), ('ml', 'Machine Learning'), ('pr', 'PageRank (PR)')]:
        m = metrics_accumulator[key]
        f.write(f"{name:<20} | {np.mean(m['mrr']):.3f} | {np.mean(m['ndcg3']):.3f} | {np.mean(m['ndcg5']):.3f} | {np.mean(m['rec1'])*100:.1f}% | {np.mean(m['rec3'])*100:.1f}% | {np.mean(m['rec5'])*100:.1f}%\n")
    f.write("==================================================================\n")

print(f"💾 Domain-isolated validation report written cleanly to:\n   -> {report_file}")
