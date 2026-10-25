import sqlite3
import numpy as np
import pandas as pd
from marsrank_mrm import compute_marsrank_mrm  # Import your theoretical engine

def extract_gitskills_matrices(db_path, target_capability_domains=None):
    """
    Connects to the GitSkills SQLite database, reconstructs the sequential 
    execution traces, maps the Capability Domain profiles, and isolates tasks vs gateways.
    """
    print(f"Connecting to GitSkills Database at: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # ------------------------------------------------------------------------
    # 1. RECONSTRUCT THE SEQUENCE FLOW (Liaisons et Transitions)
    # ------------------------------------------------------------------------
    print("Extracting sequential execution traces from artifact companion logs...")
    # Query tracking consecutive tool/skill calling sequences within the same repository run
    query_flows = """
        SELECT s1.content_hash AS source_task, s2.content_hash AS target_task, COUNT(*) as frequency
        FROM artifact_siblings s1
        JOIN artifact_siblings s2 ON s1.repo_id = s2.repo_id AND s2.sequence_order = s1.sequence_order + 1
        GROUP BY s1.content_hash, s2.content_hash
    """
    df_flows = pd.read_sql_query(query_flows, conn)
    
    # Get all unique nodes in the active dataset population
    all_nodes = list(set(df_flows['source_task'].unique()).union(set(df_flows['target_task'].unique())))
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    N = len(all_nodes)
    
    # Build the weighted global adjacency matrix A
    print(f"Building Global Adjacency Matrix of size {N}x{N}...")
    matrix_A = np.zeros((N, N))
    for _, row in df_flows.iterrows():
        i = node_to_idx[row['source_task']]
        j = node_to_idx[row['target_task']]
        matrix_A[i, j] = float(row['frequency'])
        
    # ------------------------------------------------------------------------
    # 2. ISOLATE OPERATIONAL TASKS (T) FROM SYSTEM GATEWAYS (G)
    # ------------------------------------------------------------------------
    print("Classifying node states based on front-matter metadata tags...")
    query_meta = "SELECT content_hash, parsed_front_matter FROM artifacts WHERE parsed_front_matter IS NOT NULL"
    df_meta = pd.read_sql_query(query_meta, conn)
    
    task_indices = []
    gateway_indices = []
    
    # Default categorization maps non-sémantico structural nodes as transparent gateways
    for idx, node_hash in idx_to_node.items():
        meta_row = df_meta[df_meta['content_hash'] == node_hash]
        if not meta_row.empty and "type: gateway" in str(meta_row['parsed_front_matter'].values[0]):
            gateway_indices.append(idx)
        else:
            task_indices.append(idx)
            
    # ------------------------------------------------------------------------
    # 3. BUILD THE SEMANTIC BIAS MATRIX (vs_profiles)
    # ------------------------------------------------------------------------
    if target_capability_domains is None:
        target_capability_domains = ['testing', 'deployment', 'database_management', 'security']
        
    n_domains = len(target_capability_domains)
    print(f"Extracting {n_domains} sémantico Capability Domains for vs_profiles mapping...")
    vs_profiles = np.zeros((n_domains, len(task_indices)))
    
    # Map index layout specifically for the active tasks space
    task_hash_to_sub_idx = {idx_to_node[global_idx]: sub_idx for sub_idx, global_idx in enumerate(task_indices)}
    
    for domain_idx, domain_name in enumerate(target_capability_domains):
        # Count frequency of domain capability assignments parsed in front-matter text fields
        for _, row in df_meta.iterrows():
            node_hash = row['content_hash']
            if node_hash in task_hash_to_sub_idx:
                sub_idx = task_hash_to_sub_idx[node_hash]
                if domain_name in str(row['parsed_front_matter']).lower():
                    vs_profiles[domain_idx, sub_idx] += 1.0
                    
        # Apply strict rows L1 simplex normalization requirement (vs * 1^T = 1)
        row_sum = vs_profiles[domain_idx, :].sum()
        if row_sum > 0:
            vs_profiles[domain_idx, :] /= row_sum
        else:
            vs_profiles[domain_idx, :] = 1.0 / len(task_indices)
            
    conn.close()
    print("Data structures successfully synchronized from database.")
    return matrix_A, task_indices, gateway_indices, vs_profiles, idx_to_node

# ============================================================================
# MAIN PIPELINE DEPLOYMENT
# ============================================================================
if __name__ == "__main__":
    # Path pointer targeting your offline local copy of the MSR database file
    SQLITE_DB_FILE = "D:\\PHD\\TACS\\Data\\gitskills.sqlite"
    
    try:
        # 1. Pull data directly through the schemas
        A, tasks, gateways, vs, mapping = extract_gitskills_matrices(SQLITE_DB_FILE)
        
        # 2. Pipe variable structural blocks into your tested MarsRank + MRM engine
        x_star, MR = compute_marsrank_mrm(A, tasks, gateways, vs)
        
        # 3. Output structural results for high-ranked items
        print("\n--- TOP RANKED SPECIFIC AGENTIC SKILLS (GitSkills Core) ---")
        top_indices = np.argsort(MR)[::-1][:5] # Sort descending and pull top 5 profiles
        for rank, sub_idx in enumerate(top_indices):
            global_idx = tasks[sub_idx]
            node_hash = mapping[global_idx]
            print(f"Rank {rank+1} -> Hash: {node_hash} | Specificity (MR) = {MR[sub_idx]:.5f} | MRM Routing Vector = {x_star[sub_idx]:.5f}")
            
    except FileNotFoundError:
        print(f"\n[!] Missing Database: To execute this live, please verify that your SQLite file is saved at: {SQLITE_DB_FILE}")
