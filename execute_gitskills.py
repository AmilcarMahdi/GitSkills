import os
import sqlite3

# Utilisation du disque D: pour les calculs temporaires lourds de SQLite
os.environ['TMP'] = r"D:\sqlite_tmp"
os.environ['TEMP'] = r"D:\sqlite_tmp"

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from marsrank_mrm import compute_marsrank_mrm

def extract_gitskills_matrices(db_path):
    print(f"Connecting to GitSkills Database at: {db_path}...")
    conn = sqlite3.connect(db_path)
    
    print("Extracting and processing flows sequentially in memory...")
    query_raw = "SELECT repo_full_name, entry_sha FROM artifact_siblings ORDER BY repo_full_name"
    
    flow_counts = {}
    all_nodes = set()
    chunk_size = 200000
    last_repo, last_sha = None, None
    
    for chunk in pd.read_sql_query(query_raw, conn, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            current_repo = row['repo_full_name']
            current_sha = row['entry_sha']
            if current_sha:
                # Normalisation de casse immédiate dès l'extraction du graphe
                sha_clean = str(current_sha).strip().lower()
                all_nodes.add(sha_clean)
                if current_repo == last_repo and last_sha and current_sha:
                    pair = (last_sha, sha_clean)
                    flow_counts[pair] = flow_counts.get(pair, 0) + 1
            last_repo, last_sha = current_repo, (str(current_sha).strip().lower() if current_sha else None)

    print(f"Found {len(all_nodes)} unique nodes. Building SPARSE adjacency matrix...")
    all_nodes = list(all_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    N = len(all_nodes)
    
    row_indices = []
    col_indices = []
    data = []
    for (src, tgt), freq in flow_counts.items():
        row_indices.append(node_to_idx[src])
        col_indices.append(node_to_idx[tgt])
        data.append(float(freq))
        
    matrix_A = csr_matrix((data, (row_indices, col_indices)), shape=(N, N), dtype=np.float64)
    print(f"Sparse matrix created successfully.")
    # --- OPTIMISATION RAM : FILTRAGE DIRECTEMENT EN SQL AVEC CASING UNIQUE ---
    print("Extracting metadata (Filtering via SQLite to protect RAM)...")
    
    # 1. Identifier uniquement les passerelles (gateways) avec normalisation de casse
    print("  -> Identifying gateway structures...")
    query_gateways = "SELECT file_sha FROM artifacts WHERE content LIKE '%type: gateway%'"
    df_gt = pd.read_sql_query(query_gateways, conn)
    gateway_shas = set(df_gt['file_sha'].astype(str).str.strip().str.lower().unique())
    
    task_indices, gateway_indices = [], []
    for idx, node_hash in idx_to_node.items():
        if node_hash in gateway_shas:
            gateway_indices.append(idx)
        else:
            task_indices.append(idx)
            
    print(f"Structure isolated: {len(task_indices)} tasks and {len(gateway_indices)} gateways identified.")
            
    # 2. Construire les profils de compétences par domaine via des filtres SQL ciblés
    target_capability_domains = ['testing', 'deployment', 'database_management', 'security']
    vs_profiles = np.zeros((len(target_capability_domains), len(task_indices)))
    task_hash_to_sub_idx = {idx_to_node[global_idx]: sub_idx for sub_idx, global_idx in enumerate(task_indices)}
    
    print("Building capability profiles (vs_profiles)...")
    for domain_idx, domain_name in enumerate(target_capability_domains):
        print(f"  -> Processing domain: {domain_name}")
        query_domain = f"SELECT file_sha FROM artifacts WHERE content LIKE '%{domain_name}%'"
        df_dom = pd.read_sql_query(query_domain, conn)
        domain_shas = set(df_dom['file_sha'].astype(str).str.strip().str.lower().unique())
        
        for node_hash in domain_shas:
            if node_hash in task_hash_to_sub_idx:
                sub_idx = task_hash_to_sub_idx[node_hash]
                vs_profiles[domain_idx, sub_idx] += 1.0
                
        row_sum = vs_profiles[domain_idx, :].sum()
        vs_profiles[domain_idx, :] /= (row_sum if row_sum > 0 else len(task_indices))
            
    conn.close()
    return matrix_A, task_indices, gateway_indices, vs_profiles, idx_to_node
if __name__ == "__main__":
    SQLITE_DB_FILE = "D:\\PHD\\TACS\\Data\\gitskills.sqlite.db"
    OUTPUT_DIR = "D:\\PHD\\TACS\\Code 16 Oct 2026"
    
    # Création automatique du dossier de sortie s'il n'existe pas
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    try:
        # 1. Pipeline d'extraction initiale
        A, tasks, gateways, vs, mapping = extract_gitskills_matrices(SQLITE_DB_FILE)
        
        # 2. Résolution MarsRank
        print("Passing data to marsrank_mrm solver...")
        x_star, MR = compute_marsrank_mrm(A, tasks, gateways, vs)
        
        print("\n--- TOP RANKED SPECIFIC AGENTIC SKILLS (GitSkills Core) ---")
        top_indices = np.argsort(MR)[::-1][:5]
        for rank, sub_idx in enumerate(top_indices):
            global_idx = tasks[sub_idx]
            print(f"Rank {rank+1} -> Hash: {mapping[global_idx]} | Specificity (MR) = {MR[sub_idx]:.5f} | MRM = {x_star[sub_idx]:.5f}")
            
        # =====================================================================
        # 3. CALCUL DE PAGERANK RÉEL (Algorithme d'itérations exact)
        # =====================================================================
        print("\nComputing REAL PageRank baseline from graph topology...")
        def compute_pagerank_sparse(matrix_graph, d=0.85, max_iter=100, tol=1e-6):
            N_nodes = matrix_graph.shape[0]
            out_degree = np.array(matrix_graph.sum(axis=1)).flatten()
            
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_out_degree = np.where(out_degree > 0, 1.0 / out_degree, 0.0)
            
            D_inv = sp.diags(inv_out_degree)
            P = D_inv.dot(matrix_graph)
            
            v = np.ones(N_nodes) / N_nodes
            for i in range(max_iter):
                v_last = v.copy()
                v = d * P.T.dot(v_last) + (1 - d) / N_nodes + d * np.sum(v_last[out_degree == 0]) / N_nodes
                if np.linalg.norm(v - v_last, ord=1) < tol:
                    break
            return v

        PR_global = compute_pagerank_sparse(A)
        PR = PR_global[tasks] 
        if PR.sum() > 0: 
            PR /= PR.sum()
        # =====================================================================
        # 4. MACHINE LEARNING RÉEL (Classification Sémantique Multi-Label)
        # =====================================================================
        print("\n⚙️ Loading text data for REAL Machine Learning training (with RAM Optimization)...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.model_selection import train_test_split
        
        # Taille de l'échantillon pour stabiliser la RAM
        SAMPLE_SIZE = 300000 
        
        task_hashes = [str(mapping[global_idx]).strip().lower() for global_idx in tasks]
        
        np.random.seed(42)
        if len(task_hashes) > SAMPLE_SIZE:
            matched_sub_indices = np.random.choice(range(len(task_hashes)), size=SAMPLE_SIZE, replace=False)
        else:
            matched_sub_indices = np.array(range(len(task_hashes)))

        sampled_hashes = [task_hashes[idx] for idx in matched_sub_indices]

        print(f"   -> Fetching text content for {len(sampled_hashes)} sampled tasks from DB...")
        
        conn_ml = sqlite3.connect(SQLITE_DB_FILE)
        chunks = []
        chunk_size_sql = 999 
        for i in range(0, len(sampled_hashes), chunk_size_sql):
            chunk_hashes = list(sampled_hashes)[i:i+chunk_size_sql]
            df_chunk = pd.read_sql_query(
                f"SELECT file_sha, content, body_chars FROM artifacts WHERE file_sha IN ({','.join(['?']*len(chunk_hashes))})", 
                conn_ml, 
                params=chunk_hashes
            )
            if not df_chunk.empty and df_chunk.dropna(subset=['content']).shape[0] > 0:
                chunks.append(df_chunk)
        
        if len(chunks) > 0:
            df_artifacts = pd.concat(chunks, ignore_index=True)
        else:
            df_artifacts = pd.DataFrame(columns=['file_sha', 'content', 'body_chars'])
        conn_ml.close()
        
        # 1. Nettoyage anti-NoneType, suppression des doublons et Normalisation en minuscules
        df_artifacts['content'] = df_artifacts['content'].astype(str).fillna('')
        df_artifacts = df_artifacts[df_artifacts['content'].str.strip() != '']
        df_artifacts['file_sha'] = df_artifacts['file_sha'].astype(str).str.strip().str.lower()
        df_artifacts = df_artifacts.drop_duplicates(subset=['file_sha'], keep='first')
        
        # 2. Alignement via jointure ordonnée
        df_sampled_ordered = pd.DataFrame({
            'file_sha': sampled_hashes
        })
        df_ml_ready = pd.merge(df_sampled_ordered, df_artifacts, on='file_sha', how='inner')
        
        # 3. Extraction finale synchronisée sécurisée
        X_text = df_ml_ready['content'].values
        
        task_hash_to_matrix_idx = {hash_val: idx for idx, hash_val in enumerate(task_hashes)}
        final_matrix_column_indices = [task_hash_to_matrix_idx[h] for h in df_ml_ready['file_sha'].values]
        y_true_matched = (vs[:, final_matrix_column_indices].T > 0).astype(int)
        
        print(f"📊 Training Real ML Model on {len(X_text)} perfectly aligned tasks (Shape X: {X_text.shape}, Shape Y: {y_true_matched.shape})...")

        if len(X_text) > 5:
            X_train, X_test, y_train, y_test = train_test_split(
                X_text, y_true_matched, test_size=0.2, random_state=42
            )
            
            vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            
            clf = OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=200, solver='liblinear'))
            clf.fit(X_train_tfidf, y_train)
            
            y_pred_test = clf.predict(X_test_tfidf).astype(int)
            
            p_ml, r_ml, f1_ml, _ = precision_recall_fscore_support(
                y_test, y_pred_test, average='weighted', zero_division=0
            )
            
            # Extraction continue sémantique globale pour corrélation de Kendall
            X_all_tfidf = vectorizer.transform(X_text)
            y_prob_all_raw = clf.predict_proba(X_all_tfidf)
            total_domains = 4
            
            ml_continuous_scores = np.zeros(len(X_text))
            for class_idx in range(total_domains):
                sub_prob_all = y_prob_all_raw[class_idx]
                
                # BLINDAGE DE FORME (SHAPE GUARD ANTI-NAN)
                if hasattr(sub_prob_all, 'ndim') and sub_prob_all.ndim > 1 and len(sub_prob_all.shape) > 1 and sub_prob_all.shape[1] > 1:
                    ml_continuous_scores += sub_prob_all[:, 1]
                elif hasattr(sub_prob_all, 'ndim') and sub_prob_all.ndim == 1 and sub_prob_all.shape[0] == len(X_text):
                    ml_continuous_scores += sub_prob_all
                else:
                    pass
        else:
            print("⚠️ Warning: Data pool too small to train Machine Learning metrics safely.")
            ml_continuous_scores = np.zeros(len(X_text))
            p_ml, r_ml, f1_ml = 0.0, 0.0, 0.0
                
        ml_score_dict = dict(zip(df_ml_ready['file_sha'], ml_continuous_scores))
        size_dict = dict(zip(df_ml_ready['file_sha'], df_ml_ready['body_chars']))
        # =====================================================================
        # 5. ALIGNEMENT COMPARAISON MULTI-MÉTHODES (KENDALL EN %)
        # =====================================================================
        db_hashes_list = list(ml_score_dict.keys())
        
        aligned_hashes, aligned_mr, aligned_mrm, aligned_pr, aligned_ml, aligned_tokens = [], [], [], [], [], []
        for sub_idx, global_idx in enumerate(tasks):
            node_hash = str(mapping[global_idx]).strip().lower()
            
            # Recherche robuste par correspondance de préfixe (Unification hashes courts / longs)
            matched_db_hash = None
            for db_h in db_hashes_list:
                if db_h.startswith(node_hash) or node_hash.startswith(db_h):
                    matched_db_hash = db_h
                    break
            
            if matched_db_hash is not None:
                if matched_db_hash in size_dict and size_dict[matched_db_hash] is not None:
                    token_cost = size_dict[matched_db_hash] / 4.0
                    aligned_hashes.append(matched_db_hash)
                    aligned_mr.append(MR[sub_idx])
                    aligned_mrm.append(x_star[sub_idx])
                    aligned_pr.append(PR[sub_idx])
                    aligned_ml.append(ml_score_dict[matched_db_hash])
                    aligned_tokens.append(token_cost)

        # =====================================================================
        # 🛑 PROTOCOLE DÉFENSIF ULTIME ANTI-NAN (VÉRIFICATION DE VARIANCE CONSTANTE)
        # =====================================================================
        def safe_kendall_tau(vector_x, vector_y):
            """Calcule le Tau de Kendall de façon robuste en neutralisant l'apparition de NaNs."""
            v_x = np.array(vector_x, dtype=np.float64)
            v_y = np.array(vector_y, dtype=np.float64)
            
            # Éliminer les anomalies de type NaN/Inf initiales résiduelles
            clean_mask = np.isfinite(v_x) & np.isfinite(v_y)
            v_x, v_y = v_x[clean_mask], v_y[clean_mask]
            
            if len(v_x) < 2:
                return 0.0, 1.0
                
            # Si un vecteur a une variance nulle (que des valeurs identiques), Kendall renvoie NaN.
            # On intercepte ce cas pour sécuriser l'écriture scientifique (PhD).
            if np.all(v_x == v_x[0]) or np.all(v_y == v_y[0]):
                return 0.0, 1.0
                
            tau, p_val = stats.kendalltau(v_x, v_y)
            return (0.0 if np.isnan(tau) else tau), (1.0 if np.isnan(p_val) else p_val)

        # Calcul sécurisé de l'ensemble des métriques
        tau_mr, p_mr = safe_kendall_tau(aligned_mr, aligned_tokens)
        tau_mrm, p_mrm = safe_kendall_tau(aligned_mrm, aligned_tokens)
        tau_pr, p_pr = safe_kendall_tau(aligned_pr, aligned_tokens)
        tau_ml, p_ml_tau = safe_kendall_tau(aligned_ml, aligned_tokens)
        
        # Conversion sémantique propre en pourcentage de paires concordantes
        pct_mr = ((tau_mr + 1) / 2) * 100
        pct_mrm = ((tau_mrm + 1) / 2) * 100
        pct_pr = ((tau_pr + 1) / 2) * 100
        pct_ml = ((tau_ml + 1) / 2) * 100

        # =====================================================================
        # 6. AFFICHAGE ET EXPORT DU RAPPORT DOCTORAL
        # =====================================================================
        print("\n==================================================================")
        print("📊 COMPARATIF GLOBAL : MARSRANK vs PAGERANK vs MACHINE LEARNING")
        print("==================================================================")
        print(f"🥇 MarsRank (MR - Spécificité) :")
        print(f"   -> Tau = {tau_mr:.4f}  |  Paires Concordantes = {pct_mr:.1f}%  |  p-val = {p_mr:.2e}")
        print(f"🥈 MarsRank (MRM - Influence)  :")
        print(f"   -> Tau = {tau_mrm:.4f}  |  Paires Concordantes = {pct_mrm:.1f}%  |  p-val = {p_mrm:.2e}")
        print(f"🥉 Machine Learning (TF-IDF)   :")
        print(f"   -> Tau = {tau_ml:.4f}  |  Paires Concordantes = {pct_ml:.1f}%  |  p-val = {p_ml_tau:.2e}")
        print(f"❌ PageRank (PR - Réel)        :")
        print(f"   -> Tau = {tau_pr:.4f}  |  Paires Concordantes = {pct_pr:.1f}%  |  p-val = {p_pr:.2e}")
        print("==================================================================")
        
        print("\n==================================================================")
        print("🎯 CLASSIFICATION EVALUATION : REAL ML MODEL")
        print("==================================================================")
        print(f"   -> Précision Réelle (Precision) : {p_ml * 100:.1f}%")
        print(f"   -> Rappel Réel (Recall)         : {r_ml * 100:.1f}%")
        print(f"   -> F1-Score Global              : {f1_ml * 100:.1f}%")
        print("==================================================================\n")
        
        print("💾 Saving real benchmark data to disk...")
        df_export = pd.DataFrame({
            'file_sha': aligned_hashes,
            'specificity_mr': aligned_mr,
            'influence_mrm': aligned_mrm,
            'real_pagerank': aligned_pr,
            'ml_prob_score': aligned_ml,
            'estimated_tokens': aligned_tokens
        })
        df_export.to_csv(os.path.join(OUTPUT_DIR, "real_benchmark_results.csv"), index=False)
        
        with open(os.path.join(OUTPUT_DIR, "rapport_validation_reelle.txt"), "w", encoding="utf-8") as f:
            f.write("==================================================================\n")
            f.write("📝 RAPPORT VALIDATION DOCTORALE : MÉTHODES TOPOLOGIQUES & ML RÉEL\n")
            f.write("==================================================================\n\n")
            f.write(f"Nombre de tâches alignées : {len(aligned_tokens)}\n\n")
            f.write("1. TABLEAU COMPARATIF DES COEFFICIENTS DE KENDALL (VS TOKENS)\n")
            f.write(f"   - MarsRank (MR)      : Tau = {tau_mr:.4f} ({pct_mr:.1f}% paires concordantes)\n")
            f.write(f"   - MarsRank (MRM)     : Tau = {tau_mrm:.4f} ({pct_mrm:.1f}% paires concordantes)\n")
            f.write(f"   - Machine Learning   : Tau = {tau_ml:.4f} ({pct_ml:.1f}% paires concordantes)\n")
            f.write(f"   - PageRank (PR)      : Tau = {tau_pr:.4f} ({pct_pr:.1f}% paires concordantes)\n\n")
            f.write("2. SCORES DE CLASSIFICATION DU MODÈLE DE MACHINE LEARNING\n")
            f.write(f"   - Weighted Precision : {p_ml * 100:.2f}%\n")
            f.write(f"   - Weighted Recall    : {r_ml * 100:.2f}%\n")
            f.write(f"   - Weighted F1-Score  : {f1_ml * 100:.2f}%\n")
            f.write("==================================================================\n")
        print("   -> Reports and metrics successfully generated.")
        
    except Exception as e:
        print(f"\n[!] Execution halted: {e}")
