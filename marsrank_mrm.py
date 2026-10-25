import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres

def compute_marsrank_mrm(matrix_A, task_indices, gateway_indices, vs_profiles, c=0.85, epsilon=0.05, max_iter=20):
    n_tasks = len(task_indices)
    n_gateways = len(gateway_indices)
    n_capacities = vs_profiles.shape[0]
    
    print(f"Solver input: {n_tasks} tasks, {n_gateways} gateways.")
    
    # 1. Normalisation stochastique par ligne de la matrice creuse
    print("Normalizing sparse matrix...")
    row_sums = np.array(matrix_A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    
    # Création de la matrice de transition M = D^-1 * A
    inv_row_sums = 1.0 / row_sums
    D_inv = sp.diags(inv_row_sums)
    M = D_inv.dot(matrix_A) # Reste au format creux
    
    # 2. Extraction des sous-matrices via indices ordonnés
    all_indices = task_indices + gateway_indices
    M = M[all_indices, :][:, all_indices].tocsr()
    
    H_TT = M[0:n_tasks, 0:n_tasks]
    H_TG = M[0:n_tasks, n_tasks:]
    H_GT = M[n_tasks:, 0:n_tasks]
    H_GG = M[n_tasks:, n_tasks:]
    
    # 3. Approximation itérative de la réduction Schur S = H_TT + H_TG @ (I - H_GG)^-1 @ H_GT
    print("Computing Schur complement via iterative approximation...")
    # Au lieu d'inverser (I - H_GG), on calcule une approximation par itération
    I_GG = sp.eye(n_gateways, format='csr')
    
    # Version itérative rapide pour éviter LinAlgError et économiser la RAM
    X = H_GT.copy()
    S_aux = H_GT.copy()
    for _ in range(5): # Quelques itérations suffisent pour converger vers l'inversion
        X = H_GG.dot(X)
        S_aux += X
    S = H_TT + H_TG.dot(S_aux)
    
    # 4. Gestion des nœuds sans issue (Dangling nodes) de H_bar
    print("Handling dangling nodes...")
    S_row_sums = np.array(S.sum(axis=1)).flatten()
    dangling_mask = (S_row_sums == 0)
    
    # 5. Calcul des profils TSPR (Itérations de puissance au lieu d'inverser I - c*H_bar)
    print("Computing TSPR matrix profiles via Power Iteration...")
    tspr_matrix = np.zeros((n_capacities, n_tasks))
    
    for s in range(n_capacities):
        v_s = vs_profiles[s, :].copy()
        # Itérations de puissance pour converger vers (1 - c) * v_s @ inv(I_TT - c * H_bar)
        x = v_s.copy()
        for i in range(max_iter):
            # x_next = c * x @ H_bar + (1 - c) * v_s
            # x @ H_bar décomposé pour préserver la structure creuse :
            x_next = c * S.T.dot(x)
            
            # Correction pour les dangling nodes (d @ e0) et le vecteur de saut global
            dangling_sum = x[dangling_mask].sum()
            x_next += (c * dangling_sum / n_tasks) + (1 - c) * v_s
            x = x_next
            
        tspr_matrix[s, :] = x

    # 6. Calcul de la spécificité (MR)
    print("Calculating Specificity (MR)...")
    tspr_mean = tspr_matrix.mean(axis=0)
    MR = np.zeros(n_tasks)
    for x in range(n_tasks):
        MR[x] = np.mean((tspr_matrix[:, x] - tspr_mean[x]) ** 2)

    sum_mr = MR.sum()
    m = MR / sum_mr if sum_mr != 0 else np.ones(n_tasks) / n_tasks
    
    # 7. Calcul final de x_star par itération de puissance
    print("Computing final equilibrium x_star...")
    x_star = m.copy()
    for _ in range(max_iter):
        x_star_next = c * S.T.dot(x_star)
        dangling_sum = x_star[dangling_mask].sum()
        x_star_next += (c * dangling_sum / n_tasks) + (1 - c - epsilon) * m + epsilon * (1.0 / n_tasks)
        x_star = x_star_next
        
    return x_star, MR
