import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from .density_estimation import *

def spectral_dimension(manifold_type, M):
    if manifold_type == 'S1': return 2*M + 1
    elif manifold_type == 'S2':return (M + 1)**2
    elif manifold_type == 'T2':return (2*M + 1)**2
    elif manifold_type == 'SO3': np.sum((2*np.arange(M+1) + 1)**2)
    else:raise ValueError(f"Unknown manifold type: {manifold_type}")
    

def select_M_rho_by_scoreMatchingKFoldCV(manifold_type, X, M_grid, rho_grid,
                                         n_splits=5,
                                         return_scores=False,
                                         random_state=None,
                                         display_tqdm = True):
    """
    Select truncation level M and density lower bound rho by K-fold CV using Hyvärinen score matching.
    """
    M_grid = np.array(M_grid)
    rho_grid = np.array(rho_grid)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Matrix to store scores (M rows, rho columns)
    cv_scores = np.zeros((len(M_grid), len(rho_grid)), dtype=float)

    if display_tqdm:
        fold_iter = tqdm(kf.split(X), total=n_splits, desc="Folds")
    else:
        fold_iter = kf.split(X)
    
    for train_idx, val_idx in fold_iter:
        X_train, X_val = X[train_idx], X[val_idx]

        for ixM, M in enumerate(M_grid):
            _, base_f, grad_f, lap_f = density_estimate(manifold_type, X_train, M, X_val, grad=True, laplacian=True)

            grad_sq = np.sum(grad_f**2, axis=1)

            for ixRho, rho in enumerate(rho_grid):
                if rho < base_f.mean():
                    hat_f = np.maximum(base_f, rho)
                    
                    # Hyvärinen Score Matching terms:
                    # grad_log_sq = ||grad f||^2 / f^2
                    # lap_log     = (lap f / f) - (||grad f||^2 / f^2)
                    # Score = grad_log_sq - 2 * lap_log
                    eps = 1e-8 
                    score_vals = (2 * lap_f / (hat_f + eps)) - (grad_sq / (hat_f**2 + eps))
                    # Optional: Clip score_vals to prevent outliers from ruining the fold mean
                    score_vals = np.clip(score_vals, -1e6, 1e6)
                    try: cv_scores[ixM, ixRho] += np.mean(score_vals) 
                    except: cv_scores[ixM, ixRho] = np.nan 
                else:
                    cv_scores[ixM, ixRho] = np.nan

    # Average over folds
    cv_scores /= n_splits

    # Info Criteria
    n = len(X)
    k_vals = np.array([spectral_dimension(manifold_type, M) for M in M_grid])
    k_penalty = k_vals[:, np.newaxis]
    AIC_scores = cv_scores + 2 * k_penalty / n
    BIC_scores = cv_scores + np.log(n) * k_penalty / n

    def get_best_params(score_matrix):
        """Helper to find the (M, rho) pair corresponding to the minimum score."""
        masked = np.where(np.isfinite(score_matrix), score_matrix, np.inf)
        idx_M, idx_Rho = np.unravel_index(np.argmin(masked), masked.shape)
        return (M_grid[idx_M], rho_grid[idx_Rho])

    Mrhostar = {
        'cv':  get_best_params(cv_scores),
        'AIC': get_best_params(AIC_scores),
        'BIC': get_best_params(BIC_scores),
    }

    if return_scores:
        return Mrhostar, cv_scores
    else:
        return Mrhostar
    