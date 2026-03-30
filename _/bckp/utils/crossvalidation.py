import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from .density_estimation import *
from .helpers import *


def get_scores(manifold_type, n, cv_scores, M_grid, rho_grid):
    def spectral_dimension(manifold_type, M):
        if manifold_type == 'S1':
            return 2*M + 1
        elif manifold_type == 'S2':
            return (M + 1)**2
        elif manifold_type == 'T2':
            return (2*M + 1)**2
        elif manifold_type == 'SO3':
            return np.sum((2*np.arange(M+1) + 1)**2)
        else:
            raise ValueError(f"Unknown manifold type: {manifold_type}")
        
    def get_best_params(score_matrix, M_grid, rho_grid):
        """Return the best (M, rho) pair; break ties by largest rho, smallest M."""
        masked = np.where(np.isfinite(score_matrix), score_matrix, np.inf)
        min_val = np.min(masked)
        if not np.isfinite(min_val):
            return (np.nan, np.nan)
        candidates = np.argwhere(masked == min_val)
        if rho_grid == None:
            best = max(candidates, key=lambda ix: -M_grid[ix[0]]) 
            ixM = int(best[0])
            return (M_grid[ixM], ixM)
        else:
            best = max(candidates, key=lambda ix: (rho_grid[ix[1]], -M_grid[ix[0]])) 
            ixM, ixRho = int(best[0]), int(best[1])
            return (M_grid[ixM], rho_grid[ixRho])

    k_vals = np.array([spectral_dimension(manifold_type, M) for M in M_grid])
    if rho_grid is None: k_penalty = k_vals
    else: k_penalty = k_vals[:, np.newaxis]
    AIC_scores = cv_scores + 2 * k_penalty / n
    BIC_scores = cv_scores + np.log(n) * k_penalty / n

    return (
        {"cv": cv_scores, "AIC": AIC_scores, "BIC": BIC_scores},
        {
            "cv": get_best_params(cv_scores, M_grid, rho_grid),
            "AIC": get_best_params(AIC_scores, M_grid, rho_grid),
            "BIC": get_best_params(BIC_scores, M_grid, rho_grid),
        },
    )



def scoreMatchingKFoldCV(manifold_type, X, M_grid, rho_grid = None,
                                         n_splits=5,
                                         return_scores=False,
                                         random_state=None,
                                         display_tqdm = True,
                                         eps = 1e-5
                                         ):
    
    """
    Select truncation level M and density lower bound rho by K-fold CV using Hyvärinen score matching.
     Score = ||grad f||^2 / f^2 - 2 * (lap f / f)
    ----
    Parameters:
    - manifold_type: str, type of manifold (e.g., 'S1', 'S2', 'T2', 'SO3').
    - X: array-like, shape (n_samples, n_features), input data.
    - M_grid: list or array of int, candidate truncation levels.
    - rho_grid: list or array of float, candidate density lower bounds. If None, a default heuristic will be used.
                (rho = 5th percentile of the estimated density values on the validation set).
    - n_splits: int, number of folds for K-fold CV.
    - return_scores: bool, if True, also return the score matrices for all (M, rho) pairs.
    - random_state: int or None, random seed for reproducibility.
    - display_tqdm: bool, if True, display a progress bar for the folds.
    ----
    Returns:
    - If return_scores=False: dict of best (M, rho) pairs for each criterion (cv, AIC, BIC).
    - If return_scores=True: tuple of (dict of best (M, rho) pairs, dict of score matrices).
    ----   
    """
    M_grid = np.array(M_grid)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_iter = (tqdm(kf.split(X), total=n_splits, desc="Folds")  if display_tqdm else kf.split(X))

    if rho_grid is None:
        cv_scores = np.zeros(len(M_grid), dtype=float)
        rhos = np.zeros(len(M_grid), dtype=float)
        for train_idx, val_idx in fold_iter:
            X_train, X_val = X[train_idx], X[val_idx]
            for ixM, M in enumerate(M_grid):
                _, base_f, grad_f, lap_f = density_estimate(manifold_type, X_train, M, X_val,grad=True, laplacian=True)
                grad_sq = np.sum(grad_f ** 2, axis=1)
                rho = np.percentile(base_f[base_f > 0], 5)
                rhos[ixM] += rho
                hat_f = np.maximum(base_f, rho) + eps
                score_vals = (2 * lap_f / (hat_f)) - (grad_sq / (hat_f**2))
                try: cv_scores[ixM] += np.mean(score_vals) 
                except: cv_scores[ixM] = np.nan 
        rhos /= n_splits
    else:
        rho_grid = np.array(rho_grid)
        cv_scores = np.zeros((len(M_grid), len(rho_grid)), dtype=float)
        for train_idx, val_idx in fold_iter:
            X_train, X_val = X[train_idx], X[val_idx]
            for ixM, M in enumerate(M_grid):
                _, base_f, grad_f, lap_f = density_estimate(manifold_type, X_train, M, X_val, grad=True, laplacian=True)
                grad_sq = np.sum(grad_f**2, axis=1)
                for ixRho, rho in enumerate(rho_grid):
                    if rho < base_f.mean():
                        hat_f = np.maximum(base_f, rho)
                        score_vals = (2 * lap_f / (hat_f)) - (grad_sq / (hat_f**2))
                        try: cv_scores[ixM, ixRho] += np.mean(score_vals) 
                        except: cv_scores[ixM, ixRho] = np.nan 
                    else: cv_scores[ixM, ixRho] = np.nan
    cv_scores /= n_splits    
    scores, params = get_scores(manifold_type, len(X), cv_scores, M_grid, rho_grid) 
    if rho_grid is None:
        params = {key: (val[0], rhos[val[1]]) for key, val in params.items()}
    if return_scores: return params, scores
    else:return params


def DensityKFoldCV(manifold_type, X, M_grid, rho_grid, 
                                    n_splits=5,
                                    return_scores=False,
                                    random_state=None,
                                    display_tqdm = True):
    '''
    Select the degree M and density lower bound rho by K-fold cross-validation.
    Optimized to compute the base density for M once and then threshold.
    '''
    M_grid = np.array(M_grid)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_iter = (tqdm(kf.split(X), total=n_splits, desc="Folds")  if display_tqdm else kf.split(X))

    if rho_grid is None:
        rhos = np.zeros(len(M_grid), dtype=float)
        cv_scores = np.zeros(len(M_grid), dtype=float)
        for train_idx, val_idx in fold_iter:
            X_train, X_val = X[train_idx], X[val_idx]
            for ixM, M in enumerate(M_grid):
                _, base_f = density_estimate(manifold_type, X_train, M, X_val, False, False)
                rho = np.percentile(base_f[base_f > 0], 5)
                rhos[ixM] += rho
                hat_f = np.maximum(base_f, rho)
                cv_scores[ixM] +=  np.mean(hat_f**2) - 2 * np.mean(hat_f)
        rhos /= n_splits
    else:
        rho_grid = np.array(rho_grid)
        cv_scores = np.zeros((len(M_grid), len(rho_grid)), dtype=float)
        for train_idx, val_idx in fold_iter:
            X_train, X_val = X[train_idx], X[val_idx]
            for ixM, M in enumerate(M_grid):
                _, base_f = density_estimate(manifold_type, X_train, M, X_val, grad=False, laplacian=False)
                for ixRho, rho in enumerate(rho_grid):
                    if rho < base_f.mean():
                        hat_f = np.maximum(base_f, rho)
                        try: cv_scores[ixM, ixRho] += np.mean(hat_f**2) - 2 * np.mean(hat_f)
                        except: cv_scores[ixM, ixRho] = np.nan 
                    else: cv_scores[ixM, ixRho] = np.nan
    cv_scores /= n_splits    
    scores, params = get_scores(manifold_type, len(X), cv_scores, M_grid, rho_grid)
    if rho_grid is None:
        params = {key: (val[0], rhos[val[1]]) for key, val in params.items()}
    if return_scores: return params, scores
    else:return params



def scoreMatching(manifold_type, X, M_grid, rho_grid, eval_grid = None, return_scores=False):
    M_grid = np.array(M_grid); rho_grid = np.array(rho_grid)
    cv_scores = np.zeros((len(M_grid), len(rho_grid)), dtype=float)
    Xeval = uniform_points(manifold_type, eval_grid) if eval_grid is not None else X
    for ixM, M in enumerate(M_grid):
        _, base_f, grad_f, lap_f = density_estimate(manifold_type, X, M, Xeval, grad=True, laplacian=True)
        grad_sq = np.sum(grad_f**2, axis=1)
        for ixRho, rho in enumerate(rho_grid):
            if rho < base_f.mean():
                hat_f = np.maximum(base_f, rho)
                eps = 1e-8
                score_vals = (2 * lap_f / (hat_f + eps)) - (grad_sq / (hat_f**2 + eps))
                mask = base_f > rho
                fill_value = np.mean(score_vals[mask]) if np.any(mask) else np.mean(score_vals)
                score_vals = np.where(mask, score_vals, fill_value)
                try: cv_scores[ixM, ixRho] += np.mean(score_vals) 
                except: cv_scores[ixM, ixRho] = np.nan 
            else:
                cv_scores[ixM, ixRho] = np.nan

    scores, Mrhostar = get_scores(manifold_type, len(X), cv_scores, M_grid, rho_grid)
    if return_scores: return Mrhostar, scores
    else:return Mrhostar