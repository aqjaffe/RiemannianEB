
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib import gridspec
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
        """Return the best (M, rho) pair; break ties by smallest rho, smallest M."""
        masked = np.where(np.isfinite(score_matrix), score_matrix, np.inf)
        min_val = np.min(masked)
        if not np.isfinite(min_val):
            return (np.nan, np.nan)
        candidates = np.argwhere(masked == min_val)
        if np.any(rho_grid == None):
            best = max(candidates, key=lambda ix: -M_grid[ix[0]])
            ixM = int(best[0])
            return (M_grid[ixM], ixM)
        else:
            best = max(candidates, key=lambda ix: (-rho_grid[ix[1]], -M_grid[ix[0]]))
            ixM, ixRho = int(best[0]), int(best[1])
            return (M_grid[ixM], rho_grid[ixRho])

    k_vals = np.array([spectral_dimension(manifold_type, M) for M in M_grid])
    if np.any(rho_grid == None): k_penalty = k_vals
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



def scoreMatchingKFoldCV(manifold_type, X, M_grid, rho_percentile = 3,
                                         n_splits=5,
                                         return_scores=False,
                                         random_state=None,
                                         display_tqdm = True,
                                         eps = 1e-5,
                                         ):
    
    """
    Select truncation level M and density lower bound rho by K-fold CV using Hyvärinen score matching.
     Score = ||grad f||^2 / max(f,rho)^2 - 2 * (lap f / max(f,rho))
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
    rho_percentile = np.atleast_1d(rho_percentile)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_iter = (tqdm(kf.split(X), total=n_splits, desc="Folds") if display_tqdm else kf.split(X))

    cv_scores = np.zeros((len(M_grid), len(rho_percentile)), dtype=float)

    for train_idx, val_idx in fold_iter:
        X_train, X_val = X[train_idx], X[val_idx]
        for ixM, M in enumerate(M_grid):
            _, base_f, grad_f, lap_f = density_estimate(manifold_type, X_train, M, X_val, grad=True, laplacian=True)
            grad_sq = np.sum(grad_f.reshape(len(grad_f), -1) ** 2, axis=1)
            positive_f = base_f[base_f > 0]
            for ixRho, p in enumerate(rho_percentile):
                rho = np.percentile(positive_f, p)
                hat_f = np.maximum(base_f, rho) + eps
                score_vals = (2 * lap_f / hat_f) - (grad_sq / hat_f**2)
                cv_scores[ixM, ixRho] += np.mean(score_vals)

    cv_scores /= n_splits

    scores, params = get_scores(manifold_type, len(X), cv_scores, M_grid, rho_percentile) # this gives best index!

    resolved_params = {}
    for key, (M, best_p) in params.items():
        _, base_f_full = density_estimate(manifold_type, X, M, X, grad=False, laplacian=False)
        resolved_params[key] = (M, np.percentile(base_f_full[base_f_full > 0], best_p)) # this gives best rho!
    if return_scores: return resolved_params, scores
    else: return resolved_params


def DensityKFoldCV(manifold_type, X, M_grid, rho_percentile = 3, 
                                    n_splits=5,
                                    return_scores=False,
                                    random_state=None,
                                    display_tqdm = True,
                                    eps = 1e-5):
    '''
    Select the degree M and density lower bound rho by K-fold cross-validation.
    Optimized to compute the base density for M once and then threshold.
    '''

    M_grid = np.array(M_grid)
    rho_percentile = np.atleast_1d(rho_percentile)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_iter = (tqdm(kf.split(X), total=n_splits, desc="Folds") if display_tqdm else kf.split(X))

    cv_scores = np.zeros((len(M_grid), len(rho_percentile)), dtype=float)

    for train_idx, val_idx in fold_iter:
        X_train, X_val = X[train_idx], X[val_idx]
        for ixM, M in enumerate(M_grid):
            _, base_f = density_estimate(manifold_type, X_train, M, X_val, grad=False, laplacian=False)
            positive_f = base_f[base_f > 0]
            for ixRho, p in enumerate(rho_percentile):
                rho = np.percentile(positive_f, p)
                hat_f = np.maximum(base_f, rho) + eps
                cv_scores[ixM, ixRho] += np.mean(hat_f**2) - 2 * np.mean(hat_f)
    cv_scores /= n_splits
    scores, params = get_scores(manifold_type, len(X), cv_scores, M_grid, rho_percentile) # this gives best index!
    resolved_params = {}
    for key, (M, best_p) in params.items():
        _, base_f_full = density_estimate(manifold_type, X, M, X, grad=False, laplacian=False)
        resolved_params[key] = (M, np.percentile(base_f_full[base_f_full > 0], best_p)) # this gives best rho!
    if return_scores: return resolved_params, scores
    else: return resolved_params


def plot_cv_scores(cv_scores, M_grid, rho_grid, title="CV Scores", ax = None):
    """
    Plots the CV score matrix using imshow with appropriate labels.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    rho_grid = np.atleast_1d(rho_grid) if rho_grid is not None else None
    if (rho_grid is not None) and (len(rho_grid) > 1 ):
        # Use imshow to display the matrix
        im = ax.imshow(cv_scores, aspect='auto', origin='lower', cmap='viridis')         # origin='lower' puts the first element of M_grid at the bottom
        ax.set_xticks(np.arange(len(rho_grid)))
        ax.set_xticklabels([f"{r:.2e}" if r < 0.01 else f"{r:.2f}" for r in rho_grid], rotation=45)
        ax.set_xlabel(r"$\rho$ (Lower Bound)")
        plt.colorbar(im, label='Score')
        ax.set_yticks(np.arange(len(M_grid)))
        ax.set_yticklabels(M_grid)
        ax.set_ylabel(r"$M$ (Expansion Degree)")
    else:
        # If there's only one rho, plot cv_scores as a line plot
        ax.plot(M_grid, cv_scores.flatten(), marker='o')
        ax.set_xlabel(r"$M$ (Expansion Degree)")
    ax.set_title(title)
    if ax is None:
        plt.tight_layout()
        plt.show()
    return None




# ----- crossvalidation visualization -----
def plot_cv_distributions_split(results_ocv, params):
    if results_ocv.mean_cv_loss.unique()[0] != results_ocv.mean_cv_loss.unique()[0]: return None
    ID, selected_sigma2 = float(params['ID']), params['sigma2']
    # Filter for the specific NMC
    df = results_ocv[(results_ocv.ID == ID) & (results_ocv.sigma2 == selected_sigma2)] .copy()
    
    # M_grid = list(map(int, results_ocv.Ms_grid.values[0].strip('[]').split()))
    # rho_grid = ast.literal_eval(results_ocv.rhos_grid.values[0])
    M_grid = params['M_grid']
    rho_grid = params['rho_grid']
    unique_Gs = df['G'].unique()
    unique_ns = sorted(df['num_samples'].unique())
    
    n_rows = len(unique_Gs)
    n_cols = len(unique_ns)
    
    # Increase figure size: each panel now has two sub-plots
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    
    # Outer grid: Rows = G, Cols = num_samples
    outer_grid = gridspec.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.5)

    for r, g_name in enumerate(unique_Gs):
        for c, n_val in enumerate(unique_ns):
            # Create a inner grid for M and Rho histograms
            inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[r, c], hspace=0.4)
            
            row = df[(df.G == g_name) & (df.num_samples == n_val)]
            
            if row.empty or row.cv_Ms_star.iloc[0] is None:
                ax_dummy = fig.add_subplot(outer_grid[r, c])
                ax_dummy.text(0.5, 0.5, "No CV Data", ha='center')
                ax_dummy.axis('off')
                continue
            
            ms_data = row.cv_Ms_star.iloc[0]
            rhos_data = row.cv_rhos_star.iloc[0]

            # --- Subplot 1: M Histogram (Top) ---
            ax_m = fig.add_subplot(inner_grid[0, 0])
            ax_m.hist(ms_data, bins=M_grid, color='tab:blue', alpha=0.7, edgecolor='black', align = 'right')
            ax_m.set_xticks(M_grid)
            ax_m.set_title(f"M dist.", fontsize=9)
            ax_m.tick_params(axis='both', labelsize=8)
            
            # --- Subplot 2: Rho Histogram (Bottom) ---
            ax_rho = fig.add_subplot(inner_grid[1, 0])
            ax_rho.hist(rhos_data, bins=rho_grid, color='tab:red', alpha=0.7, edgecolor='black', align = 'mid')
            ax_rho.set_title(f"ρ dist.", fontsize=9)
            ax_rho.tick_params(axis='both', labelsize=8)
            ax_rho.set_xticks(rho_grid)
            ax_rho.set_xscale('log')

            # Labeling the outer boundaries
            if r == 0:
                ax_m.set_title(f"n = {n_val}\n" + ax_m.get_title(), fontsize=11, fontweight='bold')
            if c == 0:
                # Add the G name to the far left
                fig.text(0.08, 1 - (r + 0.5)/n_rows, f"G: {g_name}", 
                         va='center', rotation='vertical', fontsize=12, fontweight='bold')

    plt.suptitle(f"Split CV Distributions: M (Blue) vs ρ (Red)", fontsize=20, y=0.95)
    plt.show()
