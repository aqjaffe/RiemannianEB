import numpy as np
import scipy as sp
from tqdm import tqdm
from geomstats.geometry.hypersphere import Hypersphere # type: ignore
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from numpy.polynomial.legendre import Legendre
from scipy.special import sph_harm
from .helpers import get_manifold
S1 = Hypersphere(dim=1)
S2 = Hypersphere(dim=2)
SO3 = SpecialOrthogonal(n=3)
T2 = ProductManifold([Hypersphere(1), Hypersphere(1)])
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from .density_estimation import *
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import matplotlib.gridspec as gridspec
    
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

            # Labeling the outer boundaries
            if r == 0:
                ax_m.set_title(f"n = {n_val}\n" + ax_m.get_title(), fontsize=11, fontweight='bold')
            if c == 0:
                # Add the G name to the far left
                fig.text(0.08, 1 - (r + 0.5)/n_rows, f"G: {g_name}", 
                         va='center', rotation='vertical', fontsize=12, fontweight='bold')

    plt.suptitle(f"Split CV Distributions: M (Blue) vs ρ (Red)", fontsize=20, y=0.95)
    plt.show()

def plot_density_cv_scores(cv_scores, M_grid, rho_grid, title="CV Scores", ax = None):
    """
    Plots the CV score matrix using imshow with appropriate labels.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if len(rho_grid) > 1:
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


def spectral_dimension(manifold_type, M):
    if manifold_type == 'S1':
        return 2*M + 1

    elif manifold_type == 'S2':
        return (M + 1)**2

    elif manifold_type == 'T2':
        return (2*M + 1)**2

    elif manifold_type == 'SO3':
        # sum_{m=0}^M (2m+1)^2
        return np.sum((2*np.arange(M+1) + 1)**2)

    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")
    


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
    
def select_M_rho_by_DensityKFoldCV(manifold_type, X, M_grid, rho_grid, 
                                    n_splits=5,
                                    return_scores=False,
                                    random_state=None,
                                    display_tqdm = True):
    '''
    Select the degree M and density lower bound rho by K-fold cross-validation.
    Optimized to compute the base density for M once and then threshold.
    '''
    M_grid = np.array(M_grid)
    rho_grid = np.array(rho_grid)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = np.zeros((len(M_grid), len(rho_grid)), dtype=float)
    if display_tqdm:
            fold_iter = tqdm(kf.split(X), total=n_splits, desc="Cross-validating")
    else:
        fold_iter = kf.split(X)
    for train_idx, val_idx in fold_iter:
        X_train, X_val = X[train_idx], X[val_idx]
        for ixM, M in enumerate(M_grid):
            _, base_f_val, base_grad_f_val = density_estimate(manifold_type, X_train, M, X_val)
            for ixRho, rho in enumerate(rho_grid):
                if rho < base_f_val.mean():
                    hat_f = np.maximum(base_f_val, rho)
                    score = np.mean(hat_f**2) - 2 * np.mean(hat_f)
                    try: cv_scores[ixM, ixRho] += score 
                    except: cv_scores[ixM, ixRho] = np.nan 
                else:
                    cv_scores[ixM, ixRho] = np.nan 

    # Average scores across folds
    cv_scores /= n_splits
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
    

def select_M_by_DensityKFoldCV(manifold_type, X, M_grid, 
                                n_splits=5,
                                return_scores=False,
                                random_state=None,
                                # n_grid_samples=1000, 
                                earlystop = False, 
                                display_tqdm = True
                                ):
    '''
    Select the degree M of the Legendre expansion by K-fold cross-validation.

    Parameters
    ----------
    manifold_type : str
    X : np.ndarray
    M_grid : list or np.ndarray
    n_splits : int, optional (default=5)
    n_grid_samples : int, optional
    return_scores : bool, optional
    random_state : int or None

    Returns
    -------
    dict (or tuple if return_scores=True)
    '''
    M_grid = np.array(M_grid)
    # manifold = get_manifold(manifold_type)
    # uniform_samples = manifold.random_uniform(n_grid_samples)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


    cv_scores = np.zeros_like(M_grid, dtype=float)

    for ixM, M in enumerate(M_grid):

        fold_scores = []
        if display_tqdm:
            fold_iter = tqdm(kf.split(X), total=n_splits, desc=f'kfold (M={M})', leave=False)
        else:
            fold_iter = kf.split(X)
        for train_idx, val_idx in fold_iter:
            X_train = X[train_idx]
            X_val = X[val_idx]

            hat_f_square_integral = np.square(
                density_estimate(manifold_type, X_train, M, X_train, grad=False)[1] 
                # density_estimate(manifold_type, X_train, M,uniform_samples, grad=False)[1] 
                ).mean()
                   
            val_int = density_estimate(manifold_type, X_train, M, X_val, grad=False)[1].mean()
            fold_scores.append(hat_f_square_integral - 2 * val_int)

        cv_scores[ixM] = np.mean(fold_scores)
        if earlystop and M > 10 and ixM >= 5 and np.all(np.diff(cv_scores[ixM-5:ixM]) > 0):
            print(f"Early stopping at M={M}, selected M={M_grid[np.argmin(cv_scores[:ixM+1])]}")
            cv_scores = cv_scores[:ixM+1]; M_grid = M_grid[:ixM+1]
            break
   
    n = len(X)
    k_vals = np.array([spectral_dimension(manifold_type, M) for M in M_grid])

    AIC_scores = cv_scores + 2 * k_vals / n
    BIC_scores = cv_scores + np.log(n) * k_vals / n

    Mstar = {
        'cv':  M_grid[np.argmin(cv_scores)],
        'AIC': M_grid[np.argmin(AIC_scores)],
        'BIC': M_grid[np.argmin(BIC_scores)],
    }


    if return_scores:
        return Mstar, cv_scores
    else:
        return Mstar
    


# def select_M_by_DensityLOOCV(manifold_type, X, M_grid,
#                               n_grid_samples = 1000, 
#                               return_scores = False):
#     '''
#     Select the degree M of the Legendre expansion by leave-one-out cross-validation.
#     Parameters
#     ----------
#     manifold_type : str
#     X : np.ndarray
#     M_grid : list or np.ndarray
#     penalty : str, optional (default 'none', possible 'AIC', 'BIC')
#     n_grid_samples : int, optional
#     return_scores : bool, optional
#     Returns
#     '''
#     M_grid = np.array(M_grid)

#     manifold = get_manifold(manifold_type)
#     uniform_samples = manifold.random_uniform(n_grid_samples)

#     idx = np.arange(len(X))

#     cv_scores = np.zeros_like(M_grid, dtype=float)
#     for ixM, M in enumerate(M_grid):
#         hat_f_square_integral = np.square(density_estimate(manifold_type, X, M, uniform_samples, grad = False)[1]).mean()
#         # hat_f_square_integral = np.square(density_estimate(manifold_type, X, M, X, grad = False)[1]).mean()

#         loo_vals = [
#             density_estimate(manifold_type, X[idx != i], M, np.array([X[i]]), grad = False)[1]
#             for i in tqdm(idx, desc=f'loo (M={M})', position=0, leave=False)
#         ]

#         cv_scores[ixM] = hat_f_square_integral - 2 * np.mean(loo_vals)
#         if M > 10 and ixM >= 5 and np.all(np.diff(cv_scores[ixM-5:ixM]) > 0):
#             print(f"Early stopping at M={M}, selected M={M_grid[np.argmin(cv_scores[:ixM+1])]}")
#             cv_scores = cv_scores[:ixM+1]; M_grid = M_grid[:ixM+1]
#             break
#     n = len(X)
#     k_vals = np.array([spectral_dimension(manifold_type, M) for M in M_grid])

#     AIC_scores = cv_scores + 2 * k_vals / n
#     BIC_scores = cv_scores + np.log(n) * k_vals / n

#     Mstar = {
#         'cv':  M_grid[np.argmin(cv_scores)],
#         'AIC': M_grid[np.argmin(AIC_scores)],
#         'BIC': M_grid[np.argmin(BIC_scores)],
#     }


#     if return_scores:
#         return Mstar, cv_scores
#     else:
#         return Mstar