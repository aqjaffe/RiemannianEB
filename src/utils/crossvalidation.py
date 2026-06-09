
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



def k_to_M(manifold_type, k):
    """Map k individual modes (ordered by eigenvalue) to effective spectral degree M.

    Inverse of spectral_dimension:
      S1 : k = 2M+1      → M = (k-1)//2
      S2 : k = (M+1)^2   → M = floor(sqrt(k)) - 1  (largest M with (M+1)^2 ≤ k)
      T2 : k = (2M+1)^2  → M = floor((sqrt(k)-1)/2)
    Returns at least 1 (M=0 is a trivial constant density).
    """
    k = int(k)
    if manifold_type == 'S1':
        return max(1, (k - 1) // 2)
    elif manifold_type == 'S2':
        return max(1, int(np.sqrt(k + 1e-9)) - 1)
    elif manifold_type in ('T2', 'SO3'):
        return max(1, int((np.sqrt(k + 1e-9) - 1) / 2))
    else:
        raise ValueError(f"k_to_M not defined for manifold_type='{manifold_type}'")


def scoreMatchingKFoldCV(manifold_type, X, M_grid, rho_percentile=3,
                         n_splits=5, return_scores=False, random_state=None,
                         subsample=None, k_modes_grid=None):
    """
    Select density hyperparameters by K-fold CV using Hyvärinen score matching.

    Two search modes:
    - M-search (S1, S2): pass M_grid, leave k_modes_grid=None.
    - k-search (any): pass k_modes_grid, set M_grid=None.
      For T2 density_estimate is called with k_modes=k directly.
      For S1/S2 k is mapped to M_eff = k_to_M(manifold_type, k) and density_estimate
      is called with M=M_eff; k is still used as the AIC/BIC complexity penalty.
      Resolved params always return (best_k, rho); callers use k_to_M to get M.

    Parameters
    ----------
    subsample : int or None. Randomly subsample this many points before CV.
    """
    eps = 1e-5
    rng = np.random.default_rng(random_state)
    if subsample is not None and subsample < len(X):
        idx = rng.choice(len(X), int(subsample), replace=False)
        X   = X[idx]

    rho_percentile = np.atleast_1d(rho_percentile)
    n_eff = len(X)
    kf    = KFold(n_splits=n_splits, shuffle=True,
                  random_state=int(random_state) if random_state is not None else None)

    def _score_fold(base_f, grad_f, lap_f, rho_percentile):
        grad_sq    = np.sum(grad_f.reshape(len(grad_f), -1) ** 2, axis=1)
        positive_f = base_f[base_f > 0]
        scores = np.zeros(len(rho_percentile))
        for ixR, p in enumerate(rho_percentile):
            rho   = np.percentile(positive_f, p) if len(positive_f) else eps
            hat_f = np.maximum(base_f, rho) + eps
            scores[ixR] = np.mean((2 * lap_f / hat_f) - (grad_sq / hat_f ** 2))
        return scores

    def _density_k(manifold_type, X_tr, X_val, k, eval_only=False, X_full=None):
        """Call density_estimate for a given k, dispatching on manifold_type."""
        if manifold_type == 'T2':
            return density_estimate(manifold_type, X_tr, None,
                                    X_val if not eval_only else X_full,
                                    grad=not eval_only, laplacian=not eval_only,
                                    k_modes=int(k))
        else:
            M_eff = k_to_M(manifold_type, int(k))
            return density_estimate(manifold_type, X_tr, M_eff,
                                    X_val if not eval_only else X_full,
                                    grad=not eval_only, laplacian=not eval_only)

    # ── k-search mode ─────────────────────────────────────────────────────────
    if k_modes_grid is not None:
        k_arr     = np.atleast_1d(k_modes_grid)
        cv_scores = np.zeros((len(k_arr), len(rho_percentile)), dtype=float)

        for train_idx, val_idx in tqdm(kf.split(X), total=n_splits, desc="Folds"):
            X_train, X_val = X[train_idx], X[val_idx]
            for ixK, k in enumerate(k_arr):
                _, base_f, grad_f, lap_f = _density_k(manifold_type, X_train, X_val, k)
                cv_scores[ixK] += _score_fold(base_f, grad_f, lap_f, rho_percentile)

        cv_scores /= n_splits
        k_penalty  = k_arr[:, np.newaxis].astype(float)
        score_dict = {
            "cv":  cv_scores,
            "AIC": cv_scores + 2 * k_penalty / n_eff,
            "BIC": cv_scores + np.log(n_eff) * k_penalty / n_eff,
        }

        def _best_k_rho(mat):
            masked   = np.where(np.isfinite(mat), mat, np.inf)
            ixK, ixR = np.unravel_index(np.argmin(masked), mat.shape)
            return int(k_arr[ixK]), float(rho_percentile[ixR])

        params   = {c: _best_k_rho(score_dict[c]) for c in ("cv", "AIC", "BIC")}
        resolved = {}
        for key, (best_k, best_p) in params.items():
            if manifold_type == 'T2':
                _, f_full = density_estimate(manifold_type, X, None, X,
                                             grad=False, laplacian=False, k_modes=best_k)
            else:
                M_eff = k_to_M(manifold_type, best_k)
                _, f_full = density_estimate(manifold_type, X, M_eff, X,
                                             grad=False, laplacian=False)
            pos = f_full[f_full > 0]
            resolved[key] = (best_k, float(np.percentile(pos, best_p)) if len(pos) else eps)

        if return_scores:
            return resolved, score_dict, k_arr
        return resolved

    # ── M-search mode (S1, S2) ────────────────────────────────────────────────
    M_grid    = np.array(M_grid)
    cv_scores = np.zeros((len(M_grid), len(rho_percentile)), dtype=float)

    for train_idx, val_idx in tqdm(kf.split(X), total=n_splits, desc="Folds"):
        X_train, X_val = X[train_idx], X[val_idx]
        for ixM, M in enumerate(M_grid):
            _, base_f, grad_f, lap_f = density_estimate(manifold_type, X_train, M,
                                                        X_val, grad=True, laplacian=True)
            cv_scores[ixM] += _score_fold(base_f, grad_f, lap_f, rho_percentile)

    cv_scores /= n_splits
    scores, params = get_scores(manifold_type, n_eff, cv_scores, M_grid, rho_percentile)

    resolved = {}
    for key, (M, best_p) in params.items():
        _, f_full = density_estimate(manifold_type, X, M, X, grad=False, laplacian=False)
        resolved[key] = (M, np.percentile(f_full[f_full > 0], best_p))

    if return_scores:
        return resolved, scores
    return resolved



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


def plot_scoreCV_k(manifold_type, k_arr, score_dict, rho_percentile, best_params,
                   criterion='AIC', ax=None):
    """CV scores vs k with M-region brackets on the x-axis.

    For each contiguous group of k values sharing the same M_eff, a bracket and
    label are drawn below the x-axis. Vertical grey lines mark M-change boundaries.
    The optimal k* for the chosen criterion is marked with a vertical dashed line.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    k_arr          = np.asarray(k_arr)
    rho_percentile = np.atleast_1d(rho_percentile)
    scores         = score_dict[criterion]           # (n_k, n_rho)

    # Map each searched k to its effective M
    M_arr     = np.array([k_to_M(manifold_type, int(k)) for k in k_arr])
    unique_Ms = np.unique(M_arr)

    # Compute band edges (midpoints between consecutive k values)
    if len(k_arr) > 1:
        half_gaps = np.concatenate([
            [(k_arr[1] - k_arr[0]) / 2],
            (k_arr[1:] - k_arr[:-1]) / 2,
        ])
        edges = np.concatenate([k_arr - half_gaps, [k_arr[-1] + half_gaps[-1]]])
    else:
        edges = np.array([k_arr[0] - 0.5, k_arr[0] + 0.5])

    # Vertical lines at M-change boundaries
    for i in range(1, len(M_arr)):
        if M_arr[i] != M_arr[i - 1]:
            ax.axvline(edges[i], color='gray', ls=':', lw=0.8, alpha=0.6)

    # Plot score at best rho for each k
    best_ixR    = np.argmin(scores, axis=1)
    best_scores = scores[np.arange(len(k_arr)), best_ixR]
    ax.plot(k_arr, best_scores, 'k-o', ms=4, lw=1.2, label=f'{criterion} (best $\\rho$)')

    # Mark optimal k*
    best_k, _ = best_params[criterion]
    ax.axvline(best_k, color='C3', ls='--', lw=1.5,
               label=f'$k^*={best_k}$  ($M={k_to_M(manifold_type, int(best_k))}$)')

    ax.set_xlabel('$k$  (number of modes)')
    ax.set_ylabel('Score-matching CV score')
    ax.set_title(f'{manifold_type}  —  score-matching CV over $k$  ({criterion})')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, ls=':', lw=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # M-region brackets below the x-axis
    fig = ax.get_figure()
    fig.canvas.draw()                           # needed to resolve axis coordinates
    trans = ax.get_xaxis_transform()            # x=data, y=axes [0,1]
    bracket_y  = -0.10                          # just below x-axis tick labels
    label_y    = -0.17
    tip_height = 0.025                          # bracket arm height in axes coords

    for M in unique_Ms:
        idxs  = np.where(M_arr == M)[0]
        lo    = float(edges[idxs[0]])
        hi    = float(edges[idxs[-1] + 1])
        mid   = (lo + hi) / 2

        # Horizontal bar
        ax.annotate('', xy=(hi, bracket_y), xytext=(lo, bracket_y),
                    xycoords=trans, textcoords=trans,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))
        # Left tip
        ax.annotate('', xy=(lo, bracket_y), xytext=(lo, bracket_y + tip_height),
                    xycoords=trans, textcoords=trans,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))
        # Right tip
        ax.annotate('', xy=(hi, bracket_y), xytext=(hi, bracket_y + tip_height),
                    xycoords=trans, textcoords=trans,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))
        # M label
        ax.text(mid, label_y, f'$M\\!=\\!{M}$',
                ha='center', va='top', fontsize=7, color='gray',
                transform=trans)

    ax.tick_params(axis='x', pad=2)

    return ax


