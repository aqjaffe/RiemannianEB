import numpy as np
import scipy as sp
from tqdm import tqdm
from geomstats.geometry.hypersphere import Hypersphere # type: ignore
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from numpy.polynomial.legendre import Legendre
from scipy.special import sph_harm
from .tools import get_manifold
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

from .density_estimation import *


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
    

    
def select_M_by_scoreMatchingKFoldCV(manifold_type, X, M_grid,
                                     n_splits=5,
                                     return_scores=False,
                                     random_state=None,
                                     rho=1e-10,
                                     tryearlystop = False):
    """
    Select truncation level M by K-fold cross-validation
    using Hyvärinen score matching.
    """

    M_grid = np.array(M_grid)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_scores = np.zeros_like(M_grid, dtype=float)

    for ixM, M in enumerate(M_grid):

        fold_scores = []

        for train_idx, val_idx in tqdm(
                kf.split(X),
                desc=f'Score-CV (M={M})',
                leave=False):

            X_train = X[train_idx]
            X_val = X[val_idx]

            # Evaluate f, grad f, Laplacian f on validation set
            _, hat_f, hat_grad_f, hat_lap_f = density_estimate(
                manifold_type,
                X_train,
                M,
                X_val,
                grad=True,
                laplacian=True
            )

            hat_f = np.clip(hat_f, rho, None)
            # # ||grad f||^2
            # if manifold_type == 'T2':
            #     grad_sq = np.sum(hat_grad_f**2, axis=(1,2))
            # else:
            grad_sq = np.sum(hat_grad_f**2, axis=1)

            grad_log_sq = grad_sq / (hat_f**2)
            lap_log = (hat_lap_f / hat_f) - (grad_sq / (hat_f**2))
            hyvarinen_vals = grad_log_sq + 2 * lap_log
            fold_scores.append(np.mean(hyvarinen_vals))

        cv_scores[ixM] = np.mean(fold_scores)

        # Early stopping (optional)
        
        if tryearlystop and M > 10 and ixM >= 5 and np.all(np.diff(cv_scores[ixM-5:ixM]) > 0):
            print(f"Early stopping at M={M}")
            cv_scores = cv_scores[:ixM+1]
            M_grid = M_grid[:ixM+1]
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
    

def select_M_by_DensityKFoldCV(manifold_type, X, M_grid, 
                                n_splits=5,
                                n_grid_samples=1000, 
                                return_scores=False,
                                random_state=None):
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
        for train_idx, val_idx in tqdm(kf.split(X),desc=f'kfold (M={M})',leave=False):
            X_train = X[train_idx]
            X_val = X[val_idx]

            hat_f_square_integral = np.square(
                density_estimate(manifold_type, X_train, M, X_train, grad=False)[1] 
                # density_estimate(manifold_type, X_train, M,uniform_samples, grad=False)[1] 
                ).mean()
                   
            val_int = density_estimate(manifold_type, X_train, M, X_val, grad=False)[1].mean()
            fold_scores.append(hat_f_square_integral - 2 * val_int)

        cv_scores[ixM] = np.mean(fold_scores)
        if M > 10 and ixM >= 5 and np.all(np.diff(cv_scores[ixM-5:ixM]) > 0):
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
    


def select_M_by_DensityLOOCV(manifold_type, X, M_grid,
                              n_grid_samples = 1000, 
                              return_scores = False):
    '''
    Select the degree M of the Legendre expansion by leave-one-out cross-validation.
    Parameters
    ----------
    manifold_type : str
    X : np.ndarray
    M_grid : list or np.ndarray
    penalty : str, optional (default 'none', possible 'AIC', 'BIC')
    n_grid_samples : int, optional
    return_scores : bool, optional
    Returns
    '''
    M_grid = np.array(M_grid)

    manifold = get_manifold(manifold_type)
    uniform_samples = manifold.random_uniform(n_grid_samples)

    idx = np.arange(len(X))

    cv_scores = np.zeros_like(M_grid, dtype=float)
    for ixM, M in enumerate(M_grid):
        hat_f_square_integral = np.square(density_estimate(manifold_type, X, M, uniform_samples, grad = False)[1]).mean()
        # hat_f_square_integral = np.square(density_estimate(manifold_type, X, M, X, grad = False)[1]).mean()

        loo_vals = [
            density_estimate(manifold_type, X[idx != i], M, np.array([X[i]]), grad = False)[1]
            for i in tqdm(idx, desc=f'loo (M={M})', position=0, leave=False)
        ]

        cv_scores[ixM] = hat_f_square_integral - 2 * np.mean(loo_vals)
        if M > 10 and ixM >= 5 and np.all(np.diff(cv_scores[ixM-5:ixM]) > 0):
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