import numpy as np
import scipy as sp
from geomstats.geometry.hypersphere import Hypersphere # type: ignore
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from numpy.polynomial.legendre import Legendre
from scipy.special import sph_harm

def density_estimate(manifold_type, X, M, on_X):
    """
    Vectorized spherical density estimate using Legendre expansion.

    Parameters
    ----------
    manifold_type : str
        'S1' or 'S2' for circle or sphere.
    M : int
        degree of expansion
    X : np.ndarray
        Data points on the manifold in extrinsic coordinates of shape
    on_X : np.ndarray, optional
        Points where to evaluate the density estimate in extrinsic coordinates.
        If None, evaluates on the data points X.
    Returns
    -------
    grid : np.ndarray
        Grid points in extrinsic coordinates where the density is to be evaluated.
    hat_f : np.ndarray
        Estimated density values at the grid points.
    hat_grad_f : np.ndarray
        Estimated gradient of the density at the grid points.
    """
    if manifold_type not in ['S1', 'S2','SO3']:
        raise ValueError("manifold_type must be 'S1' or 'S2'")
    
    n_samples = X.shape[0]

    if manifold_type == 'S1':
        k = np.arange(-M, M + 1)
        angles_on_X = np.arctan2(on_X[:, 1], on_X[:, 0])  
        angles_X = np.arctan2(X[:, 1], X[:, 0]) 
        diff = angles_on_X[None, :] - angles_X[:, None] 
        diff = np.angle(np.exp(1j * diff))
        norm_factor = 1 / (np.sqrt(2*np.pi) * n_samples)
        exp_k_diff = np.exp(1j * k[:, None, None] * diff[None, :, :])
        hat_f = exp_k_diff.sum(axis=(0, 1)).real * norm_factor
        hat_grad_f = (1j * k[:, None, None] * exp_k_diff).sum(axis=(0, 1)).real * norm_factor

    if manifold_type == 'S2':
        dots = np.dot(on_X,X.T)
        hat_f = np.array([
            (2*m + 1)/(4*np.pi * n_samples) * np.sum( Legendre([0]*m + [1])  (dots), axis=-1)
            for m in range(M)]).sum(0)
        hat_grad_f = np.array([[(2*m + 1)/(4*np.pi * n_samples) * np.sum(
                sp.special.legendre(m).deriv()(dots) * X[:, d], axis=-1
            ) for d in range(3)] for m in range(1, M)]).sum(0).T
    
    if manifold_type == 'SO3':
        SO3 = SpecialOrthogonal(n=3)
        pairwise_tr = np.einsum('nij,kij->nk', X, on_X)
        cos_half_dists = 0.5 * np.sqrt(np.clip(pairwise_tr + 1, 0, 4))
        m_vals = np.arange(M)
        p_vals = np.zeros((M, X.shape[0], on_X.shape[0]))
        p_deriv = np.zeros((M, X.shape[0], on_X.shape[0]))
        for m in m_vals:
            cheb = sp.special.chebyu(2*m, monic=False)
            cheb_deriv = cheb.deriv()
            p_vals[m] = cheb(cos_half_dists)
            p_deriv[m] = cheb_deriv(cos_half_dists)
        weights = (2*m_vals + 1) / X.shape[0]  # Shape: (M,)
        hat_f = (weights[:, None, None] * p_vals).sum(axis=(0, 1))
        hat_grad_f = np.einsum('m,mij,ikl->jkl', weights, p_deriv, X)


    return on_X, hat_f, hat_grad_f
    



# alternative for S1

        # k = np.arange(-M, M + 1)[:, None, None]  # (2M+1,1,1)
        # on_X_I = Hypersphere(1).extrinsic_to_intrinsic_coords(on_X).ravel()
        # X_I = Hypersphere(1).extrinsic_to_intrinsic_coords(X).ravel()
        # diff = on_X_I[None, :] - X_I[:, None]  # (N, len(x_grid))
        # hat_f = np.real(np.exp(1j * k * diff[None, :, :]).sum(axis=(0,1))) / (np.sqrt(2*np.pi) * n_samples)
        # hat_grad_f = np.real((1j * k * np.exp(1j * k * diff[None, :, :])).sum(axis=(0,1))) / (np.sqrt(2*np.pi) * n_samples)


# alternative for S3
        # pairwise_dists = np.array([[SO3.metric.dist(x, y) for y in on_X] for x in X])
        # cos_half_dists = np.cos(pairwise_dists / 2)

        # p_vals = np.array([sp.special.chebyu(2*m, monic=False)(cos_half_dists) for m in m_vals])
        # p_deriv = np.array([sp.special.chebyu(2*m).deriv()(cos_half_dists) for m in m_vals])
        # hat_f = (weights[:, None, None] * p_vals).sum(axis=(0, 1))
        # hat_grad_f = np.einsum('m,mij,ikl->jkl', weights, p_deriv, X)