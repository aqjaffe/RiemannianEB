
import numpy as np
from geomstats.geometry.hypersphere import Hypersphere # type: ignore
from numpy.polynomial.legendre import Legendre
import scipy as sp 
import numpy as np
from geomstats.geometry.hypersphere import Hypersphere # type: ignore
from numpy.polynomial.legendre import Legendre
import scipy as sp
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
    if manifold_type not in ['S1', 'S2']:
        raise ValueError("manifold_type must be 'S1' or 'S2'")

    n_samples = X.shape[0]
    if manifold_type == 'S1':
        k = np.arange(-M, M + 1)[:, None, None]  # (2M+1,1,1)
        on_X_I = Hypersphere(1).extrinsic_to_intrinsic_coords(on_X).ravel()
        X_I = Hypersphere(1).extrinsic_to_intrinsic_coords(X).ravel()
        diff = on_X_I[None, :] - X_I[:, None]  # (N, len(x_grid))
        hat_f = np.real(np.exp(1j * k * diff[None, :, :]).sum(axis=(0,1))) / (np.sqrt(2*np.pi) * n_samples)
        hat_grad_f = np.real((1j * k * np.exp(1j * k * diff[None, :, :])).sum(axis=(0,1))) / (np.sqrt(2*np.pi) * n_samples)

    if manifold_type == 'S2':
        dots = np.dot(on_X,X.T)
        hat_f = np.array([
            (2*m + 1)/(4*np.pi * n_samples) * np.sum( Legendre([0]*m + [1])  (dots), axis=-1)
            for m in range(M)]).sum(0)

        hat_grad_f = np.array([[(2*m + 1)/(4*np.pi * n_samples) * np.sum(
                sp.special.legendre(m).deriv()(dots) * X[:, d], axis=-1
            ) for d in range(3)] for m in range(1, M)]).sum(0).T
        
    return on_X, hat_f, hat_grad_f
    