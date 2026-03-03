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



def density_estimate(manifold_type, X, M, on_X, grad=True, laplacian=False):
    """
    Spectral density estimate on compact manifolds.
    Optionally returns gradient and Laplacian.
    """

    n_samples = X.shape[0]

    hat_grad_f = None
    hat_lap_f = None

    # ============================================================
    # S1
    # ============================================================
    if manifold_type == 'S1':
        theta = np.arctan2(X[:, 1], X[:, 0])
        phi = np.arctan2(on_X[:, 1], on_X[:, 0])

        k = np.arange(-M, M + 1)[:, None]
        moments = np.sum(np.exp(-1j * k * theta[None, :]), axis=1, keepdims=True)
        exp_k_phi = np.exp(1j * k * phi[None, :])

        norm_factor = 1 / (np.sqrt(2 * np.pi) * n_samples)

        hat_f = (moments * exp_k_phi).sum(axis=0).real * norm_factor

        if grad:
            d_f_d_phi = ((1j * k) * moments * exp_k_phi).sum(axis=0).real * norm_factor
            tangent_basis = np.stack([-on_X[:, 1], on_X[:, 0]], axis=1)
            hat_grad_f = d_f_d_phi[:, None] * tangent_basis

        if laplacian:
            hat_lap_f = ((-k**2) * moments * exp_k_phi).sum(axis=0).real * norm_factor


    # ============================================================
    # S2
    # ============================================================
    elif manifold_type == 'S2':

        poly = sp.special.legendre(0)
        poly_deriv = poly.deriv()
        lap_poly = 0 * poly

        for m in range(1, M):
            Pm = sp.special.legendre(m)
            poly += (2*m + 1) * Pm
            poly_deriv += (2*m + 1) * Pm.deriv()

            lap_poly += (2*m + 1) * (-m*(m+1)) * Pm

        hat_f = np.zeros(on_X.shape[0])
        if grad:
            hat_grad_f = np.zeros((on_X.shape[0], 3))
        if laplacian:
            hat_lap_f = np.zeros(on_X.shape[0])

        for i in range(n_samples):
            dot = on_X @ X[i, :]
            hat_f += poly(dot)

            if grad:
                hat_grad_f += np.outer(poly_deriv(dot), X[i, :])

            if laplacian:
                hat_lap_f += lap_poly(dot)

        hat_f /= (4 * np.pi * n_samples)
        if grad:
            hat_grad_f /= (4 * np.pi * n_samples)
        if laplacian:
            hat_lap_f /= (4 * np.pi * n_samples)


    # ============================================================
    # SO(3)
    # ============================================================
    elif manifold_type == 'SO3':

        m_vals = np.arange(M)
        weights = (2*m_vals + 1) / n_samples

        pairwise_tr = np.einsum('nij,kij->nk', X, on_X)
        cos_half_dists = 0.5 * np.sqrt(np.clip(pairwise_tr + 1, 0, 4))

        p_vals = np.zeros((M, X.shape[0], on_X.shape[0]))
        p_deriv = np.zeros_like(p_vals)

        for m in m_vals:
            cheb = sp.special.chebyu(2*m)
            cheb_deriv = cheb.deriv()
            p_vals[m] = cheb(cos_half_dists)
            p_deriv[m] = cheb_deriv(cos_half_dists)

        hat_f = np.einsum('m,mij->j', weights, p_vals)

        if grad:
            hat_grad_f = np.einsum('m,mij,ikl->jkl', weights, p_deriv, X)

        if laplacian:
            lap_eigs = -m_vals * (m_vals + 1)
            hat_lap_f = np.einsum('m,m,mij->j', weights, lap_eigs, p_vals)


    # ============================================================
    # T2 = S1 × S1
    # ============================================================
    elif manifold_type == 'T2':

        theta = np.arctan2(X[:, 0, 1], X[:, 0, 0])
        psi   = np.arctan2(X[:, 1, 1], X[:, 1, 0])
        phi   = np.arctan2(on_X[:, 0, 1], on_X[:, 0, 0])
        xi    = np.arctan2(on_X[:, 1, 1], on_X[:, 1, 0])

        k = np.arange(-M, M + 1)

        phase1 = k[:, None, None] * (phi[None, None, :] - theta[None, :, None])
        phase2 = k[:, None, None] * (xi[None, None, :] - psi[None, :, None])

        exp1 = np.exp(1j * phase1)
        exp2 = np.exp(1j * phase2)

        kernel = np.einsum('kng,lng->g', exp1, exp2)

        hat_f = kernel.real / (2 * np.pi * n_samples)

        if grad:
            dkernel_phi = np.einsum('kng,lng->g',
                                    1j * k[:, None, None] * exp1,
                                    exp2)

            dkernel_psi = np.einsum('kng,lng->g',
                                    exp1,
                                    1j * k[:, None, None] * exp2)

            d_f_d_phi = dkernel_phi.real / (2 * np.pi * n_samples)
            d_f_d_psi = dkernel_psi.real / (2 * np.pi * n_samples)

            tangent_phi = np.stack([-on_X[:, 0, 1], on_X[:, 0, 0]], axis=1)
            tangent_psi = np.stack([-on_X[:, 1, 1], on_X[:, 1, 0]], axis=1)

            hat_grad_f = np.zeros((on_X.shape[0], 2, 2))
            hat_grad_f[:, 0, :] = d_f_d_phi[:, None] * tangent_phi
            hat_grad_f[:, 1, :] = d_f_d_psi[:, None] * tangent_psi

        if laplacian:
            # Δ = ∂²/∂φ² + ∂²/∂ψ²
            lap_kernel = np.einsum(
                'kng,lng->g',
                (-k[:, None, None]**2) * exp1,
                exp2
            ) + np.einsum(
                'kng,lng->g',
                exp1,
                (-k[:, None, None]**2) * exp2
            )

            hat_lap_f = lap_kernel.real / (2 * np.pi * n_samples)


    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")

    if grad is False:
        return on_X, hat_f
    elif laplacian is False:
        return on_X, hat_f, hat_grad_f
    else:
        return on_X, hat_f, hat_grad_f, hat_lap_f



def kernel_density_estimate(manifold_type,X, on_X, kappa):
    """
    Kernel density estimate on S1.

    Parameters
    ----------
    manifold_type : str
        'S1' or 'S2' for circle or sphere.
    X : np.ndarray
        Data points on the manifold in extrinsic coordinates of shape
        (n_samples, dim) where dim=2 for S1, dim=3 for S2.
    on_X : np.ndarray, optional
        Points where to evaluate the density estimate in extrinsic coordinates.
        If None, evaluates on the data points X.
    kappa : float
        Bandwidth (concentration parameter for von Mises-Fisher kernel). Larger kappa = more smoothing.
        
    Returns
    -------
    grid : np.ndarray
        Grid points in extrinsic coordinates where the density is evaluated.
    hat_f : np.ndarray
        Estimated density values at the grid points.
    hat_grad_f : np.ndarray
        Estimated gradient of the density at the grid points.
    """
    n = X.shape[0]
    grid = on_X
    
    n_grid = grid.shape[0]
    
    if manifold_type == 'S1':
        # Circle case: von Mises kernel
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        grid_norm = grid / np.linalg.norm(grid, axis=1, keepdims=True)
        
        hat_f, hat_grad_f = np.zeros(n_grid), np.zeros((n_grid, 2))
        for i in range(n_grid):
            inner_prods = X_norm @ grid_norm[i]
            weights = np.exp(kappa * inner_prods)
            hat_f[i] = np.mean(weights)
            
            tangent_vecs = X_norm - inner_prods[:, np.newaxis] * grid_norm[i]
            hat_grad_f[i] = kappa * np.mean(weights[:, np.newaxis] * tangent_vecs, axis=0)

        # Normalize density (approximate normalization constant)
        normalization = 1.0 / (2 * np.pi * np.i0(kappa))
        hat_f *= normalization
        hat_grad_f *= normalization
        return grid, hat_f, hat_grad_f
        
    elif manifold_type == 'S2':
        # Sphere case: von Mises-Fisher kernel
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        grid_norm = grid / np.linalg.norm(grid, axis=1, keepdims=True)
        
        hat_f = np.zeros(n_grid)
        hat_grad_f = np.zeros((n_grid, 3))
        
        for i in range(n_grid):
            inner_prods = X_norm @ grid_norm[i]
            weights = np.exp(kappa * inner_prods)
            hat_f[i] = np.mean(weights)

            tangent_vecs = X_norm - inner_prods[:, np.newaxis] * grid_norm[i]
            hat_grad_f[i] = kappa * np.mean(
                weights[:, np.newaxis] * tangent_vecs, axis=0
            )
        # Normalize density (approximate normalization constant)
        normalization = kappa / (4 * np.pi * np.sinh(kappa))
        hat_f *= normalization
        hat_grad_f *= normalization
        
    else:
        raise ValueError(f"Unsupported manifold type: {manifold_type}")
    
    return grid, hat_f, hat_grad_f




# def density_estimate(manifold_type, X, M, on_X, grad = True):
#     """
#     Vectorized spherical density estimate using Legendre expansion.

#     Parameters
#     ----------
#     manifold_type : str
#         'S1' or 'S2' for circle or sphere.
#     M : int
#         degree of expansion
#     X : np.ndarray
#         Data points on the manifold in extrinsic coordinates of shape
#     on_X : np.ndarray, optional
#         Points where to evaluate the density estimate in extrinsic coordinates.
#         If None, evaluates on the data points X.
#     Returns
#     -------
#     grid : np.ndarray
#         Grid points in extrinsic coordinates where the density is to be evaluated.
#     hat_f : np.ndarray
#         Estimated density values at the grid points.
#     hat_grad_f : np.ndarray
#         Estimated gradient of the density at the grid points.
#     """

#     n_samples = X.shape[0]

#     if manifold_type == 'S1':
#         theta = np.arctan2(X[:, 1], X[:, 0])         
#         phi = np.arctan2(on_X[:, 1], on_X[:, 0])     
#         k = np.arange(-M, M + 1)[:, None]            
#         moments = np.sum(np.exp(-1j * k * theta[None, :]), axis=1, keepdims=True)
#         exp_k_phi = np.exp(1j * k * phi[None, :])
#         norm_factor = 1 / (np.sqrt(2 * np.pi) * n_samples)
#         hat_f = (moments * exp_k_phi).sum(axis=0).real * norm_factor
#         if grad:
#             d_f_d_phi = ((1j * k) * moments * exp_k_phi).sum(axis=0).real * norm_factor  # (G,)
#             tangent_basis = np.stack([-on_X[:, 1], on_X[:, 0]], axis=1)  # (G,2)
#             hat_grad_f = d_f_d_phi[:, None] * tangent_basis

#     elif manifold_type == 'S2':       
#         if False:
#             dots = np.dot(on_X,X.T)
#             hat_f = np.array([
#                 (2*m + 1)/(4*np.pi * n_samples) * np.sum( Legendre([0]*m + [1])  (dots), axis=-1)
#                 for m in range(M)]).sum(0)
#             hat_grad_f = np.array([[(2*m + 1)/(4*np.pi * n_samples) * np.sum(
#                     sp.special.legendre(m).deriv()(dots) * X[:, d], axis=-1
#                 ) for d in range(3)] for m in range(1, M)]).sum(0).T
#         else:
#             poly = sp.special.legendre(0)
#             poly_deriv = poly.deriv()
#             for m in range(1,M):
#                 poly += (2*m+1)*sp.special.legendre(m)
#                 poly_deriv += (2*m+1)*sp.special.legendre(m).deriv()

#             hat_f = np.zeros(shape=on_X.shape[0])
#             hat_grad_f = np.zeros(shape=(on_X.shape[0],3))
#             for i in range(X.shape[0]):  
#                 dot = on_X @ X[i,:]
#                 hat_f += poly(dot)
#                 if grad: hat_grad_f += np.outer(poly_deriv(dot), X[i,:])

#             hat_f /= (4*np.pi*X.shape[0])
#             if grad: hat_grad_f /= (4*np.pi*X.shape[0])
        

#     elif manifold_type == 'SO3':
#         m_vals = np.arange(M)
#         p_vals = np.zeros((M, X.shape[0], on_X.shape[0]))
#         weights = (2*m_vals + 1) / X.shape[0]  # Shape: (M,)
#         hat_f = (weights[:, None, None] * p_vals).sum(axis=(0, 1))

#         if grad:
#             pairwise_tr = np.einsum('nij,kij->nk', X, on_X)
#             cos_half_dists = 0.5 * np.sqrt(np.clip(pairwise_tr + 1, 0, 4))
#             p_deriv = np.zeros((M, X.shape[0], on_X.shape[0]))
#             for m in m_vals:
#                 cheb = sp.special.chebyu(2*m, monic=False)
#                 cheb_deriv = cheb.deriv()
#                 p_vals[m] = cheb(cos_half_dists)
#                 p_deriv[m] = cheb_deriv(cos_half_dists)
#             hat_grad_f = np.einsum('m,mij,ikl->jkl', weights, p_deriv, X)


#     elif manifold_type == 'T2':
#         # Extract angles from extrinsic coordinates
#         theta = np.arctan2(X[:, 0, 1], X[:, 0, 0])      # (n,)
#         psi   = np.arctan2(X[:, 1, 1], X[:, 1, 0])      # (n,)
#         phi   = np.arctan2(on_X[:, 0, 1], on_X[:, 0, 0])  # (G,)
#         xi    = np.arctan2(on_X[:, 1, 1], on_X[:, 1, 0])  # (G,)

#         k = np.arange(-M, M + 1)  
#         phase = (k[:, None, None] * (phi[None, None, :] - theta[None, :, None]))
#         phase_psi = (k[:, None, None] * (xi[None, None, :] - psi[None, :, None]) )

#         kernel = np.einsum('kng,lng->g',
#                         np.exp(1j * phase),
#                         np.exp(1j * phase_psi))  # (G,)
#         hat_f = kernel.real / (2 * np.pi * n_samples)

#         if grad:
#             dkernel_phi = np.einsum('kng,lng->g',
#                                     1j * k[:, None, None] * np.exp(1j * phase),
#                                     np.exp(1j * phase_psi))
#             dkernel_psi = np.einsum('kng,lng->g',
#                                     np.exp(1j * phase),
#                                     1j * k[:, None, None] * np.exp(1j * phase_psi))

#             d_f_d_phi = dkernel_phi.real / (2 * np.pi * n_samples)  # (G,)
#             d_f_d_psi = dkernel_psi.real / (2 * np.pi * n_samples)  # (G,)

#             # Tangent vectors for each circle factor: d/dphi of (cos,sin) = (-sin,cos)
#             tangent_phi = np.stack([-on_X[:, 0, 1], on_X[:, 0, 0]], axis=1)  # (G, 2)
#             tangent_psi = np.stack([-on_X[:, 1, 1], on_X[:, 1, 0]], axis=1)  # (G, 2)

#             # hat_grad_f shape (G, 2, 2): [point, factor, extrinsic_dim]
#             hat_grad_f = (
#                 d_f_d_phi[:, None] * tangent_phi[:, None, :] * np.array([1, 0])[None, :, None]
#                 + d_f_d_psi[:, None] * tangent_psi[:, None, :] * np.array([0, 1])[None, :, None]
#             )

#     else: 
#         raise ValueError(f"Unknown manifold type: {manifold_type}")
    
#     if grad is False:
#         hat_grad_f = None

#     return on_X, hat_f, hat_grad_f
    