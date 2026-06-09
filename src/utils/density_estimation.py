import numpy as np
import scipy as sp
<<<<<<< HEAD


def density_estimate(manifold_type, X, M, on_X, grad=True, laplacian=False, normalise=False, k_modes=None):
=======
from geomstats.geometry.hypersphere import Hypersphere # type: ignore
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from numpy.polynomial.legendre import Legendre

from scipy.special import sph_harm

circle = Hypersphere(dim=1)
sphere = Hypersphere(dim=2)


import numpy as np



def density_estimate(manifold_type, X, M, on_X):
>>>>>>> f7c80cf7d36e48d2656bd3e47eace04afa3fcb5c
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

        norm_factor = 1 / (2 * np.pi * n_samples)

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
                g_amb = np.outer(poly_deriv(dot), X[i, :])
                radial_comp = np.sum(g_amb * on_X, axis=1, keepdims=True)
                hat_grad_f += (g_amb - radial_comp * on_X)
                # hat_grad_f += np.outer(poly_deriv(dot), X[i, :])

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

        theta = np.arctan2(X[:, 1, 0], X[:, 0, 0])
        psi   = np.arctan2(X[:, 1, 1], X[:, 0, 1])
        phi   = np.arctan2(on_X[:, 1, 0], on_X[:, 0, 0])
        xi    = np.arctan2(on_X[:, 1, 1], on_X[:, 0, 1])

        if k_modes is not None:
            # Individual Fourier modes ordered by energy |k1|²+|k2|² (the T2
            # Laplacian eigenvalue). k_modes=1 is just the DC mode (0,0).
            # The requested count is snapped *up* to a complete energy shell so
            # the truncation set is symmetric under k1<->k2 and sign flips —
            # i.e. balanced resolution across both torus coordinates instead of
            # giving one coordinate more modes than the other mid-shell.
            M_bound = int(np.ceil(np.sqrt(k_modes))) + 2
            all_pairs = [(k1, k2)
                         for k1 in range(-M_bound, M_bound + 1)
                         for k2 in range(-M_bound, M_bound + 1)]
            all_pairs.sort(key=lambda p: (p[0]**2 + p[1]**2, abs(p[0]), abs(p[1])))
            energies = np.array([p[0]**2 + p[1]**2 for p in all_pairs])
            cutoff   = energies[min(k_modes, len(all_pairs)) - 1]
            k_eff    = int(np.searchsorted(energies, cutoff, side='right'))
            modes    = np.array(all_pairs[:k_eff])           # (K, 2) int

            # empirical Fourier coefficients c[j] = Σ_n exp(-i(k1 θ_n + k2 ψ_n))
            phases_tr = modes[:, 0:1] * theta[None, :] + modes[:, 1:2] * psi[None, :]  # (K, n)
            c         = np.sum(np.exp(-1j * phases_tr), axis=1)                          # (K,)

            # basis at evaluation points
            phases_ev = modes[:, 0:1] * phi[None, :] + modes[:, 1:2] * xi[None, :]     # (K, n_eval)
            basis     = np.exp(1j * phases_ev)                                            # (K, n_eval)

            norm = (2 * np.pi)**2 * n_samples
            hat_f = np.dot(c, basis).real / norm

            if grad:
                d_f_d_phi = np.dot((1j * modes[:, 0]) * c, basis).real / norm
                d_f_d_psi = np.dot((1j * modes[:, 1]) * c, basis).real / norm

                tangent_phi = np.stack([-on_X[:, 1, 0], on_X[:, 0, 0]], axis=1)
                tangent_psi = np.stack([-on_X[:, 1, 1], on_X[:, 0, 1]], axis=1)

                hat_grad_f = np.zeros((on_X.shape[0], 2, 2))
                hat_grad_f[:, 0, :] = d_f_d_phi[:, None] * tangent_phi
                hat_grad_f[:, 1, :] = d_f_d_psi[:, None] * tangent_psi

            if laplacian:
                lap_w     = -(modes[:, 0]**2 + modes[:, 1]**2) * c   # (K,) complex
                hat_lap_f = np.dot(lap_w, basis).real / norm

        else:
            k = np.arange(-M, M + 1)

            phase1 = k[:, None, None] * (phi[None, None, :] - theta[None, :, None])
            phase2 = k[:, None, None] * (xi[None, None, :] - psi[None, :, None])

            exp1 = np.exp(1j * phase1)
            exp2 = np.exp(1j * phase2)

            E1 = exp1.sum(axis=0)   # (N, G): D_M(φ_g - θ_n) per data point
            E2 = exp2.sum(axis=0)   # (N, G): D_M(ξ_g - ψ_n) per data point

            kernel = np.einsum('ng,ng->g', E1, E2)

            hat_f = kernel.real / ((2 * np.pi)**2 * n_samples)

            if grad:
                dE1 = (1j * k[:, None, None] * exp1).sum(axis=0)
                dE2 = (1j * k[:, None, None] * exp2).sum(axis=0)

                dkernel_phi = np.einsum('ng,ng->g', dE1, E2)
                dkernel_psi = np.einsum('ng,ng->g', E1, dE2)

                d_f_d_phi = dkernel_phi.real / ((2 * np.pi)**2 * n_samples)
                d_f_d_psi = dkernel_psi.real / ((2 * np.pi)**2 * n_samples)

                tangent_phi = np.stack([-on_X[:, 1, 0], on_X[:, 0, 0]], axis=1)
                tangent_psi = np.stack([-on_X[:, 1, 1], on_X[:, 0, 1]], axis=1)

                hat_grad_f = np.zeros((on_X.shape[0], 2, 2))
                hat_grad_f[:, 0, :] = d_f_d_phi[:, None] * tangent_phi
                hat_grad_f[:, 1, :] = d_f_d_psi[:, None] * tangent_psi

            if laplacian:
                # Δ = ∂²/∂φ² + ∂²/∂ψ²
                d2E1 = ((-k[:, None, None]**2) * exp1).sum(axis=0)
                d2E2 = ((-k[:, None, None]**2) * exp2).sum(axis=0)

                lap_kernel = (np.einsum('ng,ng->g', d2E1, E2) +
                              np.einsum('ng,ng->g', E1, d2E2))

                hat_lap_f = lap_kernel.real / ((2 * np.pi)**2 * n_samples)


    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")


    if grad is False:
        return on_X, hat_f
    elif laplacian is False:
        return on_X, hat_f, hat_grad_f
    else:
        return on_X, hat_f, hat_grad_f, hat_lap_f



def T2_shell_modes(max_modes):
    """Cumulative T2 Fourier-mode counts at complete energy-shell boundaries.

    Returns the strictly increasing array of k_modes values for which the
    energy-ordered truncation {(k1,k2) : k1²+k2² ≤ E} is a *complete* shell,
    i.e. symmetric under k1<->k2 and sign flips. Use these as the CV grid
    (k_modes_grid) so resolution stays balanced across both torus coordinates
    and the AIC/BIC mode-count penalty matches the truncation exactly (no
    partial shells, no wasted duplicate candidates).

    Yields counts like 1, 5, 9, 13, 21, 25, ... up to (and including) max_modes.
    """
    M_bound  = int(np.ceil(np.sqrt(max_modes))) + 2
    energies = np.array(sorted(
        k1**2 + k2**2
        for k1 in range(-M_bound, M_bound + 1)
        for k2 in range(-M_bound, M_bound + 1)
    ))
    boundaries = np.searchsorted(energies, np.unique(energies), side='right')
    return boundaries[boundaries <= max_modes]


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
<<<<<<< HEAD
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


=======
        theta = np.arctan2(X[:, 1], X[:, 0])         
        phi = np.arctan2(on_X[:, 1], on_X[:, 0])     
        k = np.arange(-M, M + 1)[:, None]            
        moments = np.sum(np.exp(-1j * k * theta[None, :]), axis=1, keepdims=True)
        exp_k_phi = np.exp(1j * k * phi[None, :])
        norm_factor = 1 / (np.sqrt(2 * np.pi) * n_samples)
        hat_f = (moments * exp_k_phi).sum(axis=0).real * norm_factor
        d_f_d_phi = ((1j * k) * moments * exp_k_phi).sum(axis=0).real * norm_factor  # (G,)
        tangent_basis = np.stack([-on_X[:, 1], on_X[:, 0]], axis=1)  # (G,2)
        hat_grad_f = d_f_d_phi[:, None] * tangent_basis

    if manifold_type == 'S2':       
        if False:
            dots = np.dot(on_X,X.T)
            hat_f = np.array([
                (2*m + 1)/(4*np.pi * n_samples) * np.sum( Legendre([0]*m + [1])  (dots), axis=-1)
                for m in range(M)]).sum(0)
            hat_grad_f = np.array([[(2*m + 1)/(4*np.pi * n_samples) * np.sum(
                    sp.special.legendre(m).deriv()(dots) * X[:, d], axis=-1
                ) for d in range(3)] for m in range(1, M)]).sum(0).T
        else:
            poly = sp.special.legendre(0)
            poly_deriv = poly.deriv()
            for m in range(1,M):
                poly += (2*m+1)*sp.special.legendre(m)
                poly_deriv += (2*m+1)*sp.special.legendre(m).deriv()

            hat_f = np.zeros(shape=on_X.shape[0])
            hat_grad_f = np.zeros(shape=(on_X.shape[0],3))
            for i in range(X.shape[0]):  
                dot = on_X @ X[i,:]
                hat_f += poly(dot)
                hat_grad_f += np.outer(poly_deriv(dot), X[i,:])

            hat_f /= (4*np.pi*X.shape[0])
            hat_grad_f /= (4*np.pi*X.shape[0])
        

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
# # FIRST
#         k = np.arange(-M, M + 1)
#         dots = np.dot(X, on_X.T)                         
#         cross = X[:, 0:1] * on_X[:, 1] - X[:, 1:2] * on_X[:, 0] 
#         diff = np.arctan2(cross, dots)    
#         norm_factor = 1 / (np.sqrt(2*np.pi) * n_samples)
#         exp_k_diff = np.exp(1j * k[:, None, None] * diff[None, :, :])
#         hat_f = exp_k_diff.sum(axis=(0, 1)).real * norm_factor
#         hat_grad_f = (1j * k[:, None, None] * exp_k_diff).sum(axis=(0, 1)).real * norm_factor
# SECOND
        # k = np.arange(-M, M + 1)[:, None, None]  # (2M+1,1,1)
        # on_X_I = Hypersphere(1).extrinsic_to_intrinsic_coords(on_X).ravel()
        # X_I = Hypersphere(1).extrinsic_to_intrinsic_coords(X).ravel()
        # diff = on_X_I[None, :] - X_I[:, None]  # (N, len(x_grid))
        # hat_f = np.real(np.exp(1j * k * diff[None, :, :]).sum(axis=(0,1))) / (np.sqrt(2*np.pi) * n_samples)
        # hat_grad_f = np.real((1j * k * np.exp(1j * k * diff[None, :, :])).sum(axis=(0,1))) / (np.sqrt(2*np.pi) * n_samples)

# alternative for S2
        # poly = sp.special.legendre(0)
        # poly_deriv = poly.deriv()
        # for m in range(1,M):
        #     poly += (2*m+1)*sp.special.legendre(m)
        #     poly_deriv += (2*m+1)*sp.special.legendre(m).deriv()

        # hat_f = np.zeros(shape=on_X.shape[0])
        # hat_grad_f = np.zeros(shape=(on_X.shape[0],3))
        # for i in range(X.shape[0]):  
        #     dot = on_X @ X[i,:]
        #     hat_f += poly(dot)
        #     hat_grad_f += np.outer(poly_deriv(dot), X[i,:])
        
        # hat_f /= (4*np.pi*X.shape[0])
        # hat_grad_f /= (4*np.pi*X.shape[0])

# alternative for S3
        # pairwise_dists = np.array([[SO3.metric.dist(x, y) for y in on_X] for x in X])
        # cos_half_dists = np.cos(pairwise_dists / 2)

        # p_vals = np.array([sp.special.chebyu(2*m, monic=False)(cos_half_dists) for m in m_vals])
        # p_deriv = np.array([sp.special.chebyu(2*m).deriv()(cos_half_dists) for m in m_vals])
        # hat_f = (weights[:, None, None] * p_vals).sum(axis=(0, 1))
        # hat_grad_f = np.einsum('m,mij,ikl->jkl', weights, p_deriv, X)



def kernel_density_estimate(manifold_type, X, M, on_X):
    """
    Kernel density estimate on manifolds.

    Parameters
    ----------
    manifold_type : str
        'S1' or 'S2' for circle or sphere.
    M : int
        Bandwidth parameter (controls smoothing). Larger M = more smoothing.
    X : np.ndarray
        Data points on the manifold in extrinsic coordinates of shape
        (n_samples, dim) where dim=2 for S1, dim=3 for S2.
    on_X : np.ndarray, optional
        Points where to evaluate the density estimate in extrinsic coordinates.
        If None, evaluates on the data points X.
        
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
    
    # Bandwidth (concentration parameter for von Mises-Fisher kernel)
    kappa = M
    
    # Evaluation points
    if on_X is None:
        grid = on_X = X
    else:
        grid = on_X
    
    n_grid = grid.shape[0]
    
    if manifold_type == 'S1':
        # Circle case: von Mises kernel
        # Normalize data points
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        grid_norm = grid / np.linalg.norm(grid, axis=1, keepdims=True)
        
        hat_f = np.zeros(n_grid)
        hat_grad_f = np.zeros((n_grid, 2))
        
        for i in range(n_grid):
            # Compute inner products (cosine of geodesic distance)
            inner_prods = X_norm @ grid_norm[i]
            
            # von Mises kernel: exp(kappa * cos(distance))
            weights = np.exp(kappa * inner_prods)
            
            # Density estimate
            hat_f[i] = np.mean(weights)
            
            # Gradient estimate
            # Gradient direction is sum of weighted differences on tangent space
            tangent_vecs = X_norm - inner_prods[:, np.newaxis] * grid_norm[i]
            hat_grad_f[i] = kappa * np.mean(
                weights[:, np.newaxis] * tangent_vecs, axis=0
            )
        
        # Normalize density (approximate normalization constant)
        normalization = 1.0 / (2 * np.pi * np.i0(kappa))
        hat_f *= normalization
        hat_grad_f *= normalization
        
    elif manifold_type == 'S2':
        # Sphere case: von Mises-Fisher kernel
        # Normalize data points
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        grid_norm = grid / np.linalg.norm(grid, axis=1, keepdims=True)
        
        hat_f = np.zeros(n_grid)
        hat_grad_f = np.zeros((n_grid, 3))
        
        for i in range(n_grid):
            # Compute inner products (cosine of geodesic distance)
            inner_prods = X_norm @ grid_norm[i]
            
            # von Mises-Fisher kernel: exp(kappa * cos(distance))
            weights = np.exp(kappa * inner_prods)
            
            # Density estimate
            hat_f[i] = np.mean(weights)
            
            # Gradient estimate
            # Gradient direction is sum of weighted tangent vectors
            tangent_vecs = X_norm - inner_prods[:, np.newaxis] * grid_norm[i]
            hat_grad_f[i] = kappa * np.mean(
                weights[:, np.newaxis] * tangent_vecs, axis=0
            )
        
        # Normalize density (approximate normalization constant)
        normalization = kappa / (4 * np.pi * np.sinh(kappa))
        hat_f *= normalization
        hat_grad_f *= normalization
        
    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")
    
    return grid, hat_f, hat_grad_f






        # def D(m):
        #     return lambda z : 1 / np.sqrt(2 * np.pi) *np.sin((m + .5)*z)/ np.sin(z/2)
        # diffs = circle.metric.dist_broadcast(X, on_X)

        # hat_f = np.mean(D(M)(diffs), axis=0)
        
>>>>>>> f7c80cf7d36e48d2656bd3e47eace04afa3fcb5c
