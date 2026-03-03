

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






# def kernel_density_estimate(manifold_type, X, M, on_X):
#     """
#     Kernel density estimate on manifolds.

#     Parameters
#     ----------
#     manifold_type : str
#         'S1' or 'S2' for circle or sphere.
#     M : int
#         Bandwidth parameter (controls smoothing). Larger M = more smoothing.
#     X : np.ndarray
#         Data points on the manifold in extrinsic coordinates of shape
#         (n_samples, dim) where dim=2 for S1, dim=3 for S2.
#     on_X : np.ndarray, optional
#         Points where to evaluate the density estimate in extrinsic coordinates.
#         If None, evaluates on the data points X.
        
#     Returns
#     -------
#     grid : np.ndarray
#         Grid points in extrinsic coordinates where the density is evaluated.
#     hat_f : np.ndarray
#         Estimated density values at the grid points.
#     hat_grad_f : np.ndarray
#         Estimated gradient of the density at the grid points.
#     """
#     n = X.shape[0]
    
#     # Bandwidth (concentration parameter for von Mises-Fisher kernel)
#     kappa = M
    
#     # Evaluation points
#     if on_X is None:
#         grid = on_X = X
#     else:
#         grid = on_X
    
#     n_grid = grid.shape[0]
    
#     if manifold_type == 'S1':
#         # Circle case: von Mises kernel
#         # Normalize data points
#         X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
#         grid_norm = grid / np.linalg.norm(grid, axis=1, keepdims=True)
        
#         hat_f = np.zeros(n_grid)
#         hat_grad_f = np.zeros((n_grid, 2))
        
#         for i in range(n_grid):
#             # Compute inner products (cosine of geodesic distance)
#             inner_prods = X_norm @ grid_norm[i]
            
#             # von Mises kernel: exp(kappa * cos(distance))
#             weights = np.exp(kappa * inner_prods)
            
#             # Density estimate
#             hat_f[i] = np.mean(weights)
            
#             # Gradient estimate
#             # Gradient direction is sum of weighted differences on tangent space
#             tangent_vecs = X_norm - inner_prods[:, np.newaxis] * grid_norm[i]
#             hat_grad_f[i] = kappa * np.mean(
#                 weights[:, np.newaxis] * tangent_vecs, axis=0
#             )
        
#         # Normalize density (approximate normalization constant)
#         normalization = 1.0 / (2 * np.pi * np.i0(kappa))
#         hat_f *= normalization
#         hat_grad_f *= normalization
        
#     elif manifold_type == 'S2':
#         # Sphere case: von Mises-Fisher kernel
#         # Normalize data points
#         X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
#         grid_norm = grid / np.linalg.norm(grid, axis=1, keepdims=True)
        
#         hat_f = np.zeros(n_grid)
#         hat_grad_f = np.zeros((n_grid, 3))
        
#         for i in range(n_grid):
#             # Compute inner products (cosine of geodesic distance)
#             inner_prods = X_norm @ grid_norm[i]
            
#             # von Mises-Fisher kernel: exp(kappa * cos(distance))
#             weights = np.exp(kappa * inner_prods)
            
#             # Density estimate
#             hat_f[i] = np.mean(weights)
            
#             # Gradient estimate
#             # Gradient direction is sum of weighted tangent vectors
#             tangent_vecs = X_norm - inner_prods[:, np.newaxis] * grid_norm[i]
#             hat_grad_f[i] = kappa * np.mean(
#                 weights[:, np.newaxis] * tangent_vecs, axis=0
#             )
        
#         # Normalize density (approximate normalization constant)
#         normalization = kappa / (4 * np.pi * np.sinh(kappa))
#         hat_f *= normalization
#         hat_grad_f *= normalization
        
#     else:
#         raise ValueError(f"Unknown manifold type: {manifold_type}")
    
#     return grid, hat_f, hat_grad_f






#         # def D(m):
#         #     return lambda z : 1 / np.sqrt(2 * np.pi) *np.sin((m + .5)*z)/ np.sin(z/2)
#         # diffs = circle.metric.dist_broadcast(X, on_X)

#         # hat_f = np.mean(D(M)(diffs), axis=0)
        