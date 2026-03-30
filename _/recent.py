


def scoreMatching(manifold_type, X, M_grid, rho_grid = None, eval = 1000, return_scores=False, rho_percentile = 3):  
    if isinstance(eval, (int, np.integer)):
        Xeval = uniform_points(manifold_type, eval)
    elif eval is None:
        Xeval = X
    else:
        Xeval = np.array(eval) 
    eps  = 1e-5
    if rho_grid is None:
        cv_scores = np.zeros(len(M_grid), dtype=float)
        rhos = np.zeros(len(M_grid), dtype=float)
        for ixM, M in enumerate(M_grid):
            _, base_f, grad_f, lap_f = density_estimate(manifold_type, X, M, Xeval, grad=True, laplacian=True)
            grad_sq = np.sum(grad_f ** 2, axis=1)
            rho = np.percentile(base_f[base_f > 0], rho_percentile)
            rhos[ixM] = rho
            hat_f = np.maximum(base_f, rho) + eps
            score_vals = (2 * lap_f / (hat_f)) - (grad_sq / (hat_f**2))
            try: cv_scores[ixM] = np.mean(score_vals) 
            except: cv_scores[ixM] = np.nan 
    else:
        rho_grid = np.array(rho_grid)
        cv_scores = np.zeros((len(M_grid), len(rho_grid)), dtype=float)
        for ixM, M in enumerate(M_grid):
            _, base_f, grad_f, lap_f = density_estimate(manifold_type, X, M, Xeval, grad=True, laplacian=True)
            grad_sq = np.sum(grad_f**2, axis=1)
            for ixRho, rho in enumerate(rho_grid):
                if rho < base_f.mean():
                    hat_f = np.maximum(base_f, rho)
                    score_vals = (2 * lap_f / (hat_f)) - (grad_sq / (hat_f**2))
                    try: cv_scores[ixM, ixRho] = np.mean(score_vals) 
                    except: cv_scores[ixM, ixRho] = np.nan 
                else: cv_scores[ixM, ixRho] = np.nan
    scores, params = get_scores(manifold_type, len(X), cv_scores, M_grid, rho_grid) 
    if rho_grid is None:
        params = {key: (val[0], rhos[val[1]]) for key, val in params.items()}
    if return_scores: return params, scores
    else:return params

