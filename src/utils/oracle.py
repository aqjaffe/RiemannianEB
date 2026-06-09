import numpy as np
from .helpers import get_manifold, uniform_points
from .priors import *
from tqdm import tqdm


def _pairwise_dist(manifold, a, b):
    """Full (len(a), len(b)) Riemannian distance matrix.

    geomstats' ``dist_broadcast`` collapses to *element-wise* distances of
    shape ``(n,)`` whenever ``len(a) == len(b)``, so it can't be relied on to
    produce the cross product. Build the (a, b) grid explicitly instead.
    """
    n_a, n_b, D = len(a), len(b), b.shape[-1]
    pts  = np.repeat(a, n_b, axis=0)                          # (n_a*n_b, D)
    base = np.broadcast_to(b[None], (n_a, n_b, D)).reshape(-1, D)
    return manifold.metric.dist(pts, base).reshape(n_a, n_b)


def _weighted_circular_mean(thetas, weights):
    weights = weights / np.sum(weights)
    sin_sum = np.sum(weights * np.sin(thetas))
    cos_sum = np.sum(weights * np.cos(thetas))
    return np.array([np.cos(np.arctan2(sin_sum, cos_sum)),
                     np.sin(np.arctan2(sin_sum, cos_sum))])


def _eval_bayes_kernel(manifold_type, manifold, Theta, X, X_to_denoise, bandwidth,
                       chunk_size=2000, desc=None):
    """Nadaraya-Watson kernel Bayes estimator at X_to_denoise given (Theta, X) pairs.

    X_to_denoise is processed in chunks of chunk_size to bound peak memory to
    O(N * chunk_size) rather than O(N * M). Pass desc to show a progress bar over
    the denoised points (used for the expensive full-set final evaluation).
    """
    from geomstats.learning.frechet_mean import FrechetMean
    point_bar = (tqdm(total=X_to_denoise.shape[0], desc=desc, unit="pt",
                      leave=False, dynamic_ncols=True) if desc is not None else None)
    denoised = []
    for start in range(0, X_to_denoise.shape[0], chunk_size):
        chunk = X_to_denoise[start:start + chunk_size]
        dists_chunk = _pairwise_dist(manifold, X, chunk)         # (N, chunk_size)
        for i in range(chunk.shape[0]):
            dists = dists_chunk[:, i]
            bw    = bandwidth
            mask  = dists < bw
            while not np.any(mask):
                bw  *= 2.0
                mask = dists < bw
            nearby_Thetas = Theta[mask]
            nearby_dists  = dists[mask]
            weights = np.exp(-(nearby_dists ** 2) / (2 * bandwidth))
            top_idx = np.argsort(weights)[-50:][::-1]

            weights = weights[top_idx]
            nearby_Thetas = nearby_Thetas[top_idx]
            weights /= np.sum(weights)


            if manifold_type == 'S1':
                angles = np.arctan2(nearby_Thetas[:, 1], nearby_Thetas[:, 0])
                denoised.append(_weighted_circular_mean(angles, weights))
            else:
                mean = FrechetMean(manifold)
                mean.fit(nearby_Thetas, weights=weights)
                est = mean.estimate_
                if not np.all(np.isfinite(est)):
                    # FrechetMean occasionally fails to converge (a single bad
                    # point otherwise turns the whole risk into NaN). Fall back
                    # to the highest-weight neighbour — always a valid manifold
                    # point and a reasonable posterior representative.
                    est = nearby_Thetas[np.argmax(weights)]
                denoised.append(est)
            if point_bar is not None:
                point_bar.update(1)
    if point_bar is not None:
        point_bar.close()
    return np.array(denoised)


def oracle_bayes__kernel_cv(manifold_type, oracle_samples, sigma2, X_to_denoise,
                             bw_grid=None, n_splits=5, max_n=10000, n_eval=None,
                             random_state=0):
    """Oracle Bayes kernel denoiser with cross-validated bandwidth.

    oracle_samples is subsampled to at most max_n before CV and evaluation to
    keep the N×M distance matrices in _eval_bayes_kernel memory-bounded.

    n_eval caps how many validation points each fold actually scores. Every
    training point is still used to build the estimator — only the number of
    (expensive) denoising evaluations is reduced, trading a slightly noisier
    bandwidth score for a large speedup. 
    """
    from sklearn.model_selection import KFold

    manifold = get_manifold(manifold_type)
    rng = np.random.default_rng(random_state)

    # Subsample oracle pairs for memory efficiency
    N = len(oracle_samples)
    if N > max_n:
        idx = rng.choice(N, max_n, replace=False)
        oracle_samples = oracle_samples[idx]
        N = max_n

    X_oracle = manifold.random_riemannian_normal(oracle_samples, 1.0 / sigma2, N)

    if bw_grid is None:
        default_bw = np.sqrt(sigma2) * N ** (-1.0 / (manifold.dim + 4))
        bw_grid = default_bw * np.array([1e-4, 1e-2, 1])
    bw_grid = np.atleast_1d(bw_grid)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = np.zeros(len(bw_grid))

    cv_bar = tqdm(total=n_splits * len(bw_grid), desc="Bayes bandwidth CV",
                  unit="fit", leave=False, dynamic_ncols=True)
    for train_idx, val_idx in kf.split(X_oracle):
        # Keep all training pairs; only score a random subset of the fold.
        if n_eval is not None and len(val_idx) > n_eval:
            val_idx = rng.choice(val_idx, n_eval, replace=False)
        Theta_tr, X_tr = oracle_samples[train_idx], X_oracle[train_idx]
        Theta_val, X_val = oracle_samples[val_idx], X_oracle[val_idx]
        for ixh, h in enumerate(bw_grid):
            denoised_val = _eval_bayes_kernel(manifold_type, manifold, Theta_tr, X_tr, X_val, h)
            cv_scores[ixh] += np.mean(
                np.array([manifold.metric.dist(denoised_val[j], Theta_val[j]) ** 2
                          for j in range(len(Theta_val))])
            )
            cv_bar.update(1)
    cv_bar.close()

    cv_scores /= n_splits
    best_h = bw_grid[int(np.argmin(cv_scores))]
    return _eval_bayes_kernel(manifold_type, manifold, oracle_samples, X_oracle,
                              X_to_denoise, best_h, desc="Bayes final denoise")


def oracle_denoiser(manifold_type, oracle_samples, sigma2, X_to_denoise, G=None, n_bins = None):
    oracle_samples = oracle_samples if not np.isscalar(oracle_samples) else G(oracle_samples)
    if n_bins is not None:
        n_bins = min(len(oracle_samples)//10 , n_bins )
    else: n_bins  = len(oracle_samples)//10

    manifold = get_manifold(manifold_type)
    bin_centers = uniform_points(manifold_type, n_bins)          # (n_bins, D)

    # Assign each oracle sample to its nearest bin. Chunk the broadcast so the
    # (N, n_bins) distance matrix never has to exist all at once.
    labels = np.empty(len(oracle_samples), dtype=int)           # (N,)
    chunk_size = 5000
    for start in range(0, len(oracle_samples), chunk_size):
        block = oracle_samples[start:start + chunk_size]
        dists = _pairwise_dist(manifold, block, bin_centers)        # (chunk, n_bins)
        labels[start:start + chunk_size] = np.argmin(dists, axis=1)
    bin_weights = np.bincount(labels, minlength=n_bins) / len(oracle_samples)  # (n_bins,)

    # ------ Score estimation using bin centers.
    # Vectorised over X_to_denoise in chunks: instead of one geomstats log call
    # per point, flatten the (chunk, n_bins) grid and call metric.log once per
    # chunk. logs[i, k] = log_{bin_centers[k]}(x_i), identical to the per-point
    # loop but manifold-generic (any geomstats metric).
    D = bin_centers.shape[1]
    oracle_score = np.empty((len(X_to_denoise), D), dtype=float)
    chunk_size = 1000
    for start in range(0, len(X_to_denoise), chunk_size):
        x = X_to_denoise[start:start + chunk_size]               # (c, D)
        c = len(x)
        dists = _pairwise_dist(manifold, x, bin_centers)         # (c, n_bins)
        pts  = np.repeat(x, n_bins, axis=0)                       # (c*n_bins, D)
        base = np.broadcast_to(bin_centers[None], (c, n_bins, D)).reshape(-1, D)
        logs = manifold.metric.log(pts, base).reshape(c, n_bins, D)
        weights = bin_weights[None, :] * np.exp(-(dists ** 2) / (2 * sigma2))  # (c, n_bins)
        oracle_score[start:start + chunk_size] = (
            -(1 / sigma2) * (weights[..., None] * logs).sum(axis=1) / weights.sum(axis=1, keepdims=True)
        )

    oracle_tangent_vecs = X_to_denoise + sigma2 * oracle_score
    return manifold.metric.exp(oracle_tangent_vecs, X_to_denoise)

