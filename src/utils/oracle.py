import numpy as np
from tqdm import tqdm
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.frechet_mean import FrechetMean

def weighted_circular_mean(thetas, weights):
    """
    thetas: angles in radians, shape (N,)
    weights: shape (N,)
    """
    weights = weights / np.sum(weights)
    sin_sum = np.sum(weights * np.sin(thetas))
    cos_sum = np.sum(weights * np.cos(thetas))
    avrg_angle = np.arctan2(sin_sum, cos_sum)
    return  np.array([np.cos(avrg_angle), np.sin(avrg_angle)])


def oracle_denoiser(manifold_type, X, Theta, bandwidth, X_to_denoise):
    """
    Oracle approximate Bayes (Tweedie-style) denoiser using
    distance-weighted Fréchet means on a Riemannian manifold.
    Parameters:
    - manifold_type: 'S1', 'S2', or 'SO3'
    - X: shape (N, D) - Noisy samples on the manifold
    - Theta: shape (N, D) - True parameters corresponding to X
    - bandwidth: scalar - Neighborhood radius for weighting
    - X_to_denoise: shape (M, D) - Noisy points to denoise
    """
    if manifold_type == 'S1':  
        manifold = Hypersphere(1)
    elif manifold_type == 'S2':
        manifold = Hypersphere(2)
    elif manifold_type == 'SO3':
        manifold = SpecialOrthogonal(n=3)
    else:
        raise ValueError( "Unsupported manifold type. Supported types are 'S1', 'S2', and 'SO3'." )

    denoised = []
    for i in tqdm(range(X_to_denoise.shape[0]), desc="Denoising"):
        dists = manifold.metric.dist_broadcast(X, X_to_denoise[i])
        mask = dists < bandwidth # Neighborhood
        if not np.any(mask):
            raise ValueError( "No points found within the bandwidth. Consider increasing the bandwidth.")

        nearby_Thetas = Theta[mask]
        nearby_dists = dists[mask]
        weights = np.exp(-(nearby_dists ** 2) / (2 * bandwidth ** 2))  # Distance-based weights (Gaussian kernel)
        weights /= np.sum(weights)

        if manifold_type == 'S1':
            angles = np.arctan2(nearby_Thetas[:, 1], nearby_Thetas[:, 0])
            denoised.append(weighted_circular_mean(angles, weights))
        else:
            mean = FrechetMean(manifold)
            mean.fit(nearby_Thetas, weights=weights)
            denoised.append(mean.estimate_)

    return np.array(denoised)
