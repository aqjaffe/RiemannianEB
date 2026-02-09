import numpy as np
from tqdm import tqdm
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.frechet_mean import FrechetMean

from .priors import *

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

def oracle_bayes(manifold_type, num_oracle_samples, G_params, sigma2, oracle_bandwidth, X_to_denoise):
    """
    Oracle Bayes denoiser using distance-weighted Fréchet means on a Riemannian manifold.
    Parameters:
    - manifold_type: 'S1', 'S2', or 'SO3'
    - num_oracle_samples: Number of samples to use for the oracle denoiser
    - G_params: Parameters of the generative model (e.g., distribution parameters)
    - sigma2: Noise variance
    - oracle_bandwidth: Bandwidth for neighborhood weighting
    - X: shape (M, D) - Noisy points to denoise
    Returns:
    - denoised: shape (M, D) - Denoised points on the manifold
    """
    if manifold_type == 'S1':  
        manifold = Hypersphere(1)
        Theta = S1_multimodal_prior(num_oracle_samples, G_params)
 
    elif manifold_type == 'S2':
        manifold = Hypersphere(2)
        Theta = S2_multimodal_prior(num_oracle_samples, G_params)

    elif manifold_type == 'SO3':
        manifold = SpecialOrthogonal(n=3)
        Theta = SO3_multimodal_prior(num_oracle_samples, G_params)
    else:
        raise ValueError( "Unsupported manifold type. Supported types are 'S1', 'S2', and 'SO3'." )

    X = manifold.random_riemannian_normal(Theta, 1./np.sqrt(sigma2), num_oracle_samples)


    denoised = []
    for i in tqdm(range(X_to_denoise.shape[0]), desc="Denoising", leave=False):
        dists = manifold.metric.dist_broadcast(X, X_to_denoise[i])

        # Ensure we have at least one neighbor; if not, increase bandwidth until we do.
        bandwidth = oracle_bandwidth
        mask = dists < bandwidth  # Neighborhood
        while not np.any(mask):
            print(f"No points found within bandwidth {bandwidth:.4f}. Increasing bandwidth.")
            bandwidth *= 2.0
            mask = dists < bandwidth

        nearby_Thetas = Theta[mask]
        nearby_dists = dists[mask]
        weights = np.exp(-(nearby_dists ** 2) / (2 * oracle_bandwidth ** 2))  # Distance-based weights (Gaussian kernel)
        weights /= np.sum(weights)

        if manifold_type == 'S1':
            angles = np.arctan2(nearby_Thetas[:, 1], nearby_Thetas[:, 0])
            denoised.append(weighted_circular_mean(angles, weights))
        else:
            mean = FrechetMean(manifold)
            mean.fit(nearby_Thetas, weights=weights)
            denoised.append(mean.estimate_)

    return np.array(denoised)


def oracle_denoiser(manifold_type, num_oracle_samples, G_params, sigma2, oracle_bandwidth, X_to_denoise):
    """
    Oracle Bayes denoiser using distance-weighted Fréchet means on a Riemannian manifold.
    Parameters:
    - manifold_type: 'S1', 'S2', or 'SO3'
    - num_oracle_samples: Number of samples to use for the oracle denoiser
    - G_params: Parameters of the generative model (e.g., distribution parameters)
    - sigma2: Noise variance
    - oracle_bandwidth: Bandwidth for neighborhood weighting
    - X: shape (M, D) - Noisy points to denoise
    Returns:
    - denoised: shape (M, D) - Denoised points on the manifold
    """
    if manifold_type == 'S1':  
        manifold = Hypersphere(1)
        Theta = S1_multimodal_prior(num_oracle_samples, G_params)
 
    elif manifold_type == 'S2':
        manifold = Hypersphere(2)
        Theta = S2_multimodal_prior(num_oracle_samples, G_params)

    elif manifold_type == 'SO3':
        manifold = SpecialOrthogonal(n=3)
        Theta = SO3_multimodal_prior(num_oracle_samples, G_params)
    else:
        raise ValueError( "Unsupported manifold type. Supported types are 'S1', 'S2', and 'SO3'." )

    X = manifold.random_riemannian_normal(Theta, 1./np.sqrt(sigma2), num_oracle_samples)
    denoised = []
    for i in tqdm(range(X_to_denoise.shape[0]), desc="Denoising", leave=False):
        dists = manifold.metric.dist_broadcast(X, X_to_denoise[i])

        # Ensure we have at least one neighbor; if not, increase bandwidth until we do.
        bandwidth = oracle_bandwidth
        mask = dists < bandwidth  # Neighborhood
        while not np.any(mask):
            print(f"No points found within bandwidth {bandwidth:.4f}. Increasing bandwidth.")
            bandwidth *= 2.0
            mask = dists < bandwidth

        nearby_Thetas = Theta[mask]
        nearby_dists = dists[mask]
        weights = np.exp(-(nearby_dists ** 2) / (2 * oracle_bandwidth ** 2))  # Distance-based weights (Gaussian kernel)
        weights /= np.sum(weights)

        base_points = np.repeat(X_to_denoise[i][None, :], nearby_Thetas.shape[0], axis=0)
        nearby_logs = manifold.metric.log(point=nearby_Thetas, base_point=base_points)

        mean_log = (weights[:, None] * nearby_logs).sum(axis=0)
        denoised.append(manifold.metric.exp(tangent_vec=mean_log, base_point=X_to_denoise[i]))
    return np.array(denoised)