import numpy as np
from tqdm import tqdm
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.frechet_mean import FrechetMean
from .helpers import get_manifold
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


def oracle_denoiser(manifold_type, oracle_samples, sigma2, X_to_denoise, G = None ):
        '''
        Oracle denoiser using the score function estimated from samples of the generative model.
        Parameters:
        - manifold_type: 'S1', 'S2', or 'SO3'
        - num_oracle_samples: Number of samples to use for estimating the score function
        - G: Generative model with a .sample(num_samples) method that returns samples on the manifold
        - X_to_denoise: shape (M, D) - Noisy points to denoise
        - sigma2: Noise variance
        Returns:
        - oracle_delta: shape (M, D) - Denoised points on the manifold in extrinsic coordinates
        '''

        manifold = get_manifold(manifold_type)    
        if np.isscalar(oracle_samples):
            if G is None:
                raise ValueError("G must be provided if oracle_samples is a scalar.") 
            G_samples = G(oracle_samples)
        else: 
            G_samples = oracle_samples


        # ------ Oracle score estimation
        oracle_score = []
        for x in X_to_denoise:
        # for x in tqdm(X_to_denoise, desc="Denoising", leave=False):
            dists = manifold.metric.dist(x, G_samples)      # shape (N,)
            logs  = manifold.metric.log(x, G_samples)       # shape (N, dim)
            weights = np.exp(-(dists ** 2) / (2 * sigma2))
            oracle_score.append(
                - (1 / sigma2) * (weights[:, None] * logs).sum(axis=0) / weights.sum()
            )
        oracle_score = np.array(oracle_score)
        # ------ Oracle denoising
        oracle_tangent_vecs = X_to_denoise +  sigma2 * oracle_score
        oracle_delta =  manifold.metric.exp( oracle_tangent_vecs, X_to_denoise)
        return oracle_delta


def oracle_bayes__kernel(manifold_type, num_oracle_samples, sigma2, oracle_bandwidth, X_to_denoise, G_sampler, verbose = False):
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
    
    manifold = get_manifold(manifold_type)

    Theta = G_sampler(num_oracle_samples)
    X = manifold.random_riemannian_normal(Theta, 1./sigma2, num_oracle_samples)

    denoised = []
    dists_all = manifold.metric.dist_broadcast(X, X_to_denoise)
    for i in tqdm(range(X_to_denoise.shape[0]), desc="Denoising", leave=False):
        dists = dists_all[:, i]

        # Ensure we have at least one neighbor; if not, increase bandwidth until we do.
        bandwidth = oracle_bandwidth
        mask = dists < bandwidth  # Neighborhood
        while not np.any(mask):
            bandwidth *= 2.0
            mask = dists < bandwidth

        nearby_Thetas = Theta[mask]
        nearby_dists = dists[mask]
        weights = np.exp(-(nearby_dists ** 2) / (2 * oracle_bandwidth))  # Distance-based weights (Gaussian kernel)
        weights /= np.sum(weights)

        if manifold_type == 'S1':
            angles = np.arctan2(nearby_Thetas[:, 1], nearby_Thetas[:, 0])
            denoised.append(weighted_circular_mean(angles, weights))
        else:
            mean = FrechetMean(manifold)
            mean.fit(nearby_Thetas, weights=weights)
            denoised.append(mean.estimate_)

    return np.array(denoised)


def oracle_denoiser__kernel(manifold_type, num_oracle_samples, sigma2, oracle_bandwidth, X_to_denoise, G_sampler, verbose = False):
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

    manifold = get_manifold(manifold_type)

    Theta = G_sampler(num_oracle_samples)
    X = manifold.random_riemannian_normal(Theta, 1./sigma2, num_oracle_samples)

    denoised = []
    dists_all = manifold.metric.dist_broadcast(X, X_to_denoise)
    for i in tqdm(range(X_to_denoise.shape[0]), desc="Denoising", leave=False):
        dists = dists_all[:, i]

        # Ensure we have at least one neighbor; if not, increase bandwidth until we do.
        bandwidth = oracle_bandwidth
        mask = dists < bandwidth  # Neighborhood
        while not np.any(mask):
            bandwidth *= 2.0
            mask = dists < bandwidth

        nearby_Thetas = Theta[mask]
        nearby_dists = dists[mask]
        weights = np.exp(-(nearby_dists ** 2) / (2 * oracle_bandwidth ))  # Distance-based weights (Gaussian kernel)
        weights /= np.sum(weights)

        base_points = np.repeat(X_to_denoise[i][None, :], nearby_Thetas.shape[0], axis=0)
        nearby_logs = manifold.metric.log(point=nearby_Thetas, base_point=base_points)

        mean_log = (weights[:, None] * nearby_logs).sum(axis=0)
        denoised.append(manifold.metric.exp(tangent_vec=mean_log, base_point=X_to_denoise[i]))
    return np.array(denoised)