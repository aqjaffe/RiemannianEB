import numpy as np
from tqdm import tqdm
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.frechet_mean import FrechetMean
from .helpers import get_manifold
from .priors import *

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