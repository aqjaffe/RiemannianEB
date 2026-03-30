import numpy as np
from .density_estimation import density_estimate
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
import time
from .helpers import get_manifold

def denoiser(manifold_type, X, M, rho, sigma2, X_to_denoise, densityIn=None):
    """
    Perform a denoising step on a Riemannian manifold.
    Args:
        manifold_type: Manifold type
        X: Data points on the manifold in extrinsic coordinates
        M: Parameter for density estimation (number of LB eigenfunctions to project onto)
        rho: Regularization parameter to avoid division by zero
        sigma2: noise variance
        X_to_denoise: Points to denoise in extrinsic coordinates
    
    Returns:
        delta: Denoised points on the manifold in extrinsic coordinates
    """
    manifold = get_manifold(manifold_type)
    if densityIn is None:
        _, hat_f, hat_grad_f = density_estimate(manifold_type, X, M, X_to_denoise)
    else:
        hat_f, hat_grad_f = densityIn
    hat_score = hat_grad_f / np.maximum(hat_f.reshape(hat_f.shape + (1,) * (hat_grad_f.ndim - 1)), rho)
    tangent_vecs = X_to_denoise + sigma2 * hat_score
    delta =  manifold.metric.exp( tangent_vecs, X_to_denoise)
    return delta    

