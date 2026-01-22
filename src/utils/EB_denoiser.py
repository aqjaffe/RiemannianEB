import numpy as np
from .LB_density import density_estimate
from scipy.interpolate import RegularGridInterpolator
from geomstats.geometry.hypersphere import Hypersphere


def denoiser(manifold_type, X, M, rho, sigma2, X_to_denoise):
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
        delta: Denoised points on the manifold, in intrinsic coordinates
    """
    _, hat_f, hat_grad_f = density_estimate(manifold_type, X, M, X_to_denoise)

    if manifold_type == 'S1':
        hat_score = hat_grad_f / np.maximum(hat_f.ravel(), rho)
        X_to_denoise_I = Hypersphere(1).extrinsic_to_intrinsic_coords(X_to_denoise).ravel()
        delta_I =  X_to_denoise_I + sigma2 * hat_score
        return np.asarray([np.cos(delta_I),np.sin(delta_I)]).T
    
    if manifold_type == 'S2':
        hat_score = hat_grad_f / np.maximum(hat_f[:, np.newaxis], rho)
        tangents = sigma2*np.array([ v - (np.dot(v, x))*x for v, x in zip(hat_score, X_to_denoise) ]) 
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        return  np.cos(norms) * X_to_denoise + np.sin(norms) * (tangents / norms)  
    
    
        # hat_score = hat_grad_f / np.maximum(hat_f[:, np.newaxis], rho)
        # # delta = X_to_denoise +  sigma2 * hat_score; delta /= np.linalg.norm(delta, axis=1, keepdims=True)
        
        #  # compute denoiser
        # delta = np.zeros((X_to_denoise.shape[0],3))
        # for j in range(X_to_denoise.shape[0]):
        #     x_ = X_to_denoise[j,:].reshape(-1,1)
        #     v = X_to_denoise[j,:] + (np.eye(3) - x_@ x_.T)@hat_score[j,:]
        #     delta[j,:] = Hypersphere(2).metric.exp(sigma2*v, X_to_denoise[j,:])
    
        # return delta 
