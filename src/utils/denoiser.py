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


    # delta = manifold.projection(manifold.metric.exp(tangent_vecs, X_to_denoise))
    
    # if manifold_type == 'S1':
    #     manifold = Hypersphere(1)
    #     hat_score = hat_grad_f /  np.maximum(hat_f.reshape(-1, 1), rho)
    #     # X_complex = X_to_denoise[:, 0] + 1j * X_to_denoise[:, 1]
    #     # delta_complex = X_complex * np.exp(1j * stepsize * sigma2 * hat_score)
    #     # delta = np.column_stack([delta_complex.real, delta_complex.imag])


    # if manifold_type == 'S2':
    #     manifold = Hypersphere(2)
    #     hat_score = hat_grad_f / np.maximum(hat_f[:, np.newaxis], rho)

    #     # hat_score = hat_grad_f / np.maximum(hat_f, rho).reshape(-1,1)
    #     # delta = np.zeros((X.shape[0],3))
    #     # for i in range(X.shape[0]):
    #     #     x_ = X[i,:].reshape(-1,1)
    #     #     v = X[i,:] + sigma2*(np.eye(3) - x_@ x_.T)@hat_score[i,:]
    #     #     delta[i,:] = S2.metric.exp(v, X[i,:])

    
    # if manifold_type == 'SO3':
    #     manifold = SpecialOrthogonal(n=3)
    #     hat_score = hat_grad_f / np.maximum(hat_f[:, np.newaxis, np.newaxis], rho)


    # if manifold_type == 'S1':
    #     S1 = Hypersphere(1)
    #     hat_score = hat_grad_f / np.maximum(hat_f.ravel(), rho)
    #     X_complex = X_to_denoise[:, 0] + 1j * X_to_denoise[:, 1]
    #     delta_complex = X_complex * np.exp(1j * sigma2 * hat_score)
    #     delta = np.column_stack([delta_complex.real, delta_complex.imag])

    # if manifold_type == 'S2':
    #     S2 = Hypersphere(2)
    #     hat_score = hat_grad_f / np.maximum(hat_f[:, np.newaxis], rho)
    #     projections = np.eye(3) - np.einsum('ij,ik->ijk', X_to_denoise, X_to_denoise)
    #     v = X_to_denoise + np.einsum('ijk,ik->ij', projections, hat_score)
    #     delta =  S2.metric.exp(sigma2 * v, X_to_denoise)
    
    # if manifold_type == 'SO3':
    #     SO3 = SpecialOrthogonal(n=3)
    #     hat_score = hat_grad_f / np.maximum(hat_f[:, np.newaxis, np.newaxis], rho)
    #     tangent_vecs = X_to_denoise + step * sigma2 * hat_score
    #     delta = SO3.projection(SO3.metric.exp(tangent_vecs, X_to_denoise))
    # return delta    




    #     tangents = sigma2*np.array([ v - (np.dot(v, x))*x for v, x in zip(hat_score, X_to_denoise) ]) 
    #     norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    #     return  np.cos(norms) * X_to_denoise + np.sin(norms) * (tangents / norms)  

    #   if manifold_type == 'S1':
    #     manifold = Hypersphere(1)
    #     hat_score = hat_grad_f / np.maximum(hat_f, rho)

    # if manifold_type == 'S2':
    #     manifold = Hypersphere(2)
    #     hat_score = hat_grad_f / np.maximum(hat_f[:, np.newaxis], rho)

    # if manifold_type == 'SO3':
    #     manifold = SpecialOrthogonal(n=3)
    #     hat_score = hat_grad_f / np.maximum(hat_f[:, np.newaxis, np.newaxis], rho)

    # print(X_to_denoise.shape)
    # print(hat_score.shape)
    # tangent_vecs = X_to_denoise + step * sigma2 * hat_score
    # delta = manifold.projection(manifold.metric.exp(tangent_vecs, X_to_denoise))

    # return delta    
