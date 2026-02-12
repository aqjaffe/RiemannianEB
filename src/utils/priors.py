import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
circle = Hypersphere(dim=1)
sphere = Hypersphere(dim=2)

def uniform_sampler(num_samples, manifold_type):
    if manifold_type == 'S1':
        return circle.random_uniform(num_samples)
    elif manifold_type == 'S2':
        return sphere.random_uniform(num_samples)
    else:
        raise NotImplementedError(f"Uniform sampling for {manifold_type} is not implemented yet.")


def multimodal_sampler(n_samples,manifold_type, G_params):
    '''
    Sample from a multimodal prior on the manifold
    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    G_params : dict
        Dictionary containing the parameters of the prior:
        - 'num_modes': number of modes in the mixture.
        - 'tau2': variance parameter for the Riemannian normal distributions.
    Returns
    -------
    samples : array-like, shape (n_samples, 2)
        Samples drawn from the multimodal prior on S^1.
    '''
    if manifold_type == 'S1':
        tau2, num_modes = G_params['tau2'], G_params['num_modes']
        angles = np.mod( np.linspace(0, 2*np.pi, num_modes, endpoint=False) + np.pi/12, 2*np.pi)
        means = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        classes = np.random.randint(0, num_modes, n_samples)
        samples = np.zeros((n_samples, 2))
        for k in range(num_modes):
            idx = (classes == k); nk = idx.sum()
            if nk > 0:
                samples[idx] = circle.random_riemannian_normal(means[k], 1. / tau2, nk)
        return samples
    
    elif manifold_type == 'S2':
        num_modes, tau2  = G_params['num_modes'], G_params['tau2']
        if num_modes == 1:
            mus = np.array([[0, 1, 0]])
        elif num_modes == 2:
            mus = np.array([[0, 1, 0], [0, -1, 0 ]])
        else:
            indices = np.arange(num_modes)
            phi = np.arccos(1 - 2 * (indices + 0.5) / num_modes)
            theta = np.pi * (1 + 5**0.5) * indices
        
            mus = np.stack([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ], axis=-1)
        
        classes = np.random.randint(0, num_modes, n_samples)
        samples = np.zeros((n_samples, 3))
        for k in range(num_modes):
            idx = (classes == k); nk = idx.sum()
            if nk > 0:
                samples[idx] = sphere.random_riemannian_normal(mus[k], 1. / tau2, nk)
        return samples
    
    else:
        raise NotImplementedError(f"Multimodal prior for {manifold_type} is not implemented yet.")