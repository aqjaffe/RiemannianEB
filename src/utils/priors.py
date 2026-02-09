import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
circle = Hypersphere(dim=1)
sphere = Hypersphere(dim=2)



def S1_multimodal_prior(n_samples, G_params):
    '''
    Sample from a multimodal prior on S^1.
    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    G_params : dict
        Dictionary containing the parameters of the prior:
        - 'tau2': variance parameter for the Riemannian normal distributions.
        - 'num_modes': number of modes in the mixture.
    Returns
    -------
    samples : array-like, shape (n_samples, 2)
        Samples drawn from the multimodal prior on S^1.
    '''
    tau2, num_modes = G_params['tau2'], G_params['num_modes']
    circle = Hypersphere(1)
    angles = np.mod( np.linspace(0, 2*np.pi, num_modes, endpoint=False) + np.pi/12, 2*np.pi)
    means = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    classes = np.random.randint(0, num_modes, n_samples)
    samples = np.zeros((n_samples, 2))
    for k in range(num_modes):
        idx = (classes == k); nk = idx.sum()
        if nk > 0:
            samples[idx] = circle.random_riemannian_normal(means[k], 1. / tau2, nk)
    return samples


def S2_multimodal_prior(n_samples,G_params):
    '''
    Sample from a multimodal prior on S^2.
    Parameters
    n_samples : int
        Number of samples to draw.
    G_params : dict
        Dictionary containing the parameters of the prior:
        - 'num_modes': number of modes in the mixture.
        - 'kappa': concentration parameter for the von Mises-Fisher distributions.
    Returns
    -------
    Theta : array-like, shape (n_samples, 3)
        Samples drawn from the multimodal prior on S^2.
    '''
    num_modes, kappa  = G_params['num_modes'], G_params['kappa']
    if num_modes == 1:
        mus = np.array([[0, 0, 1]])
    elif num_modes == 2:
        mus = np.array([[0, 0, 1], [0, 0, -1]])
    else:
        indices = np.arange(num_modes)
        phi = np.arccos(1 - 2 * (indices + 0.5) / num_modes)
        theta = np.pi * (1 + 5**0.5) * indices
        
        mus = np.stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ], axis=-1)
    
    Theta = np.vstack([ sphere.random_von_mises_fisher(kappa=kappa, mu=mu, n_samples=n_samples // num_modes) for mu in mus])
    if len(Theta) < n_samples:
        extra_samples = sphere.random_von_mises_fisher(kappa=kappa, mu=mus[0], n_samples=n_samples - len(Theta))
        Theta = np.vstack([Theta, extra_samples])
    return Theta