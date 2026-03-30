import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
circle = Hypersphere(dim=1)
sphere = Hypersphere(dim=2)


def get_G_class(manifold_type, sampler, name, params):
    class G:
        def __init__(self):
            self.name = name
            self.params = params

        def sample(self, n_samples):
            if params is not None:
                return sampler(manifold_type, n_samples, **self.params)
            else:
                return sampler(manifold_type, n_samples)
    return G()

def uniform_sampler(manifold_type, num_samples):
    if manifold_type == 'S1':
        manifold = Hypersphere(1)
    elif manifold_type == 'S2':
        manifold = Hypersphere(2)
    elif manifold_type == 'SO3':
        manifold = SpecialOrthogonal(n=3)
    else: 
        raise ValueError( "Unsupported manifold type. Supported types are 'S1', 'S2', and 'SO3'." )
    return manifold.random_uniform(num_samples)


def dirac_sampler(manifold_type, num_samples, n_points=1):
    def uniform_points_(manifold_type: str, N: int):
        if manifold_type == "S1":
            theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
            return np.stack((np.cos(theta), np.sin(theta)), axis=1)
        elif manifold_type == "S2":
            # Fibonacci sphere
            points = []
            phi = (1 + 5**0.5) / 2
            for i in range(N):
                z = 1 - 2 * (i + 0.5) / N
                r = np.sqrt(1 - z * z)
                theta = 2 * np.pi * i / phi
                points.append((r * np.cos(theta), r * np.sin(theta), z))
            return np.array(points)
        else:
            raise ValueError("manifold must be 'S1' or 'S2'")

    if manifold_type not in ("S1", "S2"):
        raise ValueError("Unsupported manifold type. Supported types are 'S1', 'S2'.")

    points = uniform_points_(manifold_type, int(n_points))  # (n_points, D)
    # Sample indices from support points (with replacement).
    idx = np.random.randint(0, points.shape[0], size=int(num_samples))
    return points[idx]


def equator_sampler(manifold_type, num_samples, tau2=0.01):
    """
    Sample near an equator of S2 (great circle) with small Gaussian perturbation,
    then project back to the sphere.

    Parameters
    ----------
    manifold_type : str
        Must be 'S2'.
    num_samples : int
        Number of samples.
    tau2 : float
        Standard deviation of ambient Gaussian noise before projection.
    Returns
    -------
    X : ndarray, shape (num_samples, 3)
        Points on S2 concentrated near the chosen equator.
    """
    if manifold_type != "S2":
        raise ValueError("equator_sampler is only implemented for manifold_type='S2'.")
    theta = np.random.uniform(0.0, 2.0 * np.pi, size=num_samples)
    c, s = np.cos(theta), np.sin(theta)
    X = np.stack([c, s, np.zeros(num_samples)], axis=1)
    if tau2 and tau2 > 0:
        X = Hypersphere(2).random_riemannian_normal(X, 1 / tau2, num_samples)
    return X


def multimodal_sampler(manifold_type, n_samples, tau2, num_modes):
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