from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.product_manifold import ProductManifold
import numpy as np
from .priors import *

def get_obs_from_G(manifold_type, G, sigma2, n_obs):
    if hasattr(G, 'sample'):
        Theta = G.sample(n_obs)
    else:
        Theta = G(n_obs)
    manifold = get_manifold(manifold_type)
    return manifold.random_riemannian_normal(Theta, 1/sigma2, n_obs)

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


def get_G_sampler_ls_from_params(params):
    G_sampler_ls = []
    manifold_type = params['manifold_type'] 
    for Gname, Gparams in zip(params['G_names'], params['G_params']):
        if 'modal' in Gname:
            G_sampler_ls.append(get_G_class(manifold_type, multimodal_sampler, Gname, Gparams))
        elif 'uniform' in Gname:
            G_sampler_ls.append(get_G_class(manifold_type, uniform_sampler, Gname, Gparams))
        elif 'equator' in Gname:
            G_sampler_ls.append(get_G_class(manifold_type, equator_sampler, Gname, Gparams))
        elif 'dirac' in Gname:
            G_sampler_ls.append(get_G_class(manifold_type, dirac_sampler, Gname, Gparams))
        else:
            raise ValueError('Unknown Gname: ' + Gname)
    if len(G_sampler_ls) != len(params['G_names']):
        raise ValueError('G_sampler_ls length does not match G_names length')
    return G_sampler_ls

def parse_np_array(s):
    return np.fromstring(s.strip('[]'), sep=' ')

def sq_loss(manifold, X, delta):
    return ( manifold.metric.dist_broadcast(X, delta) ** 2).mean()

def get_manifold(manifold_type):
    if manifold_type == 'S1':  
        manifold = Hypersphere(1)
    elif manifold_type == 'S2':
        manifold = Hypersphere(2)
    elif manifold_type == 'SO3':
        manifold = SpecialOrthogonal(n=3)
    elif manifold_type == 'T2':
        manifold = ProductManifold([Hypersphere(1), Hypersphere(1)])
    else:
        raise ValueError( "Unsupported manifold type. Supported types are 'S1', 'S2', 'SO3' and 'T2'." )
    return manifold 

def uniform_points(manifold_type: str, N: int):
    if manifold_type == "S1":
        # Equally spaced angles
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        return np.stack((x, y), axis=1)

    elif manifold_type == "S2":
        # Fibonacci sphere 
        points = []
        phi = (1 + 5**0.5) / 2  # golden ratio
        for i in range(N):
            z = 1 - 2*(i + 0.5)/N
            r = np.sqrt(1 - z*z)
            theta = 2 * np.pi * i / phi
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append((x, y, z))
        return np.array(points)
    else: raise ValueError("manifold must be 'S1' or 'S2'")


    