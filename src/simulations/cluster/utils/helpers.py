from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.product_manifold import ProductManifold
import numpy as np

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
