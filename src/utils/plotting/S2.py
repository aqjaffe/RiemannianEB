
import numpy as np
import matplotlib.pyplot as plt
from geomstats.geometry.hypersphere import Hypersphere
from matplotlib.collections import PolyCollection
import matplotlib.gridspec as gridspec
from ..density_estimation import *

sphere = Hypersphere(dim=2)

plt.rcParams.update({'font.size': 12,
                     'mathtext.fontset': 'stix',
                     'font.family': 'serif',
                     'font.serif':'Palatino'})

def grid_theta_phi_from_X_grid(X_grid):
    Xg = np.asarray(X_grid)
    res = int(np.sqrt(Xg.shape[0]))
    Xg = Xg.reshape(res, res, 3)
    x, y, z = Xg[..., 0], Xg[..., 1], Xg[..., 2]
    z_clipped = np.clip(z, -1.0, 1.0)
    grid_theta = np.arccos(z_clipped)  # colatitude
    grid_phi = np.mod(np.arctan2(y, x), 2 * np.pi)  # [0, 2pi)
    return grid_theta, grid_phi

def S2grid_fib(grid_resolution=50):
    """
    Fibonacci spiral (sunflower) grid on S^2.
    grid_resolution: total number of points N.
    Returns X_grid (N,3), grid_theta (N,), grid_phi (N,)
    """
    N = grid_resolution ** 2  # match original point count convention

    golden_ratio = (1 + np.sqrt(5)) / 2

    i = np.arange(N)

    # Colatitude: arccos maps uniform spacing in cos(theta) → uniform area
    grid_theta = np.arccos(1 - (2 * i + 1) / N)

    # Longitude: golden angle increments
    grid_phi = (2 * np.pi * i / golden_ratio) % (2 * np.pi)

    X_grid = np.stack([
        np.sin(grid_theta) * np.cos(grid_phi),
        np.sin(grid_theta) * np.sin(grid_phi),
        np.cos(grid_theta)
    ], axis=-1)

    return X_grid, grid_theta, grid_phi

def S2grid(grid_resolution=50):
    # Grid on S^2 (theta = colatitude, phi = longitude)
    res_lat = grid_resolution
    res_lon = grid_resolution
    grid_theta, grid_phi = np.meshgrid(
        np.linspace(0, np.pi, res_lat),        # colatitude
        np.linspace(0, 2*np.pi, res_lon)      # longitude
    )
    X_grid = np.stack([
        np.sin(grid_theta) * np.cos(grid_phi),
        np.sin(grid_theta) * np.sin(grid_phi),
        np.cos(grid_theta)
    ], axis=-1).reshape(-1,3)
    return X_grid, grid_theta, grid_phi



def S2scatter(X, ax, color, alpha=.5, s=5, lw=.5, title = None, marker = None):
    '''
    Scatter plot on a Mollweide projection.
    Parameters
    ----------
    X : array-like, shape (n_samples, 3)
        Extrinsic coordinates of points on the sphere.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    color : color  
        Color of the points.
    alpha : float, optional
        Transparency of the points. Default is 0.5.
    s : float, optional 
        Size of the points. Default is 5.
    lw : float, optional    
        Line width of the grid. Default is 0.5.
    '''
    X_sph = sphere.extrinsic_to_spherical(X)
    theta = X_sph[:, 0]  # colatitude
    phi = X_sph[:, 1]    # longitude
    phi_mw = phi - np.pi           # shift longitude from [0, 2π] to [-π, π]
    theta_mw = np.pi/2 - theta     # convert colatitude to latitude [-π/2, π/2]
    if marker is None:
        ax.scatter(phi_mw, theta_mw, s=s, alpha=alpha, color=color)
    ax.grid(True, color='gray', lw=lw)
    if title is not None:
        ax.set_title(title, fontsize = 15)
    return None


def S2plot_quiver(grid, vals, figax= None, scale=1, skip=1, cmap='Greens', cvals = None,
                  equirectangular=False, hide_outlier_quantile=None):
    if figax is None:
        proj = None if equirectangular else 'mollweide'
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': proj})
    else: fig, ax = figax
    grid_resolution = int(grid.shape[0]**0.5)
    grid_theta, grid_phi = grid_theta_phi_from_X_grid(grid)
    
    e_theta = np.stack([
        np.cos(grid_theta) * np.cos(grid_phi),
        np.cos(grid_theta) * np.sin(grid_phi),
        -np.sin(grid_theta)
    ], axis=-1)
    e_phi = np.stack([
        -np.sin(grid_phi),
        np.cos(grid_phi),
        np.zeros_like(grid_phi)
    ], axis=-1)

    vals_reshaped = vals.reshape(grid_resolution, grid_resolution, 3)
    U =  np.sum(vals_reshaped * e_phi,   axis=-1)
    V = -np.sum(vals_reshaped * e_theta, axis=-1)
    if cvals is not None:
        C = cvals.reshape(grid_resolution, grid_resolution)
    else:
        C = np.sqrt(U**2 + V**2)

    lon = (grid_phi - np.pi)[::skip, ::skip]
    lat = (np.pi/2 - grid_theta)[::skip, ::skip]
    C = C[::skip, ::skip]
    if equirectangular:
        # Plain (lon, lat) axes: meridians are straight, so we can draw the
        # geometrically faithful field with angles='xy'. A physical east
        # displacement U maps to a longitude increment U/sin(theta) (east arc
        # length = sin(theta)*dlon); north maps 1:1 to latitude. This gives true
        # tangent directions everywhere with no projection-induced curvature.
        Uq = (U / np.maximum(np.sin(grid_theta), 1e-6))[::skip, ::skip]
        Vq = V[::skip, ::skip]
        quiver_kw = dict(angles='xy', scale_units='xy')
    else:
        # Mollweide axes: meridians are curved, so angles='xy' would bend the
        # arrows along them. angles='uv' instead reads each arrow's direction
        # straight from its (east, north) components in screen space, keeping a
        # meridional score pointing up/down regardless of map position (at the
        # cost of not aligning with the curved graticule away from centre).
        Uq = U[::skip, ::skip]
        Vq = V[::skip, ::skip]
        quiver_kw = dict(angles='uv')

    # Optionally hide upper-outlier arrows (e.g. the 1/sin(theta) blow-up near
    # the poles, or rare huge scores in low-density regions): blank the plotted
    # components above the chosen quantile of arrow length — quiver skips NaNs.
    if hide_outlier_quantile is not None:
        norm = np.sqrt(Uq**2 + Vq**2)
        thresh = np.nanquantile(norm, hide_outlier_quantile)
        outliers = norm > thresh
        Uq = np.where(outliers, np.nan, Uq)
        Vq = np.where(outliers, np.nan, Vq)

    im = ax.quiver(lon, lat, Uq, Vq, C, scale=scale, cmap=cmap, alpha=0.85, **quiver_kw)
    if equirectangular:
        ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi/2, np.pi/2)
        ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
