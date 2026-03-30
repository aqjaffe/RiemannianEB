import numpy as np
import matplotlib.pyplot as plt
from geomstats.geometry.hypersphere import Hypersphere
from matplotlib.collections import PolyCollection
import matplotlib.gridspec as gridspec
from ..density_estimation import *


circle = Hypersphere(dim=1)

plt.rcParams.update({'font.size': 12,
                     'mathtext.fontset': 'stix',
                     'font.family': 'serif',
                     'font.serif':'Palatino'})



def S1scatter(X, ax, color, alpha=.5, s=5, jitter_std = 0, title=None,):
    '''
    Scatter plot on a polar projection.
    Parameters
    ----------
    X : array-like, shape (n_samples, 2)
        Extrinsic coordinates of points on the circle.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    color : color  
        Color of the points.
    alpha : float, optional
        Transparency of the points. Default is 0.5.
    s : float, optional 
        Size of the points. Default is 5.
    title : str, optional
        axes title object. Default is None
    jitter_std : float, optional
        Std of radial jitter. If 0, no jitter is applied.
    '''
    theta = np.arctan2(X[:, 1], X[:, 0])

    # Radial jitter clipped to avoid extreme outliers.
    if jitter_std and jitter_std > 0:
        jitter = np.random.uniform(-jitter_std,jitter_std, len(X))
        r = 1.0 + jitter
        r = np.maximum(r, 0.0)  # keep radius non-negative
    else:
        r = np.ones(len(X))

    ax.scatter(theta, r, s=s, alpha=alpha, color=color)
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    ax.set_ylim(bottom=0)
    return None



def S1_histogram(X, nbins, ax, cmap, scale = 1,disk_r = None,title= None):
    angles = np.mod(circle.extrinsic_to_angle(X), 2*np.pi)
    vals, bin_edges = np.histogram(angles, bins=nbins, range=(0, 2*np.pi))
    if vals.max() > 0: vals = vals / vals.max()*scale

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = 1.2*(2*np.pi) / nbins
    bottom = 0.8*scale

    # draw histogram
    bars = ax.bar(centers, vals, width=width, bottom=bottom, edgecolor="white", linewidth=0.5)

    # Color by *normalized* bin height so the colormap is independent of `scale`.
    cm = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
    denom = vals.max() if vals.size and vals.max() > 0 else 1.0
    vals_norm = vals / denom
    for r_norm, bar in zip(vals_norm, bars):
        bar.set_facecolor(cm(r_norm))
        bar.set_alpha(0.8)

    # central white disk
    disk_r = bottom if disk_r is None else disk_r 
    ax.bar(0, disk_r, width=2*np.pi, bottom=0, color="white", edgecolor="none", align="edge", zorder=3)
    # and its boundary
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(theta, np.full_like(theta, disk_r), color="black", linewidth=1.2, zorder=4)
    
    ax.set_ylim(0, 1 + bottom)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    if title is not None:
        ax.set_title(title, fontsize = 15)
    return None
    
    


def S1_score_quiver(X, M, rho, ax, res = 50, title = None):
    f_scale = 0.5
    bottom = 0.8              # match histogram
    grad_scale = 0.15
    r_base = bottom + f_scale * 0.5

    on_X_I = np.linspace(0, 2*np.pi, res)
    on_X = np.column_stack((np.cos(on_X_I), np.sin(on_X_I)))

    grid, hat_f, hat_grad_f = density_estimate('S1', X, M, on_X)
    theta = circle.extrinsic_to_intrinsic_coords(grid).ravel()

    hat_score = hat_grad_f / np.maximum(hat_f.ravel(), rho)
    norm_score = hat_score / np.max(np.abs(hat_score))
    colors = plt.colormaps['Greens'](np.abs(norm_score))

    for i in range(len(theta)):
        dtheta = norm_score[i] * grad_scale
        ax.annotate(
            '',
            xy=(theta[i] + dtheta, r_base),
            xytext=(theta[i] + dtheta / 2, r_base),
            arrowprops=dict(
                arrowstyle='-|>,head_width=0.8,head_length=1.2',
                linewidth=1.5,
                color=colors[i],
            ),
        )

    # --- central disk (same as histogram) ---
    ax.bar(0, bottom, width=2*np.pi, bottom=0,
           color="white", edgecolor="none",
           align="edge", zorder=3)

    theta_ring = np.linspace(0, 2*np.pi, 300)
    ax.plot(theta_ring, np.full_like(theta_ring, bottom),color="black", linewidth=1.2, zorder=4)

    ax.set_ylim(0, 1 + bottom)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    if title is not None:
        ax.set_title(title, fontsize = 15)
    return None


def S1_smooth_histogram(Theta, ax,cmap, nbins = 50,  kappa=9 ,f_scale = 0.3,bottom = 0.105,top = .5,disk_r = 0.1):
    manifold = Hypersphere(1)
    grid_I = np.linspace(0, 2*np.pi, nbins)
    on_X = manifold.intrinsic_to_extrinsic_coords( grid_I[:, None])
    hat_f = kernel_density_estimate("S1", Theta,  on_X, kappa)[1]
    hat_pos_f = np.maximum(hat_f, 0)
    normalised_hat_f = (hat_pos_f - hat_pos_f.min()) / (hat_pos_f.max() - hat_pos_f.min() + 1e-10)
    verts = [[
            (grid_I[i], bottom),
            (grid_I[i], bottom + f_scale * hat_pos_f[i]), (grid_I[i+1], bottom + f_scale * hat_pos_f[i+1]),
            # (grid_I[i],top), (grid_I[i+1], bottom + top),
            (grid_I[i+1], bottom)
        ] for i in range(len(grid_I) - 1)] # Create polygon vertices for each segment
    poly = PolyCollection(verts, facecolors=plt.colormaps[cmap](normalised_hat_f[:-1]), 
                        alpha=0.85, edgecolors='none')
    ax.add_collection(poly)

    ax.set_ylim(0, bottom + hat_f.max()*f_scale)  
    ax.set_yticks([])
    ax.bar(0, disk_r, width=2*np.pi, bottom=0, color="white", edgecolor="none", align="edge", zorder=3)
    ax.plot(grid_I, disk_r*np.ones_like(grid_I), color='black', linewidth=1.2, zorder=4)
    return None


