
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


def S2plot_quiver(grid, vals, figax= None, scale=1, skip=1, cmap='Greens', cvals = None):
    if figax is None:
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': 'mollweide'})
    else: fig, ax = figax
    grid_resolution = int(grid.shape[0]**0.5)
    grid_theta, grid_phi = grid_theta_phi_from_X_grid(grid)

        

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


    im = ax.quiver(
        (grid_phi - np.pi)[::skip, ::skip], (np.pi/2 - grid_theta)[::skip, ::skip],
        U[::skip, ::skip],           V[::skip, ::skip],
        C[::skip, ::skip],
        scale=scale, cmap=cmap, alpha=0.85,       
        angles='xy',
        scale_units='xy',
    )
    ax.grid(True, linestyle='--', alpha=0.4)
    # fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.14)




# def S2plot_quiver(
#     fig, grid_vals, vals,
#     ax = None, 
#     scale=5,
#     skip=1, 
#     title = None,
#     cmap = 'Blues',
#     arrow_width = None, 
#     headwidth = 3, 
#     headlength = 5, 
#     headaxislength = 4,
#     minlength = 0,
#     stick_length = None
# ):
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': 'mollweide'})
        
#     grid, grid_theta, grid_phi = grid_vals
#     grid_resolution = int(grid.shape[0]**.5)

#     grid_phi_mw = (grid_phi - np.pi)       # longitude in [-pi, pi]
#     grid_theta_mw = (np.pi/2 - grid_theta) # latitude in [-pi/2, pi/2]
#     e_theta = np.stack([
#         np.cos(grid_theta) * np.cos(grid_phi),
#         np.cos(grid_theta) * np.sin(grid_phi),
#         -np.sin(grid_theta)
#     ], axis=-1)
#     e_phi = np.stack([
#         -np.sin(grid_phi),
#         np.cos(grid_phi),
#         np.zeros_like(grid_phi)
#     ], axis=-1)

#     ax.grid(True, color='gray', lw=0.5)

#     quiver_kwargs = dict(
#         scale=scale,
#         cmap=cmap,
#         alpha=0.7,
#         headwidth=headwidth,
#         headlength=headlength,
#         headaxislength=headaxislength,
#         minlength=minlength,
#     )

#     if arrow_width is not None:
#         quiver_kwargs["width"] = arrow_width  # shaft width (axes units)

#     vals_reshaped = vals.reshape(grid_resolution, grid_resolution, 3)
#     vals_theta = -np.sum(vals_reshaped * e_theta, axis=-1)
#     vals_phi = np.sum(vals_reshaped * e_phi, axis=-1)
#     U = vals_phi
#     V = vals_theta
#     C = np.sqrt(U**2 + V**2)

#     if stick_length is not None:
#         eps = 1e-12
#         mag = np.maximum(C, eps)
#         U = U / mag * stick_length
#         V = V / mag * stick_length

#     im = ax.quiver(
#         grid_phi_mw[::skip, ::skip], grid_theta_mw[::skip, ::skip],
#         U[::skip, ::skip], V[::skip, ::skip],
#         C[::skip, ::skip],
#         **quiver_kwargs
#     )
#     ax.set_title(title)
#     fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.04)
#     return None




# def S2plot_quiver(fig, density_args, rho, mode, ax, skip = 1, grid_resolution = 50, scale = 5):
#     if mode not in ['gradient', 'score']:
#         raise ValueError("mode must be 'gradient' or 'score'")
        
#     grid, grid_theta, grid_phi = S2grid(grid_resolution)

#     if 'X' in density_args.keys():
#         X = density_args['X']
#         M = density_args['M']
#         _, f, grad_f = density_estimate('S2', X, M, grid) 
#     else:
#         f = density_args['f']
#         grad_f = density_args['grad_f']


#     hat_grad_f_reshaped = grad_f.reshape(grid_resolution, grid_resolution, 3)
#     hat_score = rho * grad_f / np.maximum(f[:, np.newaxis], rho)
#     grid_phi_mw = (grid_phi - np.pi)          # longitude in [-pi, pi]
#     grid_theta_mw = (np.pi/2 - grid_theta)    # latitude in [-pi/2, pi/2]

#     e_theta = np.stack([ 
#         np.cos(grid_theta) * np.cos(grid_phi), np.cos(grid_theta) * np.sin(grid_phi), -np.sin(grid_theta)
#                     ], axis=-1)
#     e_phi = np.stack([
#         -np.sin(grid_phi), np.cos(grid_phi), np.zeros_like(grid_phi)
#                         ], axis=-1)
#     ax.grid(True, color='gray', lw=0.5)
#     if mode == 'gradient':
#         grad_theta = -np.sum(hat_grad_f_reshaped * e_theta, axis=-1) # Project gradient onto tangent directions
#         grad_phi = np.sum(hat_grad_f_reshaped * e_phi, axis=-1)
#         im = ax.quiver(grid_phi_mw[::skip, ::skip], grid_theta_mw[::skip, ::skip],
#                     grad_phi[::skip, ::skip], grad_theta[::skip, ::skip],
#                     np.sqrt(grad_theta**2 + grad_phi**2)[::skip, ::skip],
#                     scale= scale, cmap='C0', alpha=0.7)
#         ax.set_title(r'$\nabla \hat f$')

#     if mode == 'score':
#         hat_score_reshaped = hat_score.reshape(grid_resolution, grid_resolution, 3)
#         grad_theta_score = -np.sum(hat_score_reshaped * e_theta, axis=-1) # Project score onto tangent directions
#         grad_phi_score = np.sum(hat_score_reshaped * e_phi, axis=-1)
#         im = ax.quiver(grid_phi_mw[::skip, ::skip], grid_theta_mw[::skip, ::skip],
#                     grad_phi_score[::skip, ::skip], grad_theta_score[::skip, ::skip],
#                     np.sqrt(grad_theta_score**2 + grad_phi_score**2)[::skip, ::skip],
#                     scale= scale, cmap='C2', alpha=0.7)
#         ax.set_title(r'$\nabla \log \hat f$')
        
#     fig.colorbar(im,ax= ax,  orientation='horizontal', fraction=0.05, pad=0.04)
#     return None

# def S2plot_density_gradient_score(X,M,sigma2,rho, grid_resolution =50, skip=1, mollwide=True):
#     X_grid, grid_theta, grid_phi = S2grid(grid_resolution)
#     X_grid, hat_f, hat_grad_f = density_estimate('S2', X, M, X_grid)
#     hat_score = sigma2*  hat_grad_f / np.maximum(hat_f[:, np.newaxis], rho)
#     # -------------------------------------------------- PLOTTING -------------------------------------------------- #
#     if mollwide:
#         fig, axs = plt.subplots( 1, 3,figsize=(15, 6), subplot_kw={'projection': 'mollweide'})
#     else:
#         fig, axs = plt.subplots(1, 3, figsize=(15, 6))
#     # Plot estimated density --------------------------------------------------
#     axs[0].grid(True, color='gray', lw=0.5)
#     axs[0].set_title(r'$\hat f$')
#     im_f = axs[0].pcolormesh((grid_phi - np.pi) , (np.pi/2 - grid_theta), 
#                             hat_f.reshape(grid_resolution, grid_resolution),
#                             alpha=0.8,shading='auto',cmap='Blues')
#     fig.colorbar(im_f, ax=axs[0], orientation='horizontal', fraction=0.05, pad=0.04)
#     # Plot gradient --------------------------------------------------
#     S2plot_quiver(fig, {'f': hat_f, 'grad_f': hat_grad_f}, rho, 'gradient', axs[1], skip = 1, grid_resolution = 50, scale = 5)
#     # Plot score -----------------------------------------------------
#     S2plot_quiver(fig, {'f': hat_f, 'grad_f': hat_grad_f}, rho, 'score', axs[2], skip = 1, grid_resolution = 50, scale = 5)
#     plt.tight_layout()
#     return fig


