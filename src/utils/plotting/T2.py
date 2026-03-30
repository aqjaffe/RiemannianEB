import numpy as np
import matplotlib.pyplot as plt
from geomstats.geometry.hypersphere import Hypersphere
from matplotlib.collections import PolyCollection
import matplotlib.gridspec as gridspec
from ..density_estimation import *
from geomstats.geometry.product_manifold import ProductManifold


circle = Hypersphere(dim=1)
torus = ProductManifold([circle, circle])



plt.rcParams.update({'font.size': 12,
                     'mathtext.fontset': 'stix',
                     'font.family': 'serif',
                     'font.serif':'Palatino'})


def T2grid(grid_resolution=50):
    # Grid on T^2 parameterized by angles (phi, psi) in [-pi, pi] x [-pi, pi]
    res_phi = grid_resolution
    res_psi = grid_resolution

    phi, psi = np.meshgrid(
        np.linspace(-np.pi, np.pi, res_phi),
        np.linspace(-np.pi, np.pi, res_psi),
        indexing="xy",
    )
    # Extrinsic embedding of S^1 x S^1 into R^2 x R^2:
    X_grid = (np.asarray([[np.cos(phi),np.sin(phi)],
                    [np.cos(psi),np.sin(psi)]]).T).reshape(-1, 2, 2)

    return X_grid, phi, psi


def T2_scatter(X, ax = None, color = 'C0', s=10, alpha = 0.25):

    X_phi = np.arctan2(X[:, 1, 0], X[:, 0, 0])
    X_psi = np.arctan2(X[:, 1, 1], X[:, 0, 1])

    if ax is None:
        fig = plt.figure(figsize=(5,4))
        ax = fig.add_subplot(111)

    ax.scatter(X_phi, X_psi, color= color, alpha=alpha, s=s)
    ax.grid(True, color='gray', lw=0.5)
    ax.axis('square')
    ax.set_title('$X_i$')

    radian_ticks = [i * np.pi/4 for i in range(-4,5)]
    radian_ticklabels = ['$-\\pi$', '$-3\\pi/4$', '$-\\pi/2$', '$-\\pi/4$', '$0$',
                        '$\\pi/4$', '$\\pi/2$', '$3\\pi/4$', '$\\pi$']

    ax.set_xticks(radian_ticks)
    ax.set_yticks(radian_ticks)
    ax.set_yticklabels(radian_ticklabels)
    ax.set_xticklabels(radian_ticklabels)
    # ax.set_xlabel('$\\phi$')
    # ax.set_ylabel('$\\psi$')

    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])

    return None

def T2plot_quiver(grid, vals, figax=None, scale=1, skip=1, cmap='Greens', cvals=None):
    if figax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig, ax = figax

    grid_resolution = int(grid.shape[0] ** 0.5)

    grid_phi = np.arctan2(grid[:, 1, 0], grid[:, 0, 0]).reshape(grid_resolution, grid_resolution)
    grid_psi = np.arctan2(grid[:, 1, 1], grid[:, 0, 1]).reshape(grid_resolution, grid_resolution)

    tangent_phi = np.stack([-grid[:, 1, 0], grid[:, 0, 0]], axis=1)  # (N, 2)
    tangent_psi = np.stack([-grid[:, 1, 1], grid[:, 0, 1]], axis=1)  # (N, 2)

    d_phi = np.einsum('ni,ni->n', vals[:, 0, :], tangent_phi)
    d_psi = np.einsum('ni,ni->n', vals[:, 1, :], tangent_psi)

    # Reshape to 2D grid before slicing
    d_phi = d_phi.reshape(grid_resolution, grid_resolution)
    d_psi = d_psi.reshape(grid_resolution, grid_resolution)

    if cvals is not None:
        C = cvals.reshape(grid_resolution, grid_resolution)
    else:
        C = np.sqrt(d_phi**2 + d_psi**2)  # now already 2D

    im = ax.quiver(
        grid_phi[::skip, ::skip],
        grid_psi[::skip, ::skip],
        d_phi[::skip, ::skip],
        d_psi[::skip, ::skip],
        C[::skip, ::skip],
        angles='xy',
        scale_units='xy',
        scale=scale,
        cmap=cmap,
        alpha=0.85,
    )

    ax.set_xlabel(r'$\varphi$', fontsize=13)
    ax.set_ylabel(r'$\psi$', fontsize=13)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                  [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                  [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()

    return None

def T2_imshow(grid, vals, figax=None):
    if figax is None:
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
    else:
        fig, ax = figax

    grid_resolution = int(grid.shape[0]**0.5)
    grid_phi = np.arctan2(grid[:, 1, 0], grid[:, 0, 0]).reshape(grid_resolution, grid_resolution)
    grid_psi = np.arctan2(grid[:, 1, 1], grid[:, 0, 1]).reshape(grid_resolution, grid_resolution)

    hat_f_grid = np.asarray(vals).reshape(grid_phi.shape)

    im = ax.pcolormesh(grid_phi, grid_psi, hat_f_grid, shading="auto", cmap="Blues", alpha=0.5)

    ax.grid(True, color='gray', lw=0.5)
    # cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    # fig.colorbar(im, orientation='vertical', cax=cbar_ax)
    ax.set_title('$\\log \\;\\hat{f}$')

    ax.axis('square')
    ax.set_title('$\\hat{f}(\cdot)$')

    ax.set_xlabel(r'$\varphi$', fontsize=13)
    ax.set_ylabel(r'$\psi$',   fontsize=13)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                  [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                  [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_aspect('equal')

    ax.grid(True, linestyle='--', alpha=0.4)
    # fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.14)


    return None
