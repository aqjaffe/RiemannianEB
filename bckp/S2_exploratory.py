import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

sys.path.append('/Users/leonardosantoro/Documents/GitHub/RiemannianEB/src')
from utils import *


plt.rcParams.update({'font.size': 10,
                     'mathtext.fontset': 'stix',
                     'font.family': 'serif',
                     'font.serif':'Palatino'})
cmap_dict = {'Theta': 'Reds', 'X': 'Blues', 'Delta': 'Greens', 'f': 'Blues', 'grad': 'Blues'}
dim = 2
sphere = Hypersphere(dim)
# Function to plot spherical coordinates (theta, phi) to Mollweide lon/lat projection
# ------------------------------------------------------------------------------------------------------------
def scatter_mollwide(X, ax, color, alpha=.5, s=5, lw=.5):
    X_sph = sphere.extrinsic_to_spherical(X)
    theta = X_sph[:, 0]  # colatitude
    phi = X_sph[:, 1]    # longitude
    phi_mw = phi - np.pi           # Mollweide: shift longitude from [0, 2π] to [-π, π]
    theta_mw = np.pi/2 - theta     # Mollweide: convert colatitude to latitude [-π/2, π/2]
    ax.scatter(phi_mw, theta_mw, s=s, alpha=alpha, color=color)
    ax.grid(True, color='gray', lw=lw)
    return None
def sample_G(n_samples, kappa):
    # Generate samples from a mixture of Gaussians on the sphere
    _upper = sphere.random_von_mises_fisher(kappa=kappa, mu=np.array([0, 0, 1]), n_samples=n_samples // 2)
    _lower = sphere.random_von_mises_fisher(kappa=kappa, mu=np.array([0, 0, -1]), n_samples=n_samples // 2)
    Theta = np.vstack((_upper, _lower))
    return Theta
# ------------------------------------------------------------------------------------------------------------
n_samples = 1000
M = 5
rho = 0.05
sigma2 = .1
kappa = 20
# ------------------------------------------------------------------------------------------------------------
np.random.seed(42)
Theta = sample_G(n_samples, kappa)
X = sphere.random_riemannian_normal(Theta, 1./np.sqrt(sigma2), n_samples)
delta = denoiser('S2', X, M, rho, sigma2, X)
# ------------------------------------------------------------------------------------------------------------
fig, allaxs = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': 'mollweide'})
axs = allaxs[0]
# Left plot: Theta
scatter_mollwide(Theta, axs[0], color='C1', alpha=0.25)
axs[0].set_title('$\\Theta$')

# Central plot: X
scatter_mollwide(X, axs[1], color='C0', alpha=0.25)  # Changed axs[0] to axs[1]
axs[1].set_title('$X$')

# Right plot: Denoised points
scatter_mollwide(delta, axs[2], color='C2', alpha=0.25)
axs[2].set_title('Denoised $X$')


loss_T = (sphere.metric.dist_broadcast(delta, Theta).ravel()**2).mean()      
loss_N = (sphere.metric.dist_broadcast(X, Theta).ravel()**2).mean()
print(f'Naive Loss: {loss_N:.4f}')
print(f'Denoised Loss: {loss_T:.4f}')
# ----------------------------------------------------------------------------------
# Grid on S^2 (theta = colatitude, phi = longitude)
res_lat = 25
res_lon = 25
grid_theta, grid_phi = np.meshgrid(
    np.linspace(0, np.pi, res_lat),        # colatitude
    np.linspace(0, 2*np.pi, res_lon)      # longitude
)
X_grid = np.stack([
    np.sin(grid_theta) * np.cos(grid_phi),
    np.sin(grid_theta) * np.sin(grid_phi),
    np.cos(grid_theta)
], axis=-1).reshape(-1,3)

X_grid, hat_f, hat_grad_f = density_estimate('S2', X, M, X_grid)
hat_score = sigma2*  hat_grad_f / np.maximum(hat_f[:, np.newaxis], rho)
# -------------------------------------------------- PLOTTING -------------------------------------------------- #
axs = allaxs[1]
grid_phi_mw = (grid_phi - np.pi)          # longitude in [-pi, pi]
grid_theta_mw = (np.pi/2 - grid_theta)    # latitude in [-pi/2, pi/2]
# Plot estimated density --------------------------------------------------
axs[0].grid(True, color='gray', lw=0.5)
axs[0].set_title(r'$\hat f$')
im_f = axs[0].pcolormesh(grid_phi_mw, grid_theta_mw, 
                         hat_f.reshape(res_lat, res_lon),
                         alpha=0.8,shading='auto',cmap='Blues')
fig.colorbar(im_f, ax=axs[0], orientation='horizontal', fraction=0.05, pad=0.04)
# Plot gradient --------------------------------------------------
axs[1].grid(True, color='gray', lw=0.5)
axs[1].set_title(r'$\nabla \hat f$')
hat_grad_f_reshaped = hat_grad_f.reshape(res_lat, res_lon, 3)
e_theta = np.stack([ 
    np.cos(grid_theta) * np.cos(grid_phi), np.cos(grid_theta) * np.sin(grid_phi), -np.sin(grid_theta)
                ], axis=-1)
e_phi = np.stack([
     -np.sin(grid_phi), np.cos(grid_phi), np.zeros_like(grid_phi)
                    ], axis=-1)
grad_theta = -np.sum(hat_grad_f_reshaped * e_theta, axis=-1) # Project gradient onto tangent directions
grad_phi = np.sum(hat_grad_f_reshaped * e_phi, axis=-1)
skip = 1  # Subsample for clearer visualization
im_grad = axs[1].quiver(grid_phi_mw[::skip, ::skip], grid_theta_mw[::skip, ::skip],
              grad_phi[::skip, ::skip], grad_theta[::skip, ::skip],
              np.sqrt(grad_theta**2 + grad_phi**2)[::skip, ::skip],
              scale= 5, cmap='Reds', alpha=0.7)

fig.colorbar(im_grad, ax=axs[1], orientation='horizontal', fraction=0.05, pad=0.04)
# Plot score --------------------------------------------------
hat_score_reshaped = hat_score.reshape(res_lat, res_lon, 3)
grad_theta_score = -np.sum(hat_score_reshaped * e_theta, axis=-1) # Project score onto tangent directions
grad_phi_score = np.sum(hat_score_reshaped * e_phi, axis=-1)

axs[2].grid(True, color='gray', lw=0.5)
axs[2].set_title(r'$\nabla \log \hat f$')

skip = 1  # Subsample for clearer visualization
im_score = axs[2].quiver(grid_phi_mw[::skip, ::skip], grid_theta_mw[::skip, ::skip],
              grad_phi_score[::skip, ::skip], grad_theta_score[::skip, ::skip],
              np.sqrt(grad_theta_score**2 + grad_phi_score**2)[::skip, ::skip],
              scale= 5, cmap='Greens', alpha=0.7)
fig.colorbar(im_score, ax=axs[2], orientation='horizontal', fraction=0.05, pad=0.04)
# -------------------------------------------------------------------------------------------------------------------- #
