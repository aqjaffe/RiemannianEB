
import numpy as np
import matplotlib.pyplot as plt
from geomstats.geometry.hypersphere import Hypersphere
from matplotlib.collections import PolyCollection
import matplotlib.gridspec as gridspec

from .density_estimation import *

circle = Hypersphere(dim=1)
sphere = Hypersphere(dim=2)

plt.rcParams.update({'font.size': 12,
                     'mathtext.fontset': 'stix',
                     'font.family': 'serif',
                     'font.serif':'Palatino'})



# ------------------------------------------------------------------ CIRCLE ------------------------------------------------------------------------------------------------------
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

    # Radial jitter: zero-mean (unbiased), reproducible via global RNG state,
    # and clipped to avoid extreme outliers.
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



def S1_smooth_histogram1(X, M, ax, cmap, title = None):
    f_scale = 0.3
    res = 100
    bottom = .5
    incr = 20
    angs = np.radians(np.arange(0, 360, incr))
    angs_fill = np.append(angs, 2*np.pi)  # add 360 deg in radians
    bottom_fill = np.append(bottom*np.ones_like(angs), bottom)
    grid_I = np.linspace(0, 2*np.pi, 100)
    grid = np.asarray([np.cos(grid_I),np.sin(grid_I)]).T

    grid, hat_f, hat_grad_f = density_estimate('S1', X, M, grid)
    hat_f_pos = np.maximum(hat_f, 0); norm_hat_f = hat_f_pos / np.max(hat_f_pos)
    ax.bar(grid_I, f_scale*hat_f_pos, width=2*np.pi/res, bottom=bottom, color=plt.colormaps[cmap](norm_hat_f), alpha=0.8, align='edge')
    ax.plot(angs_fill, bottom_fill*np.ones_like(angs_fill), color='black', linewidth=1., zorder=5)
    ax.fill_between(angs_fill, np.zeros_like(angs_fill), bottom_fill, color='white', zorder=4)
    ax.set_yticklabels([])

    ax.bar(0, bottom, width=2*np.pi, bottom=0,
        color="white", edgecolor="none",
        align="edge", zorder=3)
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(theta, np.full_like(theta, bottom),
            color="black", linewidth=1.2, zorder=4)
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
# ------------------------------------------------------------------ SHPERE ------------------------------------------------------------------------------------------------------
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

def S2plot_quiver(fig, density_args, rho, mode, ax, skip = 1, grid_resolution = 50, scale = 5):
    if mode not in ['gradient', 'score']:
        raise ValueError("mode must be 'gradient' or 'score'")
        
    grid, grid_theta, grid_phi = S2grid(grid_resolution)

    if 'X' in density_args.keys():
        X = density_args['X']
        M = density_args['M']
        _, f, grad_f = density_estimate('S2', X, M, grid) 
    else:
        f = density_args['f']
        grad_f = density_args['grad_f']


    hat_grad_f_reshaped = grad_f.reshape(grid_resolution, grid_resolution, 3)
    hat_score = rho * grad_f / np.maximum(f[:, np.newaxis], rho)
    grid_phi_mw = (grid_phi - np.pi)          # longitude in [-pi, pi]
    grid_theta_mw = (np.pi/2 - grid_theta)    # latitude in [-pi/2, pi/2]

    e_theta = np.stack([ 
        np.cos(grid_theta) * np.cos(grid_phi), np.cos(grid_theta) * np.sin(grid_phi), -np.sin(grid_theta)
                    ], axis=-1)
    e_phi = np.stack([
        -np.sin(grid_phi), np.cos(grid_phi), np.zeros_like(grid_phi)
                        ], axis=-1)
    ax.grid(True, color='gray', lw=0.5)
    if mode == 'gradient':
        grad_theta = -np.sum(hat_grad_f_reshaped * e_theta, axis=-1) # Project gradient onto tangent directions
        grad_phi = np.sum(hat_grad_f_reshaped * e_phi, axis=-1)
        im = ax.quiver(grid_phi_mw[::skip, ::skip], grid_theta_mw[::skip, ::skip],
                    grad_phi[::skip, ::skip], grad_theta[::skip, ::skip],
                    np.sqrt(grad_theta**2 + grad_phi**2)[::skip, ::skip],
                    scale= scale, cmap='Blues', alpha=0.7)
        ax.set_title(r'$\nabla \hat f$')

    if mode == 'score':
        hat_score_reshaped = hat_score.reshape(grid_resolution, grid_resolution, 3)
        grad_theta_score = -np.sum(hat_score_reshaped * e_theta, axis=-1) # Project score onto tangent directions
        grad_phi_score = np.sum(hat_score_reshaped * e_phi, axis=-1)
        im = ax.quiver(grid_phi_mw[::skip, ::skip], grid_theta_mw[::skip, ::skip],
                    grad_phi_score[::skip, ::skip], grad_theta_score[::skip, ::skip],
                    np.sqrt(grad_theta_score**2 + grad_phi_score**2)[::skip, ::skip],
                    scale= scale, cmap='Greens', alpha=0.7)
        ax.set_title(r'$\nabla \log \hat f$')
        
    fig.colorbar(im,ax= ax,  orientation='horizontal', fraction=0.05, pad=0.04)
    return None

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



def S2plot_density_gradient_score(X,M,sigma2,rho, grid_resolution =50, skip=1, mollwide=True):
    X_grid, grid_theta, grid_phi = S2grid(grid_resolution)
    X_grid, hat_f, hat_grad_f = density_estimate('S2', X, M, X_grid)
    hat_score = sigma2*  hat_grad_f / np.maximum(hat_f[:, np.newaxis], rho)
    # -------------------------------------------------- PLOTTING -------------------------------------------------- #
    if mollwide:
        fig, axs = plt.subplots( 1, 3,figsize=(15, 6), subplot_kw={'projection': 'mollweide'})
    else:
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    # Plot estimated density --------------------------------------------------
    axs[0].grid(True, color='gray', lw=0.5)
    axs[0].set_title(r'$\hat f$')
    im_f = axs[0].pcolormesh((grid_phi - np.pi) , (np.pi/2 - grid_theta), 
                            hat_f.reshape(grid_resolution, grid_resolution),
                            alpha=0.8,shading='auto',cmap='Blues')
    fig.colorbar(im_f, ax=axs[0], orientation='horizontal', fraction=0.05, pad=0.04)
    # Plot gradient --------------------------------------------------
    S2plot_quiver(fig, {'f': hat_f, 'grad_f': hat_grad_f}, rho, 'gradient', axs[1], skip = 1, grid_resolution = 50, scale = 5)
    # Plot score -----------------------------------------------------
    S2plot_quiver(fig, {'f': hat_f, 'grad_f': hat_grad_f}, rho, 'score', axs[2], skip = 1, grid_resolution = 50, scale = 5)
    plt.tight_layout()
    return fig






# ----- crossvalidation
    
def plot_cv_distributions_split(results_ocv, params):
    if results_ocv.mean_cv_loss.unique()[0] != results_ocv.mean_cv_loss.unique()[0]: return None
    ID, selected_sigma2 = float(params['ID']), params['sigma2']
    # Filter for the specific NMC
    df = results_ocv[(results_ocv.ID == ID) & (results_ocv.sigma2 == selected_sigma2)] .copy()
    
    # M_grid = list(map(int, results_ocv.Ms_grid.values[0].strip('[]').split()))
    # rho_grid = ast.literal_eval(results_ocv.rhos_grid.values[0])
    M_grid = params['M_grid']
    rho_grid = params['rho_grid']
    unique_Gs = df['G'].unique()
    unique_ns = sorted(df['num_samples'].unique())
    
    n_rows = len(unique_Gs)
    n_cols = len(unique_ns)
    
    # Increase figure size: each panel now has two sub-plots
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    
    # Outer grid: Rows = G, Cols = num_samples
    outer_grid = gridspec.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.5)

    for r, g_name in enumerate(unique_Gs):
        for c, n_val in enumerate(unique_ns):
            # Create a inner grid for M and Rho histograms
            inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[r, c], hspace=0.4)
            
            row = df[(df.G == g_name) & (df.num_samples == n_val)]
            
            if row.empty or row.cv_Ms_star.iloc[0] is None:
                ax_dummy = fig.add_subplot(outer_grid[r, c])
                ax_dummy.text(0.5, 0.5, "No CV Data", ha='center')
                ax_dummy.axis('off')
                continue
            
            ms_data = row.cv_Ms_star.iloc[0]
            rhos_data = row.cv_rhos_star.iloc[0]

            # --- Subplot 1: M Histogram (Top) ---
            ax_m = fig.add_subplot(inner_grid[0, 0])
            ax_m.hist(ms_data, bins=M_grid, color='tab:blue', alpha=0.7, edgecolor='black', align = 'right')
            ax_m.set_xticks(M_grid)
            ax_m.set_title(f"M dist.", fontsize=9)
            ax_m.tick_params(axis='both', labelsize=8)
            
            # --- Subplot 2: Rho Histogram (Bottom) ---
            ax_rho = fig.add_subplot(inner_grid[1, 0])
            ax_rho.hist(rhos_data, bins=rho_grid, color='tab:red', alpha=0.7, edgecolor='black', align = 'mid')
            ax_rho.set_title(f"ρ dist.", fontsize=9)
            ax_rho.tick_params(axis='both', labelsize=8)
            ax_rho.set_xticks(rho_grid)
            ax_rho.set_xscale('log')

            # Labeling the outer boundaries
            if r == 0:
                ax_m.set_title(f"n = {n_val}\n" + ax_m.get_title(), fontsize=11, fontweight='bold')
            if c == 0:
                # Add the G name to the far left
                fig.text(0.08, 1 - (r + 0.5)/n_rows, f"G: {g_name}", 
                         va='center', rotation='vertical', fontsize=12, fontweight='bold')

    plt.suptitle(f"Split CV Distributions: M (Blue) vs ρ (Red)", fontsize=20, y=0.95)
    plt.show()

def plot_density_cv_scores(cv_scores, M_grid, rho_grid, title="CV Scores", ax = None):
    """
    Plots the CV score matrix using imshow with appropriate labels.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if (rho_grid is not None) and (len(rho_grid) > 1 ):
        # Use imshow to display the matrix
        im = ax.imshow(cv_scores, aspect='auto', origin='lower', cmap='viridis')         # origin='lower' puts the first element of M_grid at the bottom
        ax.set_xticks(np.arange(len(rho_grid)))
        ax.set_xticklabels([f"{r:.2e}" if r < 0.01 else f"{r:.2f}" for r in rho_grid], rotation=45)
        ax.set_xlabel(r"$\rho$ (Lower Bound)")
        plt.colorbar(im, label='Score')
        ax.set_yticks(np.arange(len(M_grid)))
        ax.set_yticklabels(M_grid)
        ax.set_ylabel(r"$M$ (Expansion Degree)")
    else:
        # If there's only one rho, plot cv_scores as a line plot
        ax.plot(M_grid, cv_scores.flatten(), marker='o')
        ax.set_xlabel(r"$M$ (Expansion Degree)")
    ax.set_title(title)
    if ax is None:
        plt.tight_layout()
        plt.show()
    return None

