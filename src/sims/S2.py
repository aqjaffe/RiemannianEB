# --------------------------------------------------------------------------------
n = 1000
M = 5
rho = 0.001
sigma2 = .1
kappa = 10
sigma2_grid = [0.05, 0.1, 0.2, 0.4, 0.8]
modes = [1,2,3,4,5]
NMC = 5
# --------------------------------------------------------------------------------
# CODE BELOW
# --------------------------------------------------------------------------------
import numpy as np# type: ignore
import scipy as sp# type: ignore
import pandas as pd# type: ignore
import matplotlib.pyplot as plt# type: ignore
import sys
sys.path.append('/Users/leonardosantoro/Documents/GitHub/RiemannianEB/src')
from utils import *
from tqdm import tqdm
import seaborn as sns
plt.rcParams.update({'font.size': 10,
                     'mathtext.fontset': 'stix',
                     'font.family': 'serif',
                     'font.serif':'Palatino'})
sphere = Hypersphere(2)

def sample_G(n_samples,G_params):
    num_modes, kappa  = G_params['num_modes'], G_params['kappa']
    if num_modes == 1:
        mus = np.array([[0, 0, 1]])
    elif num_modes == 2:
        mus = np.array([[0, 0, 1], [0, 0, -1]])
    else:
        # Use Fibonacci lattice for evenly distributed points
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

def MCrun(sample_G, G_params, n, sigma2, M , rho):
    sphere = Hypersphere(2)
    Theta = sample_G(n,G_params)
    print(Theta.shape)
    X = sphere.random_riemannian_normal(Theta, 1./sigma2, n)
    delta = denoiser('S2', X, M, rho, sigma2, X)
    loss_T = (sphere.metric.dist_broadcast(delta, Theta).ravel()**2).mean()      
    loss_N = (sphere.metric.dist_broadcast(X, Theta).ravel()**2).mean()
    return loss_T, loss_N

# --------------------------------------------------------------------------------

dfs = []
for num_modes in modes:
    G_params = {'kappa': kappa, 'num_modes': num_modes}
    records = []
    for sigma2 in sigma2_grid:
        for _ in tqdm(range(NMC), desc=f"sigma2={sigma2}"):
            loss_T, loss_N = MCrun(sample_G, G_params, n, sigma2, M, rho)
            records.append({
                "num_modes": num_modes,
                "sigma2": sigma2,
                "Method": "Denoised (T)",
                "RMSE": np.sqrt(loss_T)
            })
            records.append({
                "num_modes": num_modes,
                "sigma2": sigma2,
                "Method": "Naive",
                "RMSE": np.sqrt(loss_N)
            })
    dfs.append(pd.DataFrame(records))
dfs = pd.concat(dfs, axis=0)
# --------------------------------------------------------------------------------
# Plot with confidence band (median with IQR band)

# Grid on S^2 (theta = colatitude, phi = longitude)
res_lat = 50
res_lon = 50
grid_theta, grid_phi = np.meshgrid(
np.linspace(0, np.pi, res_lat),        # colatitude
np.linspace(0, 2*np.pi, res_lon)      # longitude
)
X_grid = np.stack([
np.sin(grid_theta) * np.cos(grid_phi),
np.sin(grid_theta) * np.sin(grid_phi),
np.cos(grid_theta)
], axis=-1).reshape(-1,3)
grid_phi_mw = grid_phi - np.pi          # longitude in [-pi, pi]
grid_theta_mw = np.pi/2 - grid_theta    # latitude in [-pi/2, pi/2]


fig, axs = plt.subplots(nrows=4, ncols=len(modes), figsize=(4*len(modes), 14))
fig.suptitle("MC Risk: Naive vs Denoised", y=1.1)
for idx, (num_modes, df) in enumerate(dfs.groupby("num_modes")):
    ax = axs[0,idx]
    ax.set_title(f"Number of Modes = {num_modes}")

    summary = (
        df.groupby(["sigma2", "Method"], as_index=False)["RMSE"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
    )

    for method, g in summary.groupby("Method"):
        g = g.sort_values("sigma2")
        ax.plot(
            g["sigma2"],
            g["median"],
            "o-",
            linewidth=1.5,
            markersize=4,
            label=method,
        )

        # Confidence band: IQR (q25 to q75)
        ax.fill_between(
            g["sigma2"].to_numpy(),
            g["q25"].to_numpy(),
            g["q75"].to_numpy(),
            alpha=0.2,
            linewidth=0,
        )

    # Text box
    textstr = "\n".join((
        f"NMC = {NMC}",
        f"n = {n}",
        f"$\kappa$ = {kappa}",
        f"M = {M}",
        f"$\\rho$ = {rho:.0e}",
    ))

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        1.05, 0.5, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        bbox=props,
    )

    ax.set_xlabel("$\\sigma^2$")
    ax.set_ylabel("RMSE (median with IQR band)")
    ax.legend(title="Method")
    ax.grid(True, alpha=0.3)

    # Compute densities
    ax = axs[1,idx]
    G_params = {'kappa': kappa, 'num_modes': num_modes}
    Theta = sample_G(1000, G_params)
    X_grid, hat_f_prior, _  = density_estimate('S2', Theta, M, X_grid)
    ax.grid(True, color='gray', lw=0.5)
    ax.set_title('$G_{}$'.format(num_modes))
    im = ax.pcolormesh(grid_phi_mw, grid_theta_mw, 
                        hat_f_prior.reshape(res_lat, res_lon),
                        alpha=0.8, shading='auto', cmap='Reds')
    fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.04)
    ax.set_aspect("equal", adjustable="box")

    ax = axs[2,idx]
    X = sphere.random_riemannian_normal(Theta, 1./sigma2, 1000)
    X_grid, hat_f_posterior, _  = density_estimate('S2', X, M, X_grid)
    ax.grid(True, color='gray', lw=0.5)
    ax.set_title('$f_{}$'.format(num_modes))
    im = ax.pcolormesh(grid_phi_mw, grid_theta_mw, 
                        hat_f_posterior.reshape(res_lat, res_lon),
                        alpha=0.8, shading='auto', cmap='Blues')
    fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.04)
    ax.set_aspect("equal", adjustable="box")

    ax = axs[3,idx]
    delta = denoiser('S2', X, M, rho, sigma2, X_grid)
    X_grid, hat_deltaf, _  = density_estimate('S2', delta, M, X_grid)
    ax.grid(True, color='gray', lw=0.5)
    ax.set_title('$\Delta_{}$'.format(num_modes))
    im = ax.pcolormesh(grid_phi_mw, grid_theta_mw, 
                        hat_deltaf.reshape(res_lat, res_lon),
                        alpha=0.8, shading='auto', cmap='Greens')
    fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.04)
    ax.set_aspect("equal", adjustable="box")


plt.tight_layout()
plt.show()
fig.savefig("out/S2.png", dpi=300, bbox_inches="tight")







