




# --------------------------------------------------------------------------------
n = 1000
M = 5
rho = 1e-2
NMC = 5
tau2 = 0.05
sigma2_grid = [0.05, 0.1, 0.2, 0.4, 0.8]
modes = [1,2,3,4,5]
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
 
def MCrun(G_params, n, sigma2, M , rho):
    circle = Hypersphere(1)
    Theta = S1_multimodal_prior(n,G_params)
    X = circle.random_riemannian_normal(Theta, 1./sigma2, n)
    delta = denoiser('S1', X, M, rho, sigma2, X)
    loss_T = (circle.metric.dist_broadcast(delta, Theta).ravel()**2).mean()      
    loss_N = (circle.metric.dist_broadcast(X, Theta).ravel()**2).mean()
    return loss_T, loss_N

# --------------------------------------------------------------------------------
dfs = []
for num_modes in modes:
    G_params = {'tau2': tau2, 'num_modes': num_modes}
    records = []
    for sigma2 in sigma2_grid:
        for _ in tqdm(range(NMC), desc=f"sigma2={sigma2}"):
            loss_T, loss_N = MCrun(G_params, n, sigma2, M, rho)
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
fig, axs = plt.subplots(nrows=4, ncols=len(modes), figsize=(4*len(modes), 12), sharey=False)
fig.suptitle("MC Risk: Naive vs Denoised")
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
        f"$\\tau^2$ = {tau2}",
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


    ax = axs[1,idx]
    Theta = S1_multimodal_prior(1000, {'tau2': tau2, 'num_modes': num_modes})
    sns.kdeplot(x=Hypersphere(1).extrinsic_to_intrinsic_coords(Theta).ravel(), fill=True, ax=ax, color= 'red')
    ax.set_title("$g_{}$".format(num_modes))

    ax = axs[2, idx]
    X = Hypersphere(1).random_riemannian_normal(Theta, 1./sigma2_grid[2], 1000)
    sns.kdeplot(x=Hypersphere(1).extrinsic_to_intrinsic_coords(X).ravel(), fill=True, ax=ax, color= 'blue')
    ax.set_title("$f_{}$".format(num_modes))


    ax = axs[3, idx]
    delta = denoiser('S1', X, M, rho, sigma2_grid[2], X)
    sns.kdeplot(x=Hypersphere(1).extrinsic_to_intrinsic_coords(delta).ravel(), fill=True, ax=ax, color= 'green')
    ax.set_title("$\delta_{}$".format(num_modes))
    for ax in axs[1:,:].flatten():
        ax.set_xlabel("Angle (radians)")
        ax.set_xlim(-2*np.pi, 2*np.pi)
plt.tight_layout()
plt.show()
fig.savefig("figures/S1.png", dpi=300, bbox_inches="tight")