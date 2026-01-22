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
tau2 = 0.05
def sample_G(n_samples, tau2 = tau2): 
    circle = Hypersphere(1)
    mean1 = np.array([1./np.sqrt(2), 1./np.sqrt(2)])
    mean2 = -mean1
    classes = np.random.randint(0,2,n_samples).reshape(-1,1)
    samples1 = circle.random_riemannian_normal(mean1, 1./tau2, n_samples)
    samples2 = circle.random_riemannian_normal(mean2, 1./tau2, n_samples)
    return classes*samples1 + (1-classes)*samples2

def MCrun(sample_G, n, sigma2, M , rho):
    circle = Hypersphere(1)
    Theta = sample_G(n)
    X = circle.random_riemannian_normal(Theta, 1./sigma2, n)
    delta = denoiser('S1', X, M, rho, sigma2, X)
    loss_T = (circle.metric.dist_broadcast(delta, Theta).ravel()**2).mean()      
    loss_N = (circle.metric.dist_broadcast(X, Theta).ravel()**2).mean()
    return loss_T, loss_N

# --------------------------------------------------------------------------------
n = 1000
M = 5
rho = 1e-12
NMC = 20
sigma2_grid = [0.05, 0.1, 0.2, 0.4, 0.8]
# --------------------------------------------------------------------------------
records = []
for sigma2 in sigma2_grid:
    for _ in tqdm(range(NMC), desc=f"sigma2={sigma2}"):
        loss_T, loss_N = MCrun(sample_G, n, sigma2, M, rho)
        records.append({
            "sigma2": sigma2,
            "Method": "Denoised (T)",
            "RMSE": np.sqrt(loss_T)
        })
        records.append({
            "sigma2": sigma2,
            "Method": "Naive",
            "RMSE": np.sqrt(loss_N)
        })
df = pd.DataFrame(records)
# --------------------------------------------------------------------------------
# Errorbar plot (mean ± standard error over MC runs)
summary = (
    df.groupby(["sigma2", "Method"], as_index=False)["RMSE"]
      .agg(mean="mean", std="std", n="count")
)
# Calculate percentiles (matching boxplot representation)
summary = (
    df.groupby(["sigma2", "Method"], as_index=False)["RMSE"]
      .agg(
          median="median",
          q25=lambda x: x.quantile(0.25),
          q75=lambda x: x.quantile(0.75),
          mean="mean"
      )
)

# Create asymmetric error bars
summary["err_lower"] = summary["median"] - summary["q25"]
summary["err_upper"] = summary["q75"] - summary["median"]

fig, ax = plt.subplots(figsize=(8, 4))
for method, g in summary.groupby("Method"):
    g = g.sort_values("sigma2")
    ax.errorbar(
        g["sigma2"],
        g["median"],  # Use median (like boxplot center line)
        yerr=[g["err_lower"], g["err_upper"]],  # Asymmetric errors to q25 and q75
        fmt="o-",
        capsize=3,
        linewidth=1.5,
        markersize=4,
        label=method,
    )

# Text box
textstr = '\n'.join((
    f'NMC = {NMC}',
    f'n = {n}',
    f'$\tau^2$ = {tau2}',
    f'$\sigma^2$ range = {sigma2_grid}',
    f'M = {M}',
    f'$\\rho$ = {rho:.0e}'))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(1.05, 0.5, textstr, transform=ax.transAxes, fontsize=10,
         verticalalignment='center', bbox=props)

ax.set_title("MC Risk vs Noise Level")
ax.set_xlabel("$\sigma^2$")
ax.set_ylabel("RMSE (median ± IQR)")
ax.legend(title="Method")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
fig.savefig("out/S2.png", dpi=300, bbox_inches='tight')