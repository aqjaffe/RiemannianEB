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
fig = plt.figure(figsize=(8, 4))
sns.boxplot(data=df, y="sigma2", x="RMSE", hue="Method", orient="h")
plt.title("MC Risk vs Noise Level")
plt.tight_layout()
plt.show()
fig.savefig('out/S1.png', dpi=300)
