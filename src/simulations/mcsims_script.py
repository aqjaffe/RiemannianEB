import sys, os
sys.path.append(os.getcwd().split('src')[0] + 'src')
from utils import *
from simulations.src import *


# ------------------------------------------------------------------------------------------------------------------------------------------------
# SPHERE S1
# ------------------------------------------------------------------------------------------------------------------------------------------------
manifold_type = 'S1'

G_sampler_ls = [
    get_G_class(manifold_type, sampler, name, params) for sampler, name, params 
        in [
            (multimodal_sampler, '1-modal', {'tau2' : 0.05, 'num_modes' : 1}),
            (multimodal_sampler, '2-modal', {'tau2' : 0.05, 'num_modes' : 2}),
            (multimodal_sampler, '3-modal', {'tau2' : 0.05, 'num_modes' : 3}),
            (multimodal_sampler, '4-modal', {'tau2' : 0.01, 'num_modes' : 4}),
        ]
    ]

NMC = 3
n_samples = 500
sigma2_ls = np.linspace(0.005, 0.25, 6)
num_oracle_samples = 10000
oracle_bandwidth = 0.25
# ------------------------------------------------------------------------------------------------------------------------------------------------
df_sigma = mcsims_IncreasingSigma(
    manifold_type,
    n_samples,
    G_sampler_ls,
    sigma2_ls,
    num_oracle_samples,
    oracle_bandwidth,
    NMC, 
    bayes=True)

NMC = 10
rho_ls = np.linspace(0.5, 0.005, 10)
M_ls = [ 7,7,7,7 ]
n_samples_ls = [50, 100, 250, 500]
test_size = 100
# ------------------------------------------------------------------------------------------------------------------------------------------------
df_N = mcsims_IncreasingN(
    manifold_type,
    n_samples_ls,
    M_ls,
    G_sampler_ls,
    sigma2_ls,
    rho_ls,
    test_size,
    num_oracle_samples,
    oracle_bandwidth,
    NMC)

wd = os.getcwd()
out_path = os.path.join(wd, "data", manifold_type, "df_sigma.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df_sigma.to_csv(out_path, index=False)
df_N.to_csv(os.path.join(wd, "data", manifold_type, "df_N.csv"), index=False)

plot_mcsims(manifold_type, df_sigma, df_N, G_sampler_ls, savefig='{}mc'.format(manifold_type))
# ------------------------------------------------------------------------------------------------------------------------------------------------
# SPHERE S2
# ------------------------------------------------------------------------------------------------------------------------------------------------

manifold_type = 'S2'
G_sampler_ls = [
    get_G_class(manifold_type, sampler, name, params) for sampler, name, params 
        in [
            (uniform_sampler, 'uniform', None),
            (multimodal_sampler, '1-modal', {'tau2' : 0.05, 'num_modes' : 1}),
            (multimodal_sampler, '4-modal', {'tau2' : 0.01, 'num_modes' : 4}),
            (equator_sampler, 'equator', {'tau2' : 0.001})         
        ]
    ]

NMC = 5
n_samples = 500
sigma2_ls = np.linspace(0.005, 0.25, 6)
num_oracle_samples = 10000
oracle_bandwidth = 0.25

df_sigma = mcsims_IncreasingSigma(
    manifold_type,
    n_samples,
    G_sampler_ls,
    sigma2_ls,
    num_oracle_samples,
    oracle_bandwidth,
    NMC, 
    bayes=False)

# ------------------------------------------------------------------------------------------------------------------------------------------------

NMC = 10
rho_ls = np.linspace(0.5, 0.005, 10)
M_ls = [ 7,7,7,7 ]
sigma2_ls = [0.01, 0.1, 0.2]
n_samples_ls = [100, 250, 500,1000]
test_size = 100

df_N = mcsims_IncreasingN(
    manifold_type,
    n_samples_ls,
    M_ls,
    G_sampler_ls,
    sigma2_ls,
    rho_ls,
    test_size,
    num_oracle_samples,
    oracle_bandwidth,
    NMC)

wd = os.getcwd()
out_path = os.path.join(wd, "data", manifold_type, "df_sigma.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df_sigma.to_csv(out_path, index=False)
df_N.to_csv(os.path.join(wd, "data", manifold_type, "df_N.csv"), index=False)

# ------------------------------------------------------------------------------------------------------------------------------------------------

plot_mcsims(manifold_type, df_sigma, df_N, G_sampler_ls, savefig='{}mc'.format(manifold_type))