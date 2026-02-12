import sys, os
from utils import *
from simulations.src import *
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
manifold_type = 'S2' # 'S1'

n_samples = 1000

num_modes_ls =  [1, 2, 3, 4]
tau2_ls = [0.05 for _ in num_modes_ls]
M_ls = [ 2,2,3,4 ]

sigma2_ls = np.linspace(0.005, 0.25, 5)

rho_ls = np.linspace(0.5, 0.005, 10)

NMC = 3

num_oracle_samples = 10000
oracle_bandwidth = 0.005

# ----- Run simulations and save results -----
df_long = mcsims_IncreasingSigma(manifold_type,n_samples, M_ls, num_modes_ls, tau2_ls, sigma2_ls, rho_ls, num_oracle_samples, oracle_bandwidth, NMC)
fig = plot_mcsims_IncreasingSigma(manifold_type,df_long, num_modes_ls, tau2_ls,
                                   savefig = '{}_mcsims_IncreasingSigma.png'.format(manifold_type))

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

sigma2 = 0.1
rho = 1e-5

n_samples_ls = [100,500, 1000,2000, 3000,4000,5000,6000,7000,8000,9000,10000]
num_modes =  [1, 2, 3, 4]
M_ls = [2,2,3,4]
tau2s = [0.05 for _ in num_modes]

rho_ls = np.linspace(0.5, 0.005, 10)

NMC = 3

num_oracle_samples = 10000
oracle_bandwidth = 0.25

test_size = 500

# ----- Run simulations and save results -----
df_long = mcsims_IncreasingN(manifold_type, num_modes, tau2s, n_samples_ls, M_ls, rho_ls, sigma2,test_size, num_oracle_samples, oracle_bandwidth, NMC)
plot_mcsims_IncreasingN(manifold_type, df_long, num_modes, tau2s)
