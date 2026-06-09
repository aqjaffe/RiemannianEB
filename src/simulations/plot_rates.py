import sys, os
sys.path.append(os.getcwd().split('src')[0] + 'src')
from utils import *



manifold_type = 'S1'
params_files = sorted(f for f in os.listdir('data/' + manifold_type))
# ----- # ----- # ----- SELECT HERE ----- # ----- # ----
ID = str(params_files[-1]); print(ID)
# ----- # ----- # ----- # ----- # ----- # ---- # ----- # 
results_ocv = pd.read_csv(
    f'data/{manifold_type}/{ID}/rate_ocv.csv',
    converters={'cv_Ms_star': parse_np_array,  'cv_rhos_star': parse_np_array}
)
results_mc = pd.read_csv(f'data/{manifold_type}/{ID}/rate_mc.csv')
params = pickle.load(open(f'data/{manifold_type}/{ID}/rate_params.pkl', 'rb'))
# -----
C = 2
cases = [
    dict(C=C, use_log=True,         label=r"$M \propto \log n$"),
    dict(C=C, alpha=-1/6, beta=0.0, label=r"$M \propto n^{1/6}$"),
    dict(C=C, use_log=True,         label=r"$M \propto \log n$"),
    dict(C=C, use_log=True,         label=r"$M \propto \log n$"),
    dict(C=C, use_log=True,         label=r"$M \propto \log n$"),
]
fig, axs = plot_mcratesims_o(
    manifold_type, results_mc, results_ocv, params,
    cases=cases,
    selected_sigma2=0.15,
    selected_rho = 0.5,
    eps = 5e-5,
    ylim = (5e-5, 7.5e-1),
    savefig="../fig/S1_rates.pdf",
)
plt.show()




manifold_type = 'S2'
params_files = sorted(f for f in os.listdir('data/' + manifold_type))
# ----- # ----- # ----- SELECT HERE ----- # ----- # ----
ID = str(params_files[-1]); print(ID)
# ----- # ----- # ----- # ----- # ----- # ---- # ----- # 
results_ocv = pd.read_csv(
    f'data/{manifold_type}/{ID}/rate_ocv.csv',
    converters={'cv_Ms_star': parse_np_array,  'cv_rhos_star': parse_np_array}
)
results_mc = pd.read_csv(f'data/{manifold_type}/{ID}/rate_mc.csv')
params = pickle.load(open(f'data/{manifold_type}/{ID}/rate_params.pkl', 'rb'))
# -----
C= 2.25
cases = [
    # dict(C=3, alpha=-1/6, beta=0.0, label=r"$M \sim n^{1/6}$"),
    dict(C=C, alpha=-1/6, beta=0.0, label=r"$M \propto n^{1/3}$"),
    dict(C=C, alpha=-1/6, beta=0.0, label=r"$M \propto n^{1/3}$"),
    dict(C=C, alpha=-1/6, beta=0.0, label=r"$M \propto n^{1/3}$"),
    dict(C=C, alpha=-1/6, beta=0.0, label=r"$M \propto n^{1/3}$"),
    dict(C=C, alpha=-1/6, beta=0.0, label=r"$M \propto n^{1/3}$"),
]

fig, axs = plot_mcratesims_o(
    manifold_type, results_mc, results_ocv, params,
    cases=cases,
    selected_sigma2=0.15,
    selected_rho =0.5,
    eps =5e-4,
    savefig="../fig/S2_rates.pdf",
    ylim = (5e-4, 7.5e-1),

)
plt.show()


