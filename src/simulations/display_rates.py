import sys, os
sys.path.append(os.getcwd().split('src')[0] + 'src')
from utils import *
for manifold_type in ['S1','S2']:
    params_files = sorted(f for f in os.listdir('simulations/data/' + manifold_type))
    ID = str(params_files[-1])
    print(ID)
    results_ocv = pd.read_csv(
        f'simulations/data/{manifold_type}/{ID}/rate_ocv.csv',
        converters={'cv_Ms_star': parse_np_array,  'cv_rhos_star': parse_np_array}
    )
    results_mc = pd.read_csv(f'simulations/data/{manifold_type}/{ID}/rate_mc.csv')
    params = pickle.load(open(f'simulations/data/{manifold_type}/{ID}/rate_params.pkl', 'rb'))
    display(params)
    selected_sigma2=results_ocv.sigma2.unique()[3] ; print('Selected sigma2:', selected_sigma2)
    plot_sims(manifold_type, results_mc, results_ocv, params, selected_sigma2, eps = 1e-3, absolute_excess_loss = False, CI = False,
              savefig='fig/{}_mc.png'.format(manifold_type))