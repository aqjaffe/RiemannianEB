try:
    from utils import *
except:
    import sys, os
    sys.path.append(os.getcwd().split('src')[0] + 'src')
    from utils import *

def getparams(manifold_type):
    M_grid= np.arange(2, 10)
    NMC = 100
    test_size = 20000
    num_oracle_samples = 10000
    rho_grid = np.linspace(0.001, 0.1, 10)
    sigma2s = np.linspace(0.005, 0.25, 6)


    if manifold_type == 'S1':
        n_samples_ls =  np.round(np.logspace(np.log10(100), np.log10(5000), 5)).astype(int)

        G_sampler_ls = [
            get_G_class(manifold_type, sampler, name, params) for sampler, name, params 
                in [
                    (uniform_sampler, 'uniform', {}),
                    (multimodal_sampler, '2-modal', {'tau2' : 0.1, 'num_modes' : 2}),
                    (multimodal_sampler, '3-modal', {'tau2' : 0.05, 'num_modes' : 3}),
                    (multimodal_sampler, '5-modal', {'tau2' : 0.0005, 'num_modes' : 5}),
                ]
            ]

    if manifold_type == 'S2':
        n_samples_ls =  np.round(np.logspace(np.log10(500), np.log10(10000), 5)).astype(int)

        G_sampler_ls = [
            get_G_class(manifold_type, sampler, name, params) for sampler, name, params 
                in [
                    (multimodal_sampler, '2-modal', {'tau2' : 0.075, 'num_modes' : 2}),
                    (multimodal_sampler, '5-modal', {'tau2' : 0.05, 'num_modes' : 5}),
                    (multimodal_sampler, '10-modal', {'tau2' : 0.001, 'num_modes' : 10}),
                    (equator_sampler, 'equator', {'tau2' : 0.0001}),         
                ]
            ]
    return n_samples_ls, G_sampler_ls, M_grid, rho_grid, sigma2s, NMC, test_size, num_oracle_samples
