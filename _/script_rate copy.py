from utils import *

manifold_type = 'S1'; 
M_grid= np.arange(1, 9)
n_samples_ls =  [100, 500, 1000, 2500, 5000, 7500, 10000]
sigma2 = 0.1
NMC = 50
test_size = 1000
num_oracle_samples = 10000
rho_grid = np.logspace(-6, -1, 20)[:-1] 
# ------- # ------- # ------- # ------- # ------- # ------- # ------- # ------- # ------- # ------- # ------- # ------- # ------- # ------- 
timenow = str(time.time())
manifold = get_manifold(manifold_type)

if manifold_type == 'S1':
    G_sampler_ls = [
        get_G_class(manifold_type, sampler, name, params) for sampler, name, params 
            in [
                (uniform_sampler, 'uniform', None),
                (multimodal_sampler, '1-modal', {'tau2' : 0.1, 'num_modes' : 1}),
                (multimodal_sampler, '2-modal', {'tau2' : 0.075, 'num_modes' : 2}),
                (multimodal_sampler, '3-modal', {'tau2' : 0.05, 'num_modes' : 3}),
                (multimodal_sampler, '4-modal', {'tau2' : 0.025, 'num_modes' : 4}),
            ]
        ]


if manifold_type == 'S2':
    G_sampler_ls = [
        get_G_class(manifold_type, sampler, name, params) for sampler, name, params 
            in [
                (uniform_sampler, 'uniform', None),
                (multimodal_sampler, '1-modal', {'tau2' : 0.05, 'num_modes' : 1}),
                (multimodal_sampler, '4-modal', {'tau2' : 0.01, 'num_modes' : 4}),
                (equator_sampler, 'equator', {'tau2' : 0.001})         
            ]
        ]


# ------- # ------- # ------- # ------- # ------- # ------- # ------- # ------- # ------- # ------- # ------- # ------- # ------- # ------- 

def converenge_rate_experiment(manifold_type, G, n_samples_ls, M_grid, rho_grid, sigma2, test_size, num_oracle_samples, NMC, timenow, cv = False):
    manifold = get_manifold(manifold_type)
    oracle_samples = G.sample(num_oracle_samples)

    oracle_loss = np.zeros(NMC, dtype=float)
    naive_loss = np.zeros(NMC, dtype=float)
    emp_loss = np.zeros((NMC, len(n_samples_ls), len(M_grid), len(rho_grid)), dtype=float)
    if cv:
        cv_loss =  np.zeros((NMC, len(n_samples_ls)), dtype=float)
        cv_Ms_star = np.zeros((NMC, len(n_samples_ls)), dtype=float)
        cv_rhos_star = np.zeros((NMC, len(n_samples_ls)), dtype=float)
        cv_displacements = np.zeros_like(cv_loss)   
    displacements = np.zeros_like(emp_loss)

    total_steps = (NMC* len(n_samples_ls)* len(M_grid)* len(rho_grid))

    all_records = []; all_oracleandcv_results = []
    with tqdm(total=total_steps,
              desc=f'G="{G.name}", σ²={sigma2}',
              dynamic_ncols=True) as pbar:

        for imc in range(NMC):
            test_Theta = G.sample(test_size)
            test_X = manifold.random_riemannian_normal(test_Theta, 1.0 / sigma2, test_size)
            naive_loss[imc] = sq_loss(manifold, test_X, test_Theta)
            # Oracle
            oracle_delta = oracle_denoiser(manifold_type, oracle_samples, sigma2, test_X)
            oracle_loss[imc] = sq_loss(manifold, oracle_delta, test_Theta)

            for ixn, n_samples in enumerate(n_samples_ls):
                train_Theta = G.sample(n_samples)
                train_X = manifold.random_riemannian_normal(train_Theta, 1.0 / sigma2, n_samples)

                for ixM, M in enumerate(M_grid):
                    density_onX = density_estimate(manifold_type, train_X, M, test_X)
                    for ixrho, rho in enumerate(rho_grid):
                        delta = denoiser(manifold_type, train_X, M, rho, sigma2, test_X,densityIn=(density_onX[1], density_onX[2]))
                        emp_loss[imc, ixn, ixM, ixrho] = sq_loss(manifold, delta, test_Theta)
                        displacements[imc, ixn, ixM, ixrho] = sq_loss(manifold, oracle_delta, delta)
                        pbar.update(1)
                if cv:
                    (M_star,rho_star) = scoreMatchingKFoldCV(manifold_type, train_X, M_grid, rho_grid, n_splits=5, display_tqdm = False)['cv']
                    cv_loss[imc,ixn] = emp_loss[imc, ixn, np.where(M_grid==M_star)[0][0], np.where(rho_grid==rho_star)[0][0]]
                    cv_displacements[imc, ixn] =  displacements[imc, ixn, np.where(M_grid==M_star)[0][0], np.where(rho_grid==rho_star)[0][0]]
                    cv_Ms_star[imc, ixn] = M_star
                    cv_rhos_star[imc, ixn] = rho_star

                # Optional: show live info
                pbar.set_postfix({"MC": f"{imc+1}/{NMC}","n": n_samples})

    oracle_loss = oracle_loss - oracle_loss.std()
    # ---- aggregation ----
    for ixn, n_samples in enumerate(n_samples_ls):
        all_oracleandcv_results.append(
            {
                'ID' : float(timenow),
                'G': str(G.name),
                'sigma2': float(sigma2),
                'num_samples': int(n_samples),
                'mean_naive_loss': float(naive_loss.mean()),
                'std_naive_loss': float(naive_loss.std()),
                'mean_oracle_loss': float(oracle_loss.mean()),
                'std_oracle_loss': float(oracle_loss.std()),
                'mean_cv_loss': float(cv_loss[:,ixn ].mean()) if cv else None,
                'std_cv_loss': float(cv_loss[:,ixn ].std()) if cv else None,
                'mean_cv_excess_loss': float((cv_loss[:,ixn ] - oracle_loss).mean()) if cv else None,
                'std_cv_excess_loss': float((cv_loss[:,ixn ] - oracle_loss).std()) if cv else None,
                'mean_cv_displacement': float(cv_displacements[:,ixn].mean()) if cv else None,
                'std_cv_displacement': float(cv_displacements[:,ixn].std()) if cv else None,
                'cv_Ms_star': (cv_Ms_star[:, ixn]).astype(int) if cv else None,
                'cv_rhos_star': (cv_rhos_star[:, ixn]).astype(float) if cv else None,
            }
        )
    

        for ixM, M in enumerate(M_grid):
            for ixrho, rho in enumerate(rho_grid):
                all_records.append(
                    {
                        'ID' : float(timenow),
                        'G': G.name,
                        'sigma2': float(sigma2),
                        'num_samples': int(n_samples),
                        'M': M,
                        'rho': rho,
                        'NMC' : NMC,
                        'mean_emp_loss': float(emp_loss[:, ixn, ixM, ixrho].mean()),
                        'std_emp_loss': float(emp_loss[:, ixn, ixM, ixrho].std()),
                        'mean_displacement': float(displacements[:, ixn, ixM, ixrho].mean()),
                        'std_displacement': float(displacements[:, ixn, ixM, ixrho].std()),
                        'mean_excess_loss': float((emp_loss[:, ixn, ixM, ixrho] - oracle_loss).mean()),
                        'std_excess_loss': float((emp_loss[:, ixn, ixM, ixrho] - oracle_loss).std()),
                    }
                )
    

    return pd.DataFrame(all_records), pd.DataFrame(all_oracleandcv_results)


results_mc = []
results_ocv = []
for G in G_sampler_ls:
    dfmc, dforaclecv = converenge_rate_experiment(manifold_type, G, n_samples_ls, M_grid, rho_grid, sigma2, test_size, num_oracle_samples, NMC, timenow, cv=True)
    results_mc.append(dfmc)
    results_ocv.append(dforaclecv)

params_ = {
    'ID' : timenow,
    'manifold_type' : manifold_type,
    'n_samples_ls' : n_samples_ls,
    'M_grid' : M_grid,
    'rho_grid' : rho_grid,
    'sigma2' : sigma2,
    'test_size' : test_size,
    'num_oracle_samples' : num_oracle_samples,
    'NMC' : NMC,
    'G_names' : [G.name for G in G_sampler_ls],
    'G_params' : [G.params for G in G_sampler_ls]
}

for name, results in zip([  'mc', 
                            'ocv',
                            'params'], 
                         [  pd.concat(results_mc, ignore_index=True), 
                            pd.concat(results_ocv, ignore_index=True), 
                            params_]):
    
    out_dir = os.path.join('data', str(manifold_type), str(timenow))
    os.makedirs(out_dir, exist_ok=True)

    filepath = 'data/{}/{}/rate_{}.csv'.format(str(manifold_type),str(timenow),name,)

    if name == 'params':
        pickle.dump(results, open(filepath, 'wb'))
    else:
        results.to_csv(filepath, index=False)
