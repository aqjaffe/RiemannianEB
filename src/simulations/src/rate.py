from utils import *

from tqdm.auto import tqdm

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
                    (M_star,rho_star) = select_M_rho_by_scoreMatchingKFoldCV(manifold_type, train_X, M_grid, rho_grid, n_splits=5, display_tqdm = False)['cv']
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






def converenge_rate_experiment__ORACLE(manifold_type, G, n_samples_ls, M_grid, rho_grid, sigma2, test_size, num_oracle_samples, NMC):
    manifold = get_manifold(manifold_type)
    
    all_records = []

    oracle_samples = G.sample(num_oracle_samples)  # pre-sample for oracle score estimation across all MC runs
    for n_samples in n_samples_ls:

        # losses indexed by [mc, M, rho]
        losses = np.zeros((NMC, len(M_grid), len(rho_grid)), dtype=float)
        displacements = np.zeros((NMC, len(M_grid), len(rho_grid)), dtype=float)
        oracle_losses = np.zeros(NMC, dtype=float)

        for imc in tqdm(range(NMC),
                        desc=f'G "{G.name}", sigma2={sigma2}, n={n_samples}',leave=False,):
            
            Theta = G.sample(n_samples)
            X = manifold.random_riemannian_normal(Theta, 1.0 / sigma2, n_samples)

            test_Theta = G.sample(test_size)
            test_X = manifold.random_riemannian_normal(test_Theta, 1.0 / sigma2, test_size)
                    
            # ------ Oracle
            oracle_delta =  oracle_denoiser(manifold_type, oracle_samples, sigma2, test_X)
            oracle_losses[imc] = sq_loss(manifold, oracle_delta, test_Theta)
            # ------ Empirical | grid-search over (M, rho)
            density_onX = [None] * len(M_grid)  
            for ixM, M in enumerate(M_grid):
                density_onX[ixM] = density_estimate(manifold_type, X, M, test_X)

                for ixrho, rho in enumerate(rho_grid):
                    delta = denoiser(manifold_type, X, M, rho, sigma2, test_X, densityIn = (density_onX[ixM][1], density_onX[ixM][2]) )
                    losses[imc, ixM, ixrho] = sq_loss(manifold, delta, test_Theta)
                    displacements[imc, ixM, ixrho] = sq_loss(manifold, oracle_delta, delta)


        # select (M*, rho*) by oracle validation
        mean_loss = losses.mean(axis=0)  # [M, rho]
        mean_displacements = displacements.mean(axis=0)  # [M, rho]
        flat_idx = int(np.argmin(mean_displacements))
        ixM_star, ixrho_star = np.unravel_index(flat_idx, mean_displacements.shape)
        M_star = int(M_grid[ixM_star])
        rho_star = float(rho_grid[ixrho_star])
        empirical_losses = losses[:, ixM_star, ixrho_star]
        
        if False:
            mean_loss_df = pd.DataFrame(mean_loss, index=M_grid, columns=rho_grid)
            mean_loss_df.index.name = "M"
            mean_loss_df.columns.name = "rho"
            print(f"\nmean_loss (G={G.name}, sigma2={sigma2}, n_samples={n_samples})")
            display(mean_loss_df)

        all_records.append(
            {
                "G": G.name,
                "sigma2": float(sigma2),
                "num_samples": int(n_samples),
                "M_star": M_star,
                "rho_star": rho_star,
                "mean_emp_loss": float(losses[:, ixM_star, ixrho_star].mean()),
                "std_emp_loss": float(losses[:, ixM_star, ixrho_star].std()),
                "mean_oracle_loss": float(oracle_losses.mean()),
                "std_oracle_loss": float(oracle_losses.std()),
                "mean_displacement": float(displacements[:, ixM_star, ixrho_star].mean()),
                "std_displacement": float(displacements[:, ixM_star, ixrho_star].std()),
                "mean_excess_loss": float((empirical_losses - oracle_losses).mean()),
                "std_excess_loss": float((empirical_losses - oracle_losses).std()),
            }
        )

    return pd.DataFrame(all_records)


def converenge_rate_experiment__CV( manifold_type, G, n_samples_ls, M_grid, rho, sigma2, test_size, num_oracle_samples, NMC, 
                                   n_splits = 5, n_grid_samples = 1000, penalty = 'cv'):
    manifold = get_manifold(manifold_type)
    
    all_records = []
    oracle_samples = G.sample(num_oracle_samples)
    for n_samples in n_samples_ls:
        empirical_losses = np.zeros(NMC, dtype=float)
        oracle_losses    = np.zeros(NMC, dtype=float)
        displacements    = np.zeros(NMC, dtype=float)
        excess_loss      = np.zeros(NMC, dtype=float)
        
        for imc in tqdm(range(NMC),
                        desc=f'G "{G.name}", sigma2={sigma2}, n={n_samples}',leave=False,):
            
            Theta = G.sample(n_samples)
            X = manifold.random_riemannian_normal(Theta, 1.0 / sigma2, n_samples)

            test_Theta = G.sample(test_size)
            test_X = manifold.random_riemannian_normal(test_Theta, 1.0 / sigma2, test_size)
                    
            # ------ Oracle
            oracle_delta =  oracle_denoiser(manifold_type, oracle_samples, sigma2, test_X)
            oracle_losses[imc] = sq_loss(manifold, oracle_delta, test_Theta)
            # ------ Empirical | select M* by CV
            if imc == 0:
                M_star = select_M_by_DensityKFoldCV(manifold_type, X, M_grid, n_splits, n_grid_samples, return_scores=False)[penalty]
            delta = denoiser(manifold_type, X, M_star, rho, sigma2, test_X )
            empirical_losses[imc] = sq_loss(manifold, delta, test_Theta)
            displacements[imc] = sq_loss(manifold, oracle_delta, delta)
            excess_loss[imc] = empirical_losses[imc] - oracle_losses[imc]

        all_records.append(
            {
                "G": G.name,
                "sigma2": float(sigma2),
                "num_samples": int(n_samples),
                "M_star": M_star,
                "rho_star": rho,
                "mean_emp_loss": float(empirical_losses.mean()),
                "std_emp_loss": float(empirical_losses.std()),
                "mean_oracle_loss": float(oracle_losses.mean()),
                "std_oracle_loss": float(oracle_losses.std()),
                "mean_displacement": float(displacements.mean()),
                "std_displacement": float(displacements.std()),
                "mean_excess_loss": float(excess_loss.mean()),
                "std_excess_loss": float(excess_loss.std()),
            }
        )
    return pd.DataFrame(all_records)



#         mean_loss = emp_loss_oracleCV.mean(axis=0)  # [M, rho]
#         mean_displacements = displacements_oracleCV.mean(axis=0)  # [M, rho]
#         flat_idx = int(np.argmin(mean_displacements))
#         ixM_star, ixrho_star = np.unravel_index(flat_idx, mean_displacements.shape)
#         M_star = int(M_grid[ixM_star])
#         rho_star = float(rho_grid[ixrho_star])
#         empirical_losses = emp_loss_oracleCV[:, ixM_star, ixrho_star]
        
        
#             # ------ Oracle
    
#             # ------ Empirical | grid-search over (M, rho)
       


#     all_records.append(
#         {
#             "G": G.name,
#             "sigma2": float(sigma2),
#             "num_samples": int(n_samples),
#             "M_star": M_star,
#             "rho_star": rho_star,
#             "mean_emp_loss": float(losses[:, ixM_star, ixrho_star].mean()),
#             "std_emp_loss": float(losses[:, ixM_star, ixrho_star].std()),
#             "mean_oracle_loss": float(oracle_losses.mean()),
#             "std_oracle_loss": float(oracle_losses.std()),
#             "mean_displacement": float(displacements[:, ixM_star, ixrho_star].mean()),
#             "std_displacement": float(displacements[:, ixM_star, ixrho_star].std()),
#             "mean_excess_loss": float((empirical_losses - oracle_losses).mean()),
#             "std_excess_loss": float((empirical_losses - oracle_losses).std()),
#         }
#     )

# return pd.DataFrame(all_records)


