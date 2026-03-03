from utils import *



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
            oracle_losses[imc] = ( manifold.metric.dist_broadcast(oracle_delta, test_Theta) ** 2).mean()

            # ------ Empirical | grid-search over (M, rho)
            density_onX = [None] * len(M_grid)  
            for ixM, M in enumerate(M_grid):
                density_onX[ixM] = density_estimate(manifold_type, X, M, test_X)

                for ixrho, rho in enumerate(rho_grid):
                    delta = denoiser(manifold_type, X, M, rho, sigma2, test_X, densityIn = (density_onX[ixM][1], density_onX[ixM][2]) )
                    losses[imc, ixM, ixrho] = ( manifold.metric.dist_broadcast(delta, test_Theta) ** 2).mean()
                    displacements[imc, ixM, ixrho] = ( manifold.metric.dist_broadcast(oracle_delta, delta) ** 2).mean()


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
            oracle_losses[imc] = ( manifold.metric.dist_broadcast(oracle_delta, test_Theta) ** 2).mean()

            # ------ Empirical | select M* by CV
            if imc == 0:
                M_star = select_M_by_DensityKFoldCV(manifold_type, X, M_grid, n_splits, n_grid_samples, return_scores=False)[penalty]
            delta = denoiser(manifold_type, X, M_star, rho, sigma2, test_X )
            empirical_losses[imc] = ( manifold.metric.dist_broadcast(delta, test_Theta) ** 2).mean()
            displacements[imc] = ( manifold.metric.dist_broadcast(oracle_delta, delta) ** 2).mean()
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