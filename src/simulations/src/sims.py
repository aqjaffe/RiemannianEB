from utils import *



def mcsims_IncreasingSigma(manifold_type, n_samples, M_ls, num_modes_ls, tau2_ls, sigma2_ls, rho_ls, num_oracle_samples, oracle_bandwidth, NMC):
    if manifold_type == 'S1':
        manifold = Hypersphere(1)
    elif manifold_type == 'S2':
        manifold = Hypersphere(2)
    elif manifold_type == 'SO3':
        manifold = SpecialOrthogonal(n=3)
    else:
        raise ValueError( "Unsupported manifold type. Supported types are 'S1', 'S2', and 'SO3'." )
        
    records = [] 
    for num_modes, tau2, M in zip(num_modes_ls, tau2_ls, M_ls):

        G_params = {'tau2': tau2 / num_modes, 'num_modes': num_modes} 

        for sigma2 in sigma2_ls:

            loss_Ns, loss_oracleTs, loss_oracleBs = [], [], []
            delta_by_rho_losses = np.zeros((NMC, len(rho_ls)))
            for imc in tqdm(range(NMC), desc=f'modes={num_modes}, σ²={sigma2}', leave=False):

                Theta = multimodal_sampler(n_samples, manifold_type, G_params)
                X = manifold.random_riemannian_normal(Theta, 1 / sigma2, n_samples)

                loss_Ns.append((manifold.metric.squared_dist(X, Theta)).mean())

                _, hat_f, hat_grad_f = density_estimate(manifold_type, X, M, X)
                for ixrho, rho in enumerate(rho_ls):
                    delta_by_rho_losses[imc, ixrho] = np.mean(
                        manifold.metric.squared_dist(
                            denoiser(manifold_type, X, M, rho, sigma2, X, densityIn=(hat_f, hat_grad_f)),
                            Theta
                        )
                    )

                oracle_delta_T = oracle_denoiser(manifold_type, num_oracle_samples, sigma2,oracle_bandwidth, X,lambda n: multimodal_sampler(n, manifold_type, G_params))
                loss_oracleTs.append((manifold.metric.squared_dist(oracle_delta_T, Theta)).mean())

                oracle_delta_B = oracle_bayes(manifold_type, num_oracle_samples, sigma2,oracle_bandwidth, X,lambda n: multimodal_sampler(n, manifold_type, G_params))
                loss_oracleBs.append((manifold.metric.squared_dist(oracle_delta_B, Theta)).mean())

            # select rho and keep only its losses
            rho_star_idx = np.argmin(delta_by_rho_losses.mean(axis=0))
            loss_Ts = delta_by_rho_losses[:, rho_star_idx]

            records.append(pd.DataFrame({
                "num_modes": [num_modes] * NMC,
                "tau2" : [tau2] * NMC,
                "sigma2": [sigma2] * NMC,
                "mc": np.arange(NMC),
                "Empirical Denoised": loss_Ts,
                "Naïve": loss_Ns,
                "Oracle Denoised": loss_oracleTs,
                "Oracle Bayes": loss_oracleBs,
                "rho": [rho_ls[rho_star_idx]] * NMC,
            }))

    df = pd.concat(records, ignore_index=True)
    df_long = df.melt(
        id_vars=["num_modes", "sigma2", "mc", "rho"],
        value_vars=["Naïve", "Empirical Denoised", "Oracle Denoised", "Oracle Bayes"],
        var_name="Loss Type",
        value_name="Loss",
    )
    return df_long





def mcsims_IncreasingN(manifold_type,num_modes_ls, tau2_ls, n_samples_ls, M_ls, rho_ls, sigma2,test_size, num_oracle_samples, oracle_bandwidth, NMC):
    if manifold_type == 'S1':
        manifold = Hypersphere(1)
    elif manifold_type == 'S2':
        manifold = Hypersphere(2)
    elif manifold_type == 'SO3':
        manifold = SpecialOrthogonal(n=3)
    else:
        raise ValueError( "Unsupported manifold type. Supported types are 'S1', 'S2', and 'SO3'." )
        

    all_records = []

    rho_stars = {
        nm: np.zeros(len(n_samples_ls)) for nm in num_modes_ls
    }

    for num_modes, tau2, M in zip(num_modes_ls, tau2_ls,M_ls):

        G_params = {'tau2': tau2, 'num_modes': num_modes} 

        for ixn, n_samples in enumerate(n_samples_ls):

            loss_Ns, loss_oracleTs = [], []

            delta_by_rho_losses = np.zeros((NMC, len(rho_ls)))

            for imc in tqdm(range(NMC), desc=f'modes={num_modes}, n={n_samples}', leave=False):

                test_Theta = multimodal_sampler(test_size, manifold_type, G_params)
                test_X = manifold.random_riemannian_normal(test_Theta, 1./sigma2, test_size)

                Theta = multimodal_sampler(n_samples, manifold_type, G_params)
                X = manifold.random_riemannian_normal(Theta, 1./sigma2, n_samples)

                loss_Ns.append(
                    (manifold.metric.dist_broadcast(test_X, test_Theta)**2).mean()
                )

                for ixrho, rho in enumerate(rho_ls):
                    delta = denoiser(manifold_type, X, M, rho, sigma2, test_X)
                    delta_by_rho_losses[imc, ixrho] = (
                        manifold.metric.dist_broadcast(delta, test_Theta)**2
                    ).mean()

                oracle_delta_T = oracle_denoiser(
                    manifold_type, num_oracle_samples, sigma2,
                    oracle_bandwidth, test_X,
                    lambda n: multimodal_sampler(n, manifold_type, G_params)
                )
                loss_oracleTs.append(
                    (manifold.metric.dist_broadcast(oracle_delta_T, test_Theta)**2).mean()
                )

            # select rho and keep only its losses
            rho_star_idx = np.argmin(delta_by_rho_losses.mean(axis=0))
            rho_stars[num_modes][ixn] = rho_ls[rho_star_idx]

            loss_Ts = delta_by_rho_losses[:, rho_star_idx]

            all_records.append(pd.DataFrame({
                "num_modes": [num_modes] * NMC,
                "tau2": [tau2] * NMC,
                "sigma2": [sigma2] * NMC,
                "rho" : [rho_ls[rho_star_idx]] * NMC,
                "num_samples": [n_samples] * NMC,
                "mc": np.arange(NMC),
                "Empirical Denoised": loss_Ts,
                "Naïve": loss_Ns,
                "Oracle Denoised": loss_oracleTs
            }))
    df = pd.concat(all_records, ignore_index=True)
    df_long = df.melt(
        id_vars=["num_modes", "num_samples", "mc", "rho"],
        value_vars=["Naïve", "Empirical Denoised", "Oracle Denoised"],
        var_name="Loss Type",
        value_name="Loss",
    )
    return df_long




# def mcsims_IncreasingN(manifold_type,num_modes_ls, tau2_ls, n_samples_ls, M_ls, rho_ls, sigma2,test_size, num_oracle_samples, oracle_bandwidth, NMC):
#     # ...existing code...

#     all_records = []

#     rho_stars = {
#         nm: np.zeros(len(n_samples_ls)) for nm in num_modes_ls
#     }

#     for num_modes, tau2, M in zip(num_modes_ls, tau2_ls, M_ls):

#         G_params = {'tau2': tau2, 'num_modes': num_modes}

#         for ixn, n_samples in enumerate(n_samples_ls):

#             loss_Ns, loss_oracleTs = [], []

#             delta_by_rho_losses = np.zeros((NMC, len(rho_ls)))

#             for imc in tqdm(range(NMC), desc=f'modes={num_modes}, n={n_samples}', leave=False):
#                 # ...existing code...
#                 pass

#             # select rho and keep only its losses
#             rho_star_idx = np.argmin(delta_by_rho_losses.mean(axis=0))
#             rho_stars[num_modes][ixn] = rho_ls[rho_star_idx]
#             loss_Ts = delta_by_rho_losses[:, rho_star_idx]

#             all_records.append(pd.DataFrame({
#                 "num_modes": [num_modes] * NMC,
#                 "tau2": [tau2] * NMC,
#                 "sigma2": [sigma2] * NMC,
#                 "rho": [rho_ls[rho_star_idx]] * NMC,
#                 "num_samples": [n_samples] * NMC,
#                 "mc": np.arange(NMC),
#                 "Empirical Denoised": loss_Ts,
#                 "Naïve": loss_Ns,
#                 "Oracle Denoised": loss_oracleTs
#             }))

#     df = pd.concat(all_records, ignore_index=True)

#     # IMPORTANT: keep tau2/sigma2 so plotting can subset correctly
#     df_long = df.melt(
#         id_vars=["num_modes", "tau2", "sigma2", "num_samples", "mc", "rho"],
#         value_vars=["Naïve", "Empirical Denoised", "Oracle Denoised"],
#         var_name="Loss Type",
#         value_name="Loss",
#     )

#     # Turn rho_stars dict into a tidy DF for plotting
#     rho_star_records = []
#     for nm in num_modes_ls:
#         for n, rho_star in zip(n_samples_ls, rho_stars[nm]):
#             # tau2 is aligned with num_modes_ls by construction (zip above)
#             tau2_for_nm = tau2_ls[num_modes_ls.index(nm)]
#             rho_star_records.append({
#                 "num_modes": nm,
#                 "tau2": tau2_for_nm,
#                 "num_samples": n,
#                 "rho_star": float(rho_star),
#             })
#     df_rho_star = pd.DataFrame(rho_star_records)

#     return df_long, df_rho_star