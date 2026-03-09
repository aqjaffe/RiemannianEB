
from utils import *



def mcsims_IncreasingSigma(manifold_type, n_samples, G_sampler_ls, sigma2_ls, num_oracle_samples, oracle_bandwidth, NMC, bayes = True):
    if manifold_type == 'S1':
        manifold = Hypersphere(1)
    elif manifold_type == 'S2':
        manifold = Hypersphere(2)
    elif manifold_type == 'SO3':
        manifold = SpecialOrthogonal(n=3)
    else:
        raise ValueError( "Unsupported manifold type. Supported types are 'S1', 'S2', and 'SO3'." )
        
    records = [] 
    for i, G in enumerate(G_sampler_ls):
        Theta = G.sample(n_samples)
        for sigma2 in sigma2_ls:

            loss_Ns, loss_oracleTs, loss_oracleBs = [], [], []
            for imc in tqdm(range(NMC), desc=f'G "{G.name}", σ²={sigma2}', leave=False, position=0):

                X = manifold.random_riemannian_normal(Theta, 1 / sigma2, n_samples)

                loss_Ns.append((manifold.metric.squared_dist(X, Theta)).mean())

                oracle_delta_T = oracle_denoiser(manifold_type, num_oracle_samples, sigma2,oracle_bandwidth, X, G.sample)
                loss_oracleTs.append((manifold.metric.squared_dist(oracle_delta_T, Theta)).mean())
                if bayes:
                    oracle_delta_B = oracle_bayes__kernel(manifold_type, num_oracle_samples, sigma2, oracle_bandwidth, X, G.sample)
                    loss_oracleBs.append((manifold.metric.squared_dist(oracle_delta_B, Theta)).mean())

            records.append(pd.DataFrame({
                "G": [G.name] * NMC,
                "sigma2": [sigma2] * NMC,
                "mc": np.arange(NMC),
                "Naïve": loss_Ns,
                "Oracle Denoised": loss_oracleTs ,
                "Oracle Bayes": loss_oracleBs if bayes else [np.nan] * NMC,
            }))

    df = pd.concat(records, ignore_index=True)
    df_long = df.melt(
        id_vars=[
                "G",
                "sigma2",
                "mc", 
                 ],
        value_vars=[
            "Naïve", 
            "Oracle Denoised", 
            "Oracle Bayes"
            ],
        var_name="Loss Type",
        value_name="Loss",
    )
    return df_long



def mcsims_IncreasingN(
    manifold_type,
    n_samples_ls,
    M_ls,
    G_sampler_ls,
    sigma2_ls,                 
    rho_ls,
    test_size,
    num_oracle_samples,
    oracle_bandwidth,
    NMC,
):
    if manifold_type == "S1":
        manifold = Hypersphere(1)
    elif manifold_type == "S2":
        manifold = Hypersphere(2)
    elif manifold_type == "SO3":
        manifold = SpecialOrthogonal(n=3)
    else:
        raise ValueError("Unsupported manifold type. Supported types are 'S1', 'S2', and 'SO3'.")

    all_records = []

    for sigma2 in sigma2_ls:
        # store chosen rho* for each G across n_samples
        rho_stars = {G.name: np.zeros(len(n_samples_ls)) for G in G_sampler_ls}

        for G, M in zip(G_sampler_ls, M_ls):
            for ixn, n_samples in enumerate(n_samples_ls):

                # we will only keep Empirical and Oracle (for diff)
                oracle_losses = np.zeros(NMC)
                delta_by_rho_losses = np.zeros((NMC, len(rho_ls)))

                for imc in tqdm(range(NMC), desc=f'G "{G.name}", sigma2={sigma2}, n={n_samples}', leave=False):
                    test_Theta = G.sample(test_size)
                    test_X = manifold.random_riemannian_normal(test_Theta, 1.0 / sigma2, test_size)

                    Theta = G.sample(n_samples)
                    X = manifold.random_riemannian_normal(Theta, 1.0 / sigma2, n_samples)

                    for ixrho, rho in enumerate(rho_ls):
                        delta = denoiser(manifold_type, X, M, rho, sigma2, test_X)
                        delta_by_rho_losses[imc, ixrho] = (
                            sq_loss(manifold, delta, test_Theta) ** 2
                        ).mean()

                    oracle_delta_T = oracle_denoiser(
                        manifold_type,
                        num_oracle_samples,
                        sigma2,
                        oracle_bandwidth,
                        test_X,
                        G.sample,
                    )
                    oracle_losses[imc] = sq_loss(manifold, oracle_delta_T, test_Theta)

                # select rho* using mean Empirical Denoised loss across MC (same as before)
                rho_star_idx = int(np.argmin(delta_by_rho_losses.mean(axis=0)))
                rho_star = float(rho_ls[rho_star_idx])
                rho_stars[G.name][ixn] = rho_star

                empirical_losses = delta_by_rho_losses[:, rho_star_idx]
                diff = empirical_losses - oracle_losses  # Empirical - Oracle, per MC iter

                all_records.append(
                    {
                        "G": G.name,
                        "sigma2": float(sigma2),
                        "num_samples": int(n_samples),
                        "rho_star": rho_star,
                        "median_emp_minus_oracle": float(np.median(diff)),
                        "mean_emp_minus_oracle": float(np.mean(diff)),
                        "std_emp_minus_oracle": float(np.std(diff)),
                    }
                )

    return pd.DataFrame(all_records)
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