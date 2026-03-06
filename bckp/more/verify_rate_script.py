# import sys, os
# sys.path.append(os.getcwd().split('src')[0] + 'src')
# from utils import *
# from simulations.src import *

# def sq_loss(manifold, X, delta):
#     return ( manifold.metric.dist_broadcast(X, delta) ** 2).mean()


# NMC = 10

# manifold_type = 'S1'; 
# G =  get_G_class(manifold_type, multimodal_sampler, '2-modal', {'tau2' : 0.05, 'num_modes' : 2})
# sigma2 = 0.1


# n_samples_ls = [100, 500, 1000, 5000, 10000]
# test_size = 1000
# num_oracle_samples = 5000

# n_splits = 8
# penalty = 'cv'  # 'cv', 'AIC', or 'BIC'

# rho_grid = [0.5, 0.25, 1e-1, 0.05, 0.025, 1e-2, 1e-4, 1e-6, 1e-8]
# M_grid =  [1,2,3,4,5,6,7,8,9,10,15,20]

# rho_hard = 1e-3
# ix_rho_hard = np.where(np.array(rho_grid) == rho_hard)[0][0]
# M_hard = 20
# ix_M_hard = np.where(np.array(M_grid) == M_hard)[0][0]



# # ======================================================================
# # ======================================================================
# # ======================================================================

# manifold = get_manifold(manifold_type)
# oracle_samples = G.sample(num_oracle_samples)
# all_records = []

# # oracle 
# oracle_loss = np.zeros(NMC, dtype=float)


# # empirical | oracle cv 
# emp_loss_oracleCV = np.zeros((NMC, len(M_grid), len(rho_grid)), dtype=float) # losses indexed by [mc, M, rho]
# displacements_oracleCV = np.zeros((NMC, len(M_grid), len(rho_grid)), dtype=float)

# # empirical | fixed choices 
# emp_loss_hard = np.zeros(NMC, dtype=float)
# displacements_hard = np.zeros(NMC, dtype=float)

# # empirical | density-based cv 
# emp_loss_densityCV = np.zeros(NMC, dtype=float)
# displacements_densityCV = np.zeros(NMC, dtype=float)


# for imc in tqdm(range(NMC),
#                 desc=f'G "{G.name}", sigma2={sigma2}, n={n_samples_ls[0]}',leave=False,):
    
#     test_Theta = G.sample(test_size)
#     test_X = manifold.random_riemannian_normal(test_Theta, 1.0 / sigma2, test_size)

#     # oracle estimation
#     oracle_delta =  oracle_denoiser(manifold_type, oracle_samples, sigma2, test_X)
#     oracle_loss[imc] = sq_loss(manifold, oracle_delta, test_Theta)

#     for n_samples in n_samples_ls:
#         train_Theta = G.sample(n_samples)
#         train_X = manifold.random_riemannian_normal(train_Theta, 1.0 / sigma2, n_samples)

#         # oracle cross-validation
#         density_onX = [None] * len(M_grid)  
#         for ixM, M in enumerate(M_grid):
#             density_onX[ixM] = density_estimate(manifold_type, train_X, M, test_X)
#             for ixrho, rho in enumerate(rho_grid):
#                 delta = denoiser(manifold_type, train_X, M, rho, sigma2, test_X, densityIn = (density_onX[ixM][1], density_onX[ixM][2]) )
#                 emp_loss_oracleCV[imc, ixM, ixrho] =  sq_loss(manifold, delta, test_Theta)
#                 displacements_oracleCV[imc, ixM, ixrho] = sq_loss(manifold, oracle_delta, delta)
                
#         # fixed choices 
#         emp_loss_hard[imc] = emp_loss_oracleCV[imc, ix_M_hard, ix_rho_hard]
#         displacements_hard[imc] = displacements_oracleCV[imc, ix_M_hard, ix_rho_hard]

#         # data-driven cross-validation
#         M_star = select_M_by_DensityKFoldCV(manifold_type, train_X, M_grid, n_splits)[penalty]
#         emp_loss_densityCV[imc] = emp_loss_oracleCV[   
#                                                         imc, 
#                                                         np.where(M_grid == M_star)[0][0],
#                                                         ix_rho_hard
#                                                     ]

#     mean_loss = emp_loss_oracleCV.mean(axis=0)  # [M, rho]
#     mean_displacements = displacements_oracleCV.mean(axis=0)  # [M, rho]
#     flat_idx = int(np.argmin(mean_displacements))
#     ixM_star, ixrho_star = np.unravel_index(flat_idx, mean_displacements.shape)
#     M_star = int(M_grid[ixM_star])
#     rho_star = float(rho_grid[ixrho_star])
#     empirical_losses = emp_loss_oracleCV[:, ixM_star, ixrho_star]
    
      
#         # ------ Oracle
   
#         # ------ Empirical | grid-search over (M, rho)
       


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
