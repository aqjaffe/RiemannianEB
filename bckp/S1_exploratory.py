# import numpy as np# type: ignore
# import scipy as sp# type: ignore
# import pandas as pd# type: ignore
# import matplotlib.pyplot as plt# type: ignore
# import sys

# sys.path.append('/Users/leonardosantoro/Documents/GitHub/RiemannianEB/src')
# from utils import *

# def sample_G(n_samples): 
#     tau2 = 0.25
#     mean1 = np.array([1./np.sqrt(2), 1./np.sqrt(2)])
#     mean2 = -mean1
#     classes = np.random.randint(0,2,n_samples).reshape(-1,1)
#     samples1 = circle.random_riemannian_normal(mean1, 1./tau2, n_samples)
#     samples2 = circle.random_riemannian_normal(mean2, 1./tau2, n_samples)
#     return classes*samples1 + (1-classes)*samples2

# plt.rcParams.update({'font.size': 10,
#                      'mathtext.fontset': 'stix',
#                      'font.family': 'serif',
#                      'font.serif':'Palatino'})
# cmap_dict = {'Theta': 'Reds', 'X': 'Blues', 'Delta': 'Greens', 'f': 'Blues', 'grad': 'Blues'}

# dim = 1
# circle = Hypersphere(dim)
# # ------------------------------------------------------------------------------------------------------------
# n = 1000
# sigma2 = 0.25
# rho = 1e-12
# M = 3
# # ------------------------------------------------------------------------------------------------------------
# np.random.seed(42)
# Theta = sample_G(n)
# X = circle.random_riemannian_normal(Theta, 1./sigma2, n)
# Theta_I = circle.extrinsic_to_intrinsic_coords(Theta).ravel()
# X_I = circle.extrinsic_to_intrinsic_coords(X).ravel()
# # -------- DENSITY ESTIMATION --------
# res = 150
# on_X_I = np.linspace(0, 2*np.pi, res)
# on_X = np.column_stack((np.cos(on_X_I), np.sin(on_X_I)))
# grid, hat_f, hat_grad_f = density_estimate('S1', X, M, on_X)
# grid_angs = circle.extrinsic_to_intrinsic_coords(grid).ravel()
# hat_score = hat_grad_f / np.maximum(hat_f.ravel(), rho)
# # -------- EB DENOISING --------
# delta = denoiser('S1', X, M, rho, sigma2, X)
# delta_angs = circle.extrinsic_to_intrinsic_coords(delta).ravel()
# # -------- HISTOGRAMS --------
# incr = 15
# Theta_freqs, _ = np.histogram(Theta_I, int(360/incr)); 
# X_freqs, _ = np.histogram(X_I, int(360/incr))
# delta_freqs, _ = np.histogram(delta_angs, int(360/incr))
# # ------------------------ FIGURE ------------------------
# fig = plt.figure(figsize=(15,8), facecolor='white')
# alpha = 0.7
# f_scale = 0.5
# bottom = 0.5
# width = np.radians(incr) # width of histogram bars
# # ---------------- TOP ROW (3 panels) ----------------
# ax_top = []
# angs = np.radians(np.arange(0, 360, incr))
# angs_fill = np.append(angs, 2*np.pi)  # add 360 deg in radians
# bottom_fill = np.append(bottom*np.ones_like(angs), bottom)
# r_max = bottom + max( Theta_freqs.max(), X_freqs.max(), delta_freqs.max())
# for i in range(3):
#     ax_top.append(fig.add_subplot(2, 3, i+1, polar=True))
#     ax_top[i].set_facecolor('white'); ax_top[i].patch.set_facecolor('white')
#     ax_top[i].set_xticks(angs)
#     ax_top[i].set_yticklabels([])
#     ax_top[i].plot(angs_fill, bottom_fill*np.ones_like(angs_fill), color='black', linewidth=1., zorder=5)
#     ax_top[i].fill_between(angs_fill, np.zeros_like(angs_fill), bottom_fill, color='white', zorder=4)

# # Panel 0: Theta_i
# norm_theta = Theta_freqs / np.max(Theta_freqs)
# colors_theta = plt.colormaps[cmap_dict['Theta']](norm_theta)
# ax_top[0].bar(angs, Theta_freqs, width=width, color=colors_theta, alpha=alpha, align='edge', bottom=bottom)
# ax_top[0].set_title('$\\Theta_i$')

# # Panel 1: X_i
# norm_X = X_freqs / np.max(X_freqs)
# colors_X     = plt.colormaps[cmap_dict['X']](norm_X)
# ax_top[1].bar(angs, X_freqs, width=width, color=colors_X, alpha=alpha, align='edge', bottom=bottom)
# ax_top[1].set_title('$X_i$')

# # Panel 2: delta_i
# norm_delta = delta_freqs / np.max(delta_freqs)
# colors_delta = plt.colormaps[cmap_dict['Delta']](norm_delta)
# ax_top[2].bar(angs, delta_freqs, width=width, color=colors_delta, alpha=alpha, align='edge', bottom=bottom)
# ax_top[2].set_title('$\\hat{\\delta}_T(X_i)$')

# for ax in ax_top:
#     ax.set_rlim(0, r_max)
# # ---------------- BOTTOM ROW (2 panels) ----------------
# ax_bottom = []
# for i in range(3):
#     ax_bottom.append(fig.add_subplot(2, 3, 4+i, polar=True))
#     ax_bottom[i].set_facecolor('white')
#     ax_bottom[i].patch.set_facecolor('white')
#     ax_bottom[i].set_xticks(angs)
#     ax_bottom[i].set_yticklabels([])
#     ax_bottom[i].plot(angs_fill, bottom_fill, color='black', linewidth=1., zorder=5)
#     ax_bottom[i].fill_between(angs_fill, np.zeros_like(angs_fill), bottom_fill, color='white', zorder=4)

# # Panel 0: hat_f
# hat_f_pos = np.maximum(hat_f, 0)
# norm_hat_f = hat_f_pos / np.max(hat_f_pos)
# colors_hat_f = plt.colormaps[cmap_dict['f']](norm_hat_f)
# ax_bottom[0].bar(grid_angs, f_scale*hat_f_pos, width=2*np.pi/res, bottom=bottom,
#                  color=colors_hat_f, alpha=alpha, align='edge')
# ax_bottom[0].set_title('$\\hat{f}(\\cdot)$')


# grad_scale = 0.15  
# r_base = bottom + f_scale * 0.5 
# theta = grid_angs

# # Panel 1: gradient of hat_f (tangential arrows)
# norm_grad = hat_grad_f / np.max(np.abs(hat_grad_f)) 
# colors_grad = plt.colormaps[cmap_dict['grad']](np.abs(norm_grad))
# for i in range(0, len(theta), 3):  # subsample for clarity
#     dtheta = norm_grad[i] * grad_scale
#     ax_bottom[1].annotate('', 
#                          xy=(theta[i] + dtheta, r_base),
#                          xytext=(theta[i] + dtheta / 2, r_base),
#                         arrowprops=dict(arrowstyle='-|>,head_width=0.8,head_length=1.2', 
#                                         linewidth=1.5, color=colors_grad[i]))
# ax_bottom[1].set_title(r'$\nabla \hat{f}$')

# # Panel 2: log projections as tangential arrows
# norm_score = hat_score / np.max(np.abs(hat_score)) 
# colors_score = plt.colormaps['Greens'](np.abs(norm_score))
# for i in range(0, len(theta), 3):  # subsample for clarity
#     dtheta = norm_score[i] * grad_scale
#     ax_bottom[2].annotate(
#         '',
#         xy=(theta[i] + dtheta, r_base),
#         xytext=(theta[i] + dtheta / 2, r_base),
#         arrowprops=dict(
#             arrowstyle='-|>,head_width=0.8,head_length=1.2',
#             linewidth=1.5,
#             color=colors_score[i],
#         ),
#     )
# ax_bottom[2].set_title(r'$\sigma^2 \nabla \log \hat{f}$')
# # ax_bottom[2].set_title(r'$\log_{x}\hat{\delta}_T(x)$')

# # ---- common rescaling
# for ax in ax_bottom:
#     ax.set_rlim(0, bottom + f_scale * np.max(np.maximum(hat_f, 0)))
# plt.tight_layout()
# plt.show()

# fig.savefig('figures/S1_exploratory.png', dpi=300)






# # f_scale = 0.3
# # res = 100
# # bottom = .5
# # incr = 20
# # angs = np.radians(np.arange(0, 360, incr))
# # angs_fill = np.append(angs, 2*np.pi)  # add 360 deg in radians
# # bottom_fill = np.append(bottom*np.ones_like(angs), bottom)
# # grid_I = np.linspace(0, 2*np.pi, 100)
# # grid = np.asarray([np.cos(grid_I),np.sin(grid_I)]).T

# # fig = plt.figure(figsize=(15, 4))
# # #  G -----------------------------------------------------------------------------------------------------------
# # ax1 = fig.add_subplot(1, 3, 1, polar=True)
# # grid, hat_f, hat_grad_f = density_estimate('S1', Theta, M,  grid)
# # hat_f_pos = np.maximum(hat_f, 0); norm_hat_f = hat_f_pos / np.max(hat_f_pos)
# # ax1.bar(grid_I, f_scale*hat_f_pos, width=2*np.pi/res, bottom=bottom, color=plt.colormaps['Reds'](norm_hat_f), alpha=0.8, align='edge')
# # ax1.set_title('PRIOR $g(\cdot )$')
# # ax1.plot(angs_fill, bottom_fill*np.ones_like(angs_fill), color='black', linewidth=1., zorder=5)
# # ax1.fill_between(angs_fill, np.zeros_like(angs_fill), bottom_fill, color='white', zorder=4)
# # ax1.set_yticklabels([])
# # # f -----------------------------------------------------------------------------------------------------------
# # ax1 = fig.add_subplot(1, 3, 2, polar=True)
# # grid, hat_f, hat_grad_f = density_estimate('S1', X, M,  grid)
# # hat_f_pos = np.maximum(hat_f, 0); norm_hat_f = hat_f_pos / np.max(hat_f_pos)
# # ax1.bar(grid_I, f_scale*hat_f_pos, width=2*np.pi/res, bottom=bottom, color=plt.colormaps['Blues'](norm_hat_f), alpha=0.8, align='edge')
# # ax1.set_title('POSTERIOR $F(\cdot )$')
# # ax1.plot(angs_fill, bottom_fill*np.ones_like(angs_fill), color='black', linewidth=1., zorder=5)
# # ax1.fill_between(angs_fill, np.zeros_like(angs_fill), bottom_fill, color='white', zorder=4)
# # ax1.set_yticklabels([])
# # # delta -----------------------------------------------------------------------------------------------------------
# # ax1 = fig.add_subplot(1, 3, 3, polar=True)
# # grid_delta, hat_f_delta, hat_grad_f_delta = density_estimate('S1', delta, M, grid)
# # hat_f_delta_pos = np.maximum(hat_f_delta, 0); norm_hat_f_delta = hat_f_delta_pos / np.max(hat_f_delta_pos)
# # ax1.bar(grid_I, f_scale*hat_f_delta_pos, width=2*np.pi/res, bottom=bottom, color=plt.colormaps['Greens'](norm_hat_f_delta), alpha=0.8, align='edge')
# # ax1.set_title('DENOISED $\delta(\cdot )$')
# # ax1.plot(angs_fill, bottom_fill*np.ones_like(angs_fill), color='black', linewidth=1., zorder=5)
# # ax1.fill_between(angs_fill, np.zeros_like(angs_fill), bottom_fill, color='white', zorder=4)
# # ax1.set_yticklabels([])
# # plt.show()