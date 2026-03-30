from .S1 import *
from .S2 import *
from .T2 import *


# def plot_dist(manifold_type, Theta, fig, ax,kappa = 50,  cmap = 'Reds',  disk_r=0.4, top= .9, upperlim = 1.5,  grid_size = 100):

#     if manifold_type == "S1":
#         manifold = Hypersphere(1)
#         grid_I = np.linspace(0, 2 * np.pi, grid_size)
#         on_X = manifold.intrinsic_to_extrinsic_coords(grid_I[:, None])
#         hat_f = kernel_density_estimate("S1", Theta, on_X, kappa)[1]
#         hat_pos_f = np.maximum(hat_f, 0)
#         normalised_hat_f = (hat_pos_f - 0.9 * hat_pos_f.min()) / ( 1.1 * hat_pos_f.max() - 0.9 * hat_pos_f.min() + 1e-10)
#         widths = np.diff(grid_I)  # angular width of each bin
#         bars = ax.bar(
#             grid_I[:-1],                        # angular position (left edge of each bin)
#             top - disk_r,                       # height of the annulus band
#             width=widths,
#             bottom=disk_r,                      # start at disk_r
#             color=plt.colormaps[cmap](normalised_hat_f[:-1]),
#             alpha=0.85,
#             edgecolor='none',
#             align='edge',
#             zorder=2,
#         )
#         ax.set_ylim(0, upperlim)
#         ax.set_yticks([])

#         # White disk to cover r < disk_r
#         ax.bar(0, disk_r, width=2 * np.pi, bottom=0, color="white",
#             edgecolor="none", align="edge", zorder=3)

#         # Circle outline at disk_r
#         ax.plot(grid_I, disk_r * np.ones_like(grid_I),
#                 color='black', linewidth=1.2, zorder=4)
        
#     elif manifold_type == "S2":
#         manifold = Hypersphere(2)
#         ax.set_xticks([]); ax.set_yticks([])
#         ax.grid(True, alpha=0.3)
#         grid, grid_theta, grid_phi = S2grid(grid_size)
#         hat_f = kernel_density_estimate("S2",  Theta ,grid, kappa)[1].reshape(grid_size, grid_size)
#         ax.pcolormesh(
#             grid_phi - np.pi,
#             np.pi / 2 - grid_theta,
#             hat_f,
#             alpha=0.8,
#             shading="auto",
#             cmap="Reds",
#         )
#     else:
#         raise ValueError("Unsupported manifold type. Supported types are 'S1' and 'S2'.")
#     return None