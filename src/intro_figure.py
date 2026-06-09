import sys, os
sys.path.append(os.getcwd().split('src')[0] + 'src')
from utils import *
from simulations.params.params import getparams
from tqdm.auto import tqdm
import pickle
# Set to True to ignore cached data and recompute everything.
force_recompute = False


sigma2 = 0.15
manifold_type = 'S2'
n_samples = 5000
manifold = get_manifold(manifold_type)
G_sampler_ls = [getparams(manifold_type)[1][i] for i in range(5)]

out_dir = os.path.join(os.getcwd().split('src')[0], 'src', 'fig')
os.makedirs(out_dir, exist_ok=True)

cache_dir = os.path.join(out_dir, 'cache')
os.makedirs(cache_dir, exist_ok=True)

SCORE_EQUIRECTANGULAR = False

def compute_plot_data(G):
    """Run the (expensive) computation needed to produce one figure."""
    # Stage-level progress bar over the heavy steps. Some stages (score-matching
    # CV, the Bayes CV/denoise) also show their own inner bar.
    stages = ['score-matching CV', 'denoiser', 'oracle delta_T', 'oracle delta_B', 'score field']
    bar = tqdm(stages, desc=f"{G.name} | compute", unit="stage", leave=False, dynamic_ncols=True)
    sit = iter(bar)
    next_stage = lambda: bar.set_postfix_str(next(sit))

    Theta = G.sample(n_samples)
    X = manifold.random_riemannian_normal(Theta, 1. / sigma2, n_samples)

    M_grid = np.arange(2, 9)
    rho_perc = np.arange(2, 20, 1)
    next_stage()
    params_cv, _ = scoreMatchingKFoldCV(manifold_type, X, M_grid, rho_perc, n_splits=10, return_scores=True, random_state=42)
    M, rho = params_cv['AIC']

    next_stage()
    delta = denoiser(manifold_type, X, M, rho, sigma2, X)

    num_oracle_samples = 50000
    oracle_Theta = G.sample(num_oracle_samples)

    next_stage()
    oracle_delta_T = oracle_denoiser(manifold_type, oracle_Theta, sigma2, X,
                                     n_bins=num_oracle_samples//10, G=lambda n, _G=G: _G.sample(n))

    next_stage()

    num_oracle_samples = 100000
    oracle_Theta = G.sample(num_oracle_samples)
    oracle_delta_B = oracle_bayes__kernel_cv(manifold_type, oracle_Theta, sigma2, X,
                                              max_n=num_oracle_samples, n_eval=500)
    next_stage()

    loss_N = sq_loss(manifold, X, Theta)
    loss_T = sq_loss(manifold, delta, Theta)
    loss_oracle_T = sq_loss(manifold, oracle_delta_T, Theta)
    loss_oracle_B = sq_loss(manifold, oracle_delta_B, Theta)

    # Score field: estimate the density/score from a much larger sample so the
    # displayed quiver is smooth. This affects ONLY the score panel — every
    # scatter panel and all losses above use the original n_samples draw.
    n_score = 50000
    Theta_score = G.sample(n_score)
    X_score     = manifold.random_riemannian_normal(Theta_score, 1. / sigma2, n_score)

    grid_resolution = 20
    grid_fib, _, _ = S2grid_fib(grid_resolution)
    _, hat_f_grid, hat_grad_f = density_estimate('S2', X_score, M, grid_fib)
    hat_score = hat_grad_f / np.maximum(hat_f_grid.reshape(hat_f_grid.shape + (1,) * (hat_grad_f.ndim - 1)), rho)

    return {
        'Theta': Theta,
        'X': X,
        'delta': delta,
        'oracle_delta_T': oracle_delta_T,
        'oracle_delta_B': oracle_delta_B,
        'loss_N': loss_N,
        'loss_T': loss_T,
        'loss_oracle_T': loss_oracle_T,
        'loss_oracle_B': loss_oracle_B,
        'grid_fib': grid_fib,
        'hat_score': hat_score,
        'hat_f_grid': hat_f_grid,
    }


def get_plot_data(G):
    """Load cached plot data for G, computing and caching it if missing."""
    cache_path = os.path.join(cache_dir, f'S2_{G.name}.pkl')
    if not force_recompute and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    data = compute_plot_data(G)
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    return data


# change of camera viewpoint for improved visualisation.
LON_ROT = np.pi / 2


def _rot_z(arr, alpha=LON_ROT):
    """Rotate extrinsic S2 points/vectors (..., 3) about the z-axis by alpha."""
    c, s = np.cos(alpha), np.sin(alpha)
    R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return arr @ R.T


def make_figure(G, data):
    fig, axs = plt.subplots(2, 3, figsize=(3 * 2 * style.PANEL_SIZE, 2 * style.PANEL_SIZE + 1),
                             subplot_kw={'projection': 'mollweide'},
                             gridspec_kw={'height_ratios': [1.2, 1.0],
                                          'hspace': style.HSPACE, 'wspace': style.WSPACE})

    def _plot_scatter(ax, data, title, loss, color):
        S2scatter(data, ax, color=color, alpha=style.SCATTER_ALPHA, s=style.SCATTER_SIZE)
        ax.set_title(title, fontsize=1.25*style.TITLE_SIZE)
        ax.grid(True, color='gray', lw=0.5)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        if loss is not None:
            ax.set_xlabel(rf'risk : {loss:.3f}', fontsize=style.LABEL_SIZE)

    # Row 1: Theta | oracle δ_B | oracle δ_T
    _plot_scatter(axs[0, 0], _rot_z(data['Theta']),          r'$\Theta_i$',               None,                  'C3')
    _plot_scatter(axs[0, 1], _rot_z(data['oracle_delta_B']), r'$\delta_B(X_i)$',             data['loss_oracle_B'], 'C4')
    _plot_scatter(axs[0, 2], _rot_z(data['oracle_delta_T']), r'$\delta_{\mathcal{T}}(X_i)$', data['loss_oracle_T'], 'C2')

    # Row 2: X | score | δ̂_T
    _plot_scatter(axs[1, 0], _rot_z(data['X']), r'$X_i$', data['loss_N'], 'C0')

    # Projection of the score panel is controlled by the module-level
    # SCORE_EQUIRECTANGULAR flag (see top of file).
    if SCORE_EQUIRECTANGULAR:
        # Swap the prebuilt Mollweide axes for a rectilinear one at the same spot.
        score_pos = axs[1, 1].get_position()
        fig.delaxes(axs[1, 1])
        ax_score = fig.add_axes(score_pos.bounds)
        axs[1, 1] = ax_score
    else:
        ax_score = axs[1, 1]

    S2plot_quiver(_rot_z(data['grid_fib']), _rot_z(data['hat_score']), figax=(fig, ax_score),
                  scale=10 if SCORE_EQUIRECTANGULAR else 40, cmap='Greens',
                  cvals=data['hat_f_grid'], equirectangular=SCORE_EQUIRECTANGULAR,
                  hide_outlier_quantile=0.95)
    ax_score.set_title(r'$\nabla  \log \hat{f}_n$', fontsize=1.25*style.TITLE_SIZE)
    ax_score.set_xticklabels([]); ax_score.set_yticklabels([])

    _plot_scatter(axs[1, 2], _rot_z(data['delta']), r'$\hat\delta_{\mathcal{T}}(X_i)$', data['loss_T'], 'C2')

    fig.savefig(os.path.join(out_dir, f'intro___S2_{G.name}.pdf'), bbox_inches='tight')
    plt.close(fig)


for G in tqdm(G_sampler_ls, desc=f"{manifold_type} | sampling G", unit="G",
              total=len(G_sampler_ls), dynamic_ncols=True, leave=False):
    data = get_plot_data(G)
    make_figure(G, data)

