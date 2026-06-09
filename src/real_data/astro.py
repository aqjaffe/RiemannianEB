import sys, os
sys.path.append(os.getcwd().split('src')[0] + 'src')
from utils import *
import pickle

manifold_type = 'S2'
manifold = get_manifold(manifold_type)

out_dir   = os.path.join(os.getcwd().split('src')[0], 'src', 'fig')
cache_dir = os.path.join(out_dir, 'cache')
os.makedirs(out_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# Set True to ignore cached data and recompute everything.
force_recompute = False

sigma2    = 3.046e-2
criterion = 'AIC'


def compute_plot_data():
    """Run the (expensive) CV + denoising needed to draw the astro figures."""
    df = pd.read_csv('real_data/data/BATSE_4B.txt', header=None, sep='\s+', encoding='utf-8')
    ra, dec = df[5].values, df[6].values
    phi   = np.deg2rad(ra)
    theta = np.pi/2 - np.deg2rad(dec)
    X = manifold.spherical_to_extrinsic(np.column_stack([theta, phi]))

    # CV over k (individual spherical-harmonic count = (M+1)^2)
    M_max        = 24
    k_grid       = np.array([(M + 1)**2 for M in range(1, M_max + 1)])
    rhoperc_grid = np.array([1.0, 1.5, 2.0, 3.0, 5.0])


    print("Running score-matching CV over k...")
    params, scores, k_arr = scoreMatchingKFoldCV(
        manifold_type, X, M_grid=None, rho_percentile=rhoperc_grid,
        n_splits=5, return_scores=True, random_state=42, k_modes_grid=k_grid,
    )

    best_k, rho = params[criterion]
    M           = k_to_M(manifold_type, int(best_k))
    print(f"  CV optimal  k={best_k}  ->  M={M},  rho={rho:.5f}")

    delta = denoiser('S2', X, M, rho, sigma2, X)

    _X_grid, _hat_f, _hat_grad_f = density_estimate('S2', X, M, S2grid_fib(50)[0])
    _score = _hat_grad_f / np.maximum(_hat_f, rho)[:, None]

    return {
        'X': X, 'delta': delta, 'M': M, 'rho': rho,
        'score_grid': _X_grid, 'score': _score, 'hat_f': _hat_f,
        'k_arr': k_arr, 'scores': scores, 'params': params,
        'rhoperc_grid': rhoperc_grid,
    }


def get_plot_data():
    cache_path = os.path.join(cache_dir, 'astro.pkl')
    if not force_recompute and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    data = compute_plot_data()
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    return data


def make_cv_figure(data):
    fig_cv, ax_cv = plt.subplots(figsize=(9, 4))
    plot_scoreCV_k(manifold_type, data['k_arr'], data['scores'],
                   data['rhoperc_grid'], data['params'], criterion=criterion, ax=ax_cv)
    fig_cv.tight_layout()
    fig_cv.savefig(os.path.join(out_dir, 'astro_cv_k.pdf'), bbox_inches='tight')
    plt.close(fig_cv)
    print("  Saved fig/astro_cv_k.pdf")


def make_main_figure(data):
    fig, axs = plt.subplots(1, 3, figsize=(3 * 2 * style.PANEL_SIZE, style.PANEL_SIZE + 2),
                            subplot_kw={'projection': 'mollweide'},
                            gridspec_kw={'wspace': style.WSPACE})

    axs[0].set_title(r'$X_i$', y=style.TITLE_Y, fontsize=1.25*style.TITLE_SIZE)
    S2scatter(data['X'], ax=axs[0], color='C0',
              alpha=2*style.SCATTER_ALPHA, s=2*style.SCATTER_SIZE)

    S2plot_quiver(data['score_grid'], data['score'], skip=1,
                  figax=(fig, axs[1]), scale=500, cmap='Greens', cvals=data['hat_f'])
    axs[1].set_title(r'$\nabla  \log \hat{f}_n$', y=style.TITLE_Y, fontsize=1.25*style.TITLE_SIZE)

    axs[2].set_title(r'$\hat{\delta}_{\mathcal{T}}(X_i)$', y=style.TITLE_Y, fontsize=1.25*style.TITLE_SIZE)
    S2scatter(data['delta'], ax=axs[2], color='C2',
              alpha=2*style.SCATTER_ALPHA, s=2*style.SCATTER_SIZE)

    for ax in axs.flatten():
        ax.grid(True, color='gray', lw=0.5)
        ax.set_xticks(np.linspace(-np.pi, np.pi, 7)[1:-1])
        ax.set_xticklabels(['120°', '60°', '0°', '60°', '120°'], fontsize=style.TICK_SIZE)
        ax.set_yticks(np.linspace(-np.pi/2, np.pi/2, 5))
        ax.set_yticklabels(['90°', '45°', '0°', '45°', '90°'], fontsize=style.TICK_SIZE)

    fig.savefig(os.path.join(out_dir, 'astro.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig/astro.pdf")


if __name__ == '__main__':
    print('Average angular variance (in rad^2): ', sigma2)
    data = get_plot_data()
    # make_cv_figure(data)
    make_main_figure(data)
