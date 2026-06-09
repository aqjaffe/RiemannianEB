import sys, os
sys.path.append(os.getcwd().split('src')[0] + 'src')
from utils import *
import pickle
import Bio.PDB
warnings.simplefilter('ignore', Bio.BiopythonWarning)

manifold_type = 'T2'

out_dir   = os.path.join(os.getcwd().split('src')[0], 'src', 'fig')
cache_dir = os.path.join(out_dir, 'cache')
os.makedirs(out_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# Set True to ignore cached data and recompute everything.
force_recompute = False

criterion = 'AIC'

# Glycolytic enzymes in pathway order
files = [
    ("1HKG.pdb1", "Hexokinase\n(1HKG)"),
    ("1IRI.pdb1", "PGI\n(1IRI)"),
    ("1PFK.pdb1", "PFK\n(1PFK)"),
    ("1ADO.pdb1", "Aldolase\n(1ADO)"),
    ("2YPI.pdb1", "TIM\n(2YPI)"),
    ("1GD1.pdb1", "GAPDH\n(1GD1)"),
    ("3PGK.pdb1", "PGK\n(3PGK)"),
    ("1E58.pdb1", "PGM\n(1E58)"),
    ("1ENO.pdb1", "Enolase\n(1ENO)"),
    ("1A3W.pdb1", "Pyruvate K.\n(1A3W)"),
]

_T2_grid = T2grid(30)[0]


def _config_data(X_pool, X_list, sigma2, k_modes, M_disp, best_perc):
    """Density / score / denoised data for one truncation (given by k_modes)."""
    _, f_k = density_estimate(manifold_type, X_pool, None, X_pool,
                              grad=False, laplacian=False, k_modes=k_modes)
    rho = float(np.percentile(f_k[f_k > 0], best_perc))

    _, hat_f_grid, hat_grad_f_grid = density_estimate(
        manifold_type, X_pool, None, _T2_grid, k_modes=k_modes)
    score_grid = hat_grad_f_grid / np.maximum(hat_f_grid[:, None, None], rho) * (2 / M_disp)

    delta_list = [denoiser(manifold_type, X_pool, None, rho, sigma2, X, k_modes=k_modes)
                  for X in X_list]

    return {'k_modes': int(k_modes), 'M_disp': int(M_disp), 'rho': rho,
            'hat_f_grid': hat_f_grid, 'score_grid': score_grid, 'delta_list': delta_list}


def compute_plot_data():
    """Parse proteins, run pooled CV, then denoise under M=2, M=3 and the
    CV-optimal exact mode count — all in one pass, all cached together."""
    X_list, names_list = [], []
    b_sum_total, b_count_total = 0., 0

    for file, label in files:
        print(f"\nParsing {file}...")
        parser      = Bio.PDB.PDBParser()
        structure   = parser.get_structure('protein', 'real_data/data/' + file)
        polypeptide = Bio.PDB.PPBuilder().build_peptides(structure[0])

        phi, psi = [], []
        for strand in polypeptide:
            for point in strand.get_phi_psi_list():
                try:
                    p, q = point[0] * 1., point[1] * 1.
                    phi.append(p); psi.append(q)
                except TypeError:
                    pass

        n = min(len(phi), len(psi))
        phi, psi = np.asarray(phi[:n]), np.asarray(psi[:n])
        X_list.append(np.asarray([[np.cos(phi), np.sin(phi)],
                                   [np.cos(psi), np.sin(psi)]]).T)
        names_list.append(label)

        
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        b_sum_total   += atom.get_bfactor()
                        b_count_total += 1

    # ── Pooled sigma2 ──────────────────────────────────────────────────────────
    mean_b_factor = b_sum_total / b_count_total
    sigma2 = np.arctan(np.sqrt(mean_b_factor / (8 * np.pi**2)) / 1.5)**2
    print(f"\nPooled  mean_B = {mean_b_factor:.4f},  sigma2 = {sigma2:.4f}")

    # ── Pooled CV over complete Fourier-mode shells ────────────────────────────
    X_pool       = np.concatenate(X_list, axis=0)
    # Index complete energy shells (balanced in both torus coordinates), not raw
    # mode counts: each candidate is a symmetric {k1²+k2² ≤ E} truncation.
    k_grid       = T2_shell_modes(100)
    rhoperc_grid = np.array([2.5, 5, 7.5])

    print(f"\nRunning pooled score-matching CV ({len(k_grid)} shells up to "
          f"k_modes={k_grid[-1]})...")
    params, scores, k_arr = scoreMatchingKFoldCV(
        manifold_type, X_pool, M_grid=None, rho_percentile=rhoperc_grid,
        n_splits=100, return_scores=True, random_state=42, k_modes_grid=k_grid,
    )

    best_k, _  = params[criterion]
    best_k     = int(best_k)
    best_M     = k_to_M(manifold_type, best_k)
    ixK        = list(k_arr).index(best_k)
    ixRho_best = int(np.argmin(scores[criterion][ixK, :]))
    best_perc  = int(rhoperc_grid[ixRho_best])
    print(f"  CV-optimal k={best_k} -> M={best_M}  [rho_pct={best_perc}]")

    # ── Three truncations: fixed M=2, fixed M=3, and the CV-optimal mode count ──
    # M=2 -> (2·2+1)²=25 modes, M=3 -> (2·3+1)²=49 modes (both complete shells);
    # 'cv' uses the exact number of modes selected by cross-validation.
    spec = [('M2', (2 * 2 + 1) ** 2, 2),
            ('M3', (2 * 3 + 1) ** 2, 3),
            ('cv', best_k,           best_M)]

    configs = {}
    for label, k_modes, M_disp in spec:
        print(f"\n  [{label}] k_modes={k_modes} (M_disp={M_disp})")
        cfg = _config_data(X_pool, X_list, sigma2, k_modes, M_disp, best_perc)
        print(f"      rho={cfg['rho']:.4f}")
        configs[label] = cfg

    return {
        'X_list': X_list, 'names_list': names_list, 'X_pool': X_pool,
        'sigma2': sigma2, 'best_perc': best_perc, 'best_k': best_k, 'best_M': best_M,
        'k_arr': k_arr, 'scores': scores, 'params': params, 'rhoperc_grid': rhoperc_grid,
        'configs': configs,
    }


def get_plot_data():
    cache_path = os.path.join(cache_dir, 'chemi.pkl')
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
    fig_cv.savefig(os.path.join(out_dir, 'chemi_cv_k.pdf'), bbox_inches='tight')
    plt.close(fig_cv)
    print("  Saved fig/chemi_cv_k.pdf")


def make_grid_figure(data, label):
    cfg = data['configs'][label]
    fig = plt.figure(figsize=(14, 20))
    gs  = fig.add_gridspec(5, 4, width_ratios=[1, 1, 1, 1],
                           wspace=style.WSPACE, hspace=style.HSPACE)

    for idx, (name, X, delta) in enumerate(zip(data['names_list'], data['X_list'], cfg['delta_list'])):
        row, pair = idx // 2, idx % 2
        col_raw, col_den = pair * 2, pair * 2 + 1

        ax_raw = fig.add_subplot(gs[row, col_raw])
        ax_den = fig.add_subplot(gs[row, col_den])

        T2_scatter(X,     ax=ax_raw, alpha=style.SCATTER_ALPHA, s=style.SCATTER_SIZE, labelsize=8)
        T2_scatter(delta, ax=ax_den, color='C2', alpha=style.SCATTER_ALPHA, s=style.SCATTER_SIZE, labelsize=8, ylabel = False)

        ax_raw.set_title(name + "\n" + r"$X_i$",               fontsize=8)
        ax_den.set_title(name + "\n" + r"$\hat{\delta}(X_i)$", fontsize=8)

    fig.savefig(os.path.join(out_dir, f'chemi_{label}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved fig/chemi_{label}.pdf")


def make_summary_figure(data, label):
    cfg        = data['configs'][label]
    delta_pool = np.concatenate(cfg['delta_list'], axis=0)
    _lo, _hi   = np.percentile(cfg['hat_f_grid'], [5, 95])
    hat_f_clip = np.clip(cfg['hat_f_grid'], _lo, _hi)

    fig2, axes = plt.subplots(1, 3, figsize=(9/2 * style.PANEL_SIZE + 1, 3/2 * style.PANEL_SIZE),
                              gridspec_kw={'wspace': style.WSPACE})

    T2_scatter(data['X_pool'], ax=axes[0], alpha=style.SCATTER_ALPHA, s=style.SCATTER_SIZE)
    axes[0].set_title(r"$X_i$", fontsize=1.15*style.TITLE_SIZE)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        T2plot_quiver(_T2_grid, cfg['score_grid'], figax=(fig2, axes[1]),
                      scale=6, skip=1, cmap="Greens", cvals=hat_f_clip)
    axes[1].set_title(r"$\nabla \log \hat{f}_n$", fontsize=1.15*style.TITLE_SIZE)

    T2_scatter(delta_pool, ax=axes[2], color='C2', alpha=style.SCATTER_ALPHA, s=style.SCATTER_SIZE)
    axes[2].set_title(r"$\hat{\delta}_{\mathcal{T}}(X_i)$", fontsize=1.15*style.TITLE_SIZE)

    fig2.savefig(os.path.join(out_dir, f'chemi_summary_{label}.pdf'), bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved fig/chemi_summary_{label}.pdf")


if __name__ == '__main__':
    data = get_plot_data()
    make_cv_figure(data)
    for label in ['M2']:#('M2', 'M3', 'cv'):
        make_grid_figure(data, label)
        make_summary_figure(data, label)
