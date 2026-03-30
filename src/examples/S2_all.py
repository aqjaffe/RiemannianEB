import sys, os
sys.path.append(os.getcwd().split('src')[0] + 'src')
from utils import *
from cluster.params import *
from tqdm.auto import tqdm

sigma2 = 0.15


manifold_type = 'S2'
n_samples = 5000
manifold = get_manifold(manifold_type)
G_sampler_ls, M_grid, rho_grid = getparams(manifold_type)[1:4]

for i, G in enumerate(tqdm(
    G_sampler_ls,
    desc=f"{manifold_type} | sampling G",
    unit="G",
    total=len(G_sampler_ls),
    dynamic_ncols=True,
    leave=False)):

    Theta = G.sample(n_samples)
    X = manifold.random_riemannian_normal(Theta, 1./(sigma2), n_samples)

    # crossvalidation (score matching) for parameter selection
    M_grid=np.arange(2, 9)
    rho_perc = np.arange(2,20,1)
    criterion = 'AIC'
    params_scoreMatching , cv_scores_scoreMatching =  scoreMatchingKFoldCV(manifold_type, X, M_grid, rho_perc, n_splits=10, return_scores=True, random_state=42)
    M,rho = params_scoreMatching[criterion]

    # --- denoisers
    delta = denoiser(manifold_type, X, M, rho, sigma2, X)
    num_oracle_samples = 10000
    oracle_delta_T = oracle_denoiser(manifold_type, num_oracle_samples, sigma2, X, n_bins = 1000, G = lambda n : G.sample(n), )

    # --- losses
    loss_N = sq_loss(manifold, X, Theta)
    loss_T = sq_loss(manifold, delta, Theta) 
    loss_oracle_T = sq_loss(manifold, oracle_delta_T, Theta) 



    alpha = .2
    grid_size  = 50;
    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)
    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    # Row 1: 3 larger polar plots (each spans 4/12 columns)
    axs_top = [fig.add_subplot(gs[0, 0:4], projection='mollweide'),
            fig.add_subplot(gs[0, 4:8], projection='mollweide'),
            fig.add_subplot(gs[0, 8:12], projection='mollweide')]

    grid_resolution = 50
    grid, grid_theta, grid_phi = S2grid(grid_resolution)
    _, hat_f, hat_grad_f, = density_estimate('S2', X, M, grid)
    hat_grad_f = np.where(hat_f[:, None] > rho, hat_grad_f, np.zeros_like(hat_grad_f))
    im = axs_top[0].pcolormesh( grid_phi - np.pi, np.pi/2 - grid_theta, hat_f.reshape(grid_resolution,grid_resolution), alpha=0.8, shading='auto', cmap='Blues')
    axs_top[0].grid(True, color='gray', lw=0.5)

    grid_resolution = 20
    grid, grid_theta, grid_phi = S2grid_fib(grid_resolution)
    _, hat_f, hat_grad_f, = density_estimate('S2', X, M, grid)
    hat_grad_f = np.where(hat_f[:, None] > rho, hat_grad_f, np.zeros_like(hat_grad_f))
    _score = hat_grad_f/np.maximum(hat_f, rho)[:, None]
    vals_to_plot =  {   'gradient': {'vals': hat_grad_f, 'cmap' : 'Blues', 'title' : r'$\nabla \hat f$' },
                        'score': {'vals': _score, 'cmap' : 'Greens', 'title' : r'$\nabla \log \hat f$' }}
    WHICH = 'gradient'
    S2plot_quiver(grid , vals_to_plot[WHICH]['vals'], figax= (fig,axs_top[1]), scale=1, cmap = vals_to_plot[WHICH]['cmap'])
    WHICH = 'score'
    S2plot_quiver(grid , vals_to_plot[WHICH]['vals'], figax= (fig,axs_top[2]), scale=10, cmap = vals_to_plot[WHICH]['cmap'], cvals = hat_f)

    axs_top[0].set_title('$\hat f$', fontsize=18)
    axs_top[1].set_title(vals_to_plot['gradient']['title'], fontsize=18)
    axs_top[2].set_title(vals_to_plot['score']['title'], fontsize=18)

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    # Row 2: 4 polar plots (each spans 3/12 columns)
    axs_bottom = [fig.add_subplot(gs[1, 0:3], projection='mollweide'),
                fig.add_subplot(gs[1, 3:6], projection='mollweide'),
                fig.add_subplot(gs[1, 6:9], projection='mollweide'),
                fig.add_subplot(gs[1, 9:12], projection='mollweide')]
    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    for ax, data,  title, loss, c, cmap in zip(axs_bottom,
                                    [Theta, X, delta, oracle_delta_T],
                                    ['$\Theta$', '$X_i$', '$\hat\delta_{\mathcal{T}}$', '$\delta_{\mathcal{T}}$', '$\delta_B$'],
                                    [None, loss_N, loss_T, loss_oracle_T],
                                    ['C3', 'C0', 'C2', 'C2', 'C4'],
                                    ['Reds', 'Blues', 'Greens', 'Greens']):  
        S2scatter(data, ax, color=c, alpha=0.25)
        ax.set_title(fr'{title}', fontsize=18)
        if loss is not None:
            ax.set_xlabel(rf'$\ell$ : {loss:.3f}', fontsize=16)
    fig.savefig(f"fig/all_examples/S2_{G.name}.png", bbox_inches='tight')


