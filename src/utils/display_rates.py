from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from .helpers import *
from .density_estimation import *
from .plotting import *



def plot_G(manifold_type, G, fig, ax, kappa = 50):
    Theta = G.sample(1000)

    if manifold_type == "S1":
        manifold = get_manifold(manifold_type)
        ss = ax.get_subplotspec()
        ax.remove()
        ax = fig.add_subplot(ss, polar=True)

        ax.set_title(f"'{G.name}'", fontsize=14)
        
        Theta = manifold.intrinsic_to_extrinsic_coords(
            manifold.extrinsic_to_intrinsic_coords(Theta) - np.pi / 12
        )

        disk_r = 0.4
        top = 0.5
        grid_I = np.linspace(0, 2 * np.pi, 100)
        on_X = manifold.intrinsic_to_extrinsic_coords(grid_I[:, None])
        hat_f = kernel_density_estimate("S1", Theta, on_X, kappa)[1]
        hat_pos_f = np.maximum(hat_f, 0)

        if 'cap' in G.name:
            # analytical support: arc of half_angle around center (rotated by -pi/12)
            half_angle = G.params.get('half_angle', np.pi / 4)
            center_angle = (0.0 - np.pi / 12) % (2 * np.pi)
            delta = ((grid_I - center_angle) + np.pi) % (2 * np.pi) - np.pi
            normalised_hat_f = np.where(np.abs(delta) <= half_angle, 0.5, 0.0)
        elif G.name == "uniform":
            normalised_hat_f = np.ones_like(hat_pos_f) * 0.5
        else:
            normalised_hat_f = (hat_pos_f - 0.9 * hat_pos_f.min()) / (1.1 * hat_pos_f.max() - 0.9 * hat_pos_f.min() + 1e-10)

        widths = np.diff(grid_I)  # angular width of each bin

        bars = ax.bar(
            grid_I[:-1],                        # angular position (left edge of each bin)
            top - disk_r,                       # height of the annulus band
            width=widths,
            bottom=disk_r,                      # start at disk_r
            color=plt.colormaps['Reds'](normalised_hat_f[:-1]),
            alpha=0.85,
            edgecolor='none',
            align='edge',
            zorder=2,
        )

        ax.set_ylim(0, top)
        ax.set_yticks([])

        # White disk to cover r < disk_r
        ax.bar(0, disk_r, width=2 * np.pi, bottom=0, color="white",
            edgecolor="none", align="edge", zorder=3)

        # Circle outline at disk_r
        ax.plot(grid_I, disk_r * np.ones_like(grid_I),
                color='black', linewidth=1.2, zorder=4)
        
    elif manifold_type == "S2":
        manifold = get_manifold(manifold_type)
        ss = ax.get_subplotspec()
        ax.remove()
        ax = fig.add_subplot(ss, projection="mollweide")
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, alpha=0.3)

        grid_resolution = 100
        grid, grid_theta, grid_phi = S2grid(grid_resolution)
        ax.set_title(f"'{G.name}'", fontsize=14)

        z_grid = np.cos(grid_theta).reshape(grid_resolution, grid_resolution)

        if 'cap' in G.name:
            # analytical support: z >= cos(half_angle) around north pole
            half_angle = G.params.get('half_angle', np.pi / 4)
            normalised_hat_f = np.where(z_grid >= np.cos(half_angle), 0.5, 0.0)
        elif 'equator' in G.name:
            # analytical support: band |z| <= 2.5*sqrt(tau2) around equator
            tau2 = G.params.get('tau2', 0.01)
            normalised_hat_f = np.where(np.abs(z_grid) <= 2.5 * np.sqrt(tau2), 0.5, 0.0)
        else:
            hat_f = kernel_density_estimate("S2", Theta, grid, kappa)[1].reshape(
                grid_resolution, grid_resolution
            )
            if G.name == "uniform":
                normalised_hat_f = np.ones((grid_resolution, grid_resolution)) * 0.5
            else:
                hat_pos_f = np.maximum(hat_f, 0)
                normalised_hat_f = (hat_pos_f - 0.9 * hat_pos_f.min()) / (1.1 * hat_pos_f.max() - 0.9 * hat_pos_f.min() + 1e-10)

        ax.pcolormesh(
            grid_phi - np.pi,
            np.pi / 2 - grid_theta,
            normalised_hat_f,
            alpha=0.8,
            shading="auto",
            cmap="Reds",
            vmin=0,
            vmax=1,
        )
    else:
        raise ValueError("Unsupported manifold type. Supported types are 'S1' and 'S2'.")



def plot_mcratesims_o(
    manifold_type, results, results_ocv, params,
    cases,
    selected_sigma2=None,
    selected_rho=None,
    eps=1e-4,
    variables=('displacement',),
    ylim=None,
    savefig=None,
):
    """Static publication-quality rate plot.

    len(cases) must equal the number of G distributions. Column i shows:
      - row 0 : prior of G_i
      - row 1+: scatter + fitted line for G_i using cases[i] M(n) rule.

    Parameters
    ----------
    cases : list of dict (one per G), keys:
        'C'       : float — prefactor
        'alpha'   : float — exponent on n  (M ~ n^{-alpha})
        'beta'    : float — exponent on σ  (M ~ σ^{-beta})
        'use_log' : bool  — use M = C·log(n) instead of power law
        'label'   : str   — subtitle shown below the prior
    selected_rho : float or None — if None, uses first rho in results.
    """
    with mpl.rc_context({
        'font.family':     'serif',
        'font.size':       10,
        'axes.labelsize':  10,
        'axes.titlesize':  10,
        'legend.fontsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.linewidth':  0.8,
        'lines.linewidth': 1.4,
        'figure.dpi':      150,
    }):
        G_sampler_ls = get_G_sampler_ls_from_params(params)
        K     = len(G_sampler_ls)
        n_var = len(variables)
        M_arr = np.asarray(sorted(results.M.unique()), dtype=float)

        assert len(cases) == K, f"len(cases)={len(cases)} must equal number of G distributions ({K})"

        if selected_sigma2 is None:
            selected_sigma2 = sorted(results.sigma2.unique())
            if len(selected_sigma2) > 2: selected_sigma2 = selected_sigma2[1:]
            # selected_sigma2 = _s2s[len(_s2s) // 2]
        if selected_rho is None:
            selected_rho = sorted(results.rho.unique())[0]

        has_bayes = ("mean_oracle_bayes_loss" in results_ocv.columns and
                     "std_oracle_bayes_loss"  in results_ocv.columns)

        LOSS_META = {
            "naive":  ("$\\delta_N$",                    "C0", "",     "o"),
            "oracle": ("$\\delta_{\\mathcal{T}}$",        "C2", "",     "s"),
        }
        if has_bayes:
            LOSS_META["oracle_bayes"] = ("$\\delta_{\\mathcal{B}}$", "purple", (3, 1), "^")

        fig, axs = plt.subplots(
            2 + n_var, K,
            figsize=(3.2 * K, 2.8 * (2 + n_var)),
            gridspec_kw={"hspace": 0.50, "wspace": 0.30},
        )
        if K == 1:
            axs = np.atleast_2d(axs).T   # shape (2+n_var, 1)

        for i, (G, case) in enumerate(zip(G_sampler_ls, cases)):
            C        = case.get('C', 1.0)
            alpha    = case.get('alpha', 0.0)
            beta     = case.get('beta', 0.0)
            use_log  = case.get('use_log', False)
            const_M  = case.get('M', None)
            lbl      = case.get('label', '')

            # ── row 0: prior ────────────────────────────────────────────
            plot_G(manifold_type, G, fig, axs[0, i])

            # ── row 1: naive vs oracle denoiser for increasing sigma ─────
            _agg_spec = dict(
                mean_naive_loss=("mean_naive_loss", "mean"),
                std_naive_loss=("std_naive_loss", "mean"),
                mean_oracle_loss=("mean_oracle_loss", "mean"),
                std_oracle_loss=("std_oracle_loss", "mean"),
                mean_naive_displacement=("mean_naive_displacement", "mean"),
            )
            if has_bayes:
                _agg_spec["mean_oracle_bayes_loss"] = ("mean_oracle_bayes_loss", "mean")
                _agg_spec["std_oracle_bayes_loss"]  = ("std_oracle_bayes_loss",  "mean")
            oracle_df_G = (
                results_ocv.loc[results_ocv["G"] == G.name]
                .groupby("sigma2", as_index=False)
                .agg(**_agg_spec)
            )
            df_plot = pd.concat(
                [
                    oracle_df_G[["sigma2", f"mean_{k}_loss", f"std_{k}_loss"]]
                    .set_axis(["sigma2", "Loss", "Std"], axis=1)
                    .assign(**{"Loss Type": label, "color": color, "dash": [dash] * len(oracle_df_G)})
                    for k, (label, color, dash, marker) in LOSS_META.items()
                    if (f"mean_{k}_loss" in oracle_df_G.columns and
                        f"std_{k}_loss"  in oracle_df_G.columns)
                ],
                ignore_index=True,
            )
            _palette = {v[0]: v[1] for v in LOSS_META.values()}
            _dashes  = {v[0]: v[2] for v in LOSS_META.values()}
            _markers = {v[0]: v[3] for v in LOSS_META.values()}
            # draw delta_T (oracle) last so it sits on top of the others
            _oracle_label = LOSS_META["oracle"][0]
            _labels = [lbl for lbl in _palette if lbl != _oracle_label] + [_oracle_label]
            sns.lineplot(
                data=df_plot, x="sigma2", y="Loss",
                hue="Loss Type", hue_order=_labels, palette=_palette,
                style="Loss Type", dashes=_dashes, markers=_markers,
                estimator=None,
                ax=axs[1, i], alpha = .5
            )
            if has_bayes:
                bayes_color = mpl.colors.to_rgba(LOSS_META["oracle_bayes"][1])
                for line in axs[1, i].get_lines():
                    if mpl.colors.to_rgba(line.get_color()) == bayes_color:
                        line.set_alpha(0.5)
                        line.set_markerfacecolor("none")
                        line.set_markeredgecolor(bayes_color)
                        line.set_markeredgewidth(1.2)
            axs[1, i].set_xlabel("$\\sigma^2$")
            axs[1, i].spines['top'].set_visible(False)
            axs[1, i].spines['right'].set_visible(False)
            axs[1, i].grid(True, which="major", ls=":", lw=0.4, alpha=0.5)
            if i == K - 1:
                axs[1, i].legend(loc="upper left", frameon=False)
            else:
                axs[1, i].legend([], [], frameon=False)
            if i == 0:
                axs[1, i].set_ylabel("$R(\delta, \sigma^2)$")
            else:
                axs[1, i].set_ylabel("")

            # ── precompute M(n) for this G/case ─────────────────────────
            selected_sigma2 = np.atleast_1d(selected_sigma2)
            for selected_sigma2 in selected_sigma2:
                sigma = np.sqrt(selected_sigma2)
                df_G = (results[
                            (results.G == G.name) &
                            (results.sigma2 == selected_sigma2) &
                            (results.rho == selected_rho)
                        ].copy().sort_values('num_samples'))

                ns = np.sort(df_G.num_samples.unique()).astype(float)
                if len(ns) > 0:
                    if const_M is not None:
                        M_chosen = M_arr[np.argmin(np.abs(M_arr - float(const_M)))] * np.ones(len(ns))
                    elif use_log:
                        M_target = C * np.log(ns)
                        M_chosen = M_arr[np.argmin(
                            np.abs(np.log(M_arr[None, :]) - np.log(M_target[:, None])), axis=1
                        )]
                    else:
                        M_target = C * (sigma ** (-beta)) * (ns ** (-alpha))
                        M_chosen = M_arr[np.argmin(np.abs(np.log(M_arr[None, :]) - np.log(M_target[:, None])), axis=1)]
                else:
                    M_chosen = np.array([])

                # ── rows 2+: one scatter+fit per variable ────────────────────
                for v_idx, variable in enumerate(variables):
                    ax  = axs[2 + v_idx, i]
                    col = 'mean_' + variable
                    ax.set_xscale("log")

                    xs_pts, ys_pts = [], []
                    for n, M_pick in zip(ns, M_chosen):
                        row = df_G[(df_G.num_samples == n) & (df_G.M == M_pick)]
                        if len(row) == 0:
                            continue
                        m = float(row[col].values[0])
                        m_c = max(m, eps) if eps is not None else m
                        xs_pts.append(float(n))
                        ys_pts.append(m_c)

                    slope = np.nan
                    if xs_pts:
                        xs_arr = np.asarray(xs_pts)
                        y_arr  = np.asarray(ys_pts)
                        ax.scatter(xs_arr, y_arr, color='C2', zorder=3, s=25)

                        valid = (np.isfinite(xs_arr) & np.isfinite(y_arr)
                                & (xs_arr > 0) & (y_arr > 0))
                        if valid.sum() > 1:
                            slope, intercept = np.polyfit(
                                np.log(xs_arr[valid]), np.log(y_arr[valid]), 1
                            )
                            ns_fit = np.array([xs_arr[valid].min(), xs_arr[valid].max()])
                            y_fit  = np.exp(intercept) * ns_fit ** slope
                            slope_str = f"{slope:.2f}" if np.isfinite(slope) else "—"

                            ax.plot(ns_fit, y_fit, color='C2', lw=1.2, ls='--', alpha=0.8)
                            ax.annotate(f"{lbl}\nslope {slope_str}", xy=(0.75, 0.95), xycoords='axes fraction',
                                        fontsize=8, ha='left', va='top')

                    # title: slope on rate rows (skip the first)
                    if v_idx != 0:
                        ax.set_title(f"slope {slope_str}", fontsize=9, pad=3)

                    # yscale
                    all_y = [np.asarray(l.get_ydata(orig=False), dtype=float).ravel()
                            for l in ax.get_lines()]
                    all_y = [y for y in all_y if y.size]
                    ly = np.concatenate(all_y) if all_y else np.array([])
                    if ly.size and np.all(np.isfinite(ly)) and np.all(ly > 0):
                        ax.set_yscale("log")
                    else:
                        ax.set_yscale("symlog", linthresh=1e-10)

                    ax.grid(True, which="major", ls=":", lw=0.4, alpha=0.5)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    if v_idx == n_var - 1:
                        ax.set_xlabel("$n$")
                    if i == 0:
                        ax.set_ylabel("$\mathbb{E}[D_{\mathcal{T}}(\hat\delta_{\mathcal{T}})]$")

        # share y-axis across columns, separately per variable row, with fixed limits
        for v_idx in range(n_var):
            for j in range(1, K):
                axs[2 + v_idx, j].sharey(axs[2 + v_idx, 0])
            if ylim is not None:
                axs[2 + v_idx, 0].set_ylim(*ylim)

        fig.align_ylabels(axs[2:, 0])

        if savefig is not None:
            fig.savefig(savefig, bbox_inches='tight')

        return fig, axs


