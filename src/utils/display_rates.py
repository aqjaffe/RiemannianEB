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

        if G.name == "uniform": normalised_hat_f = np.ones_like(hat_pos_f)*hat_pos_f.mean()
        else:normalised_hat_f = (hat_pos_f - 0.9 * hat_pos_f.min()) / ( 1.1 * hat_pos_f.max() - 0.9 * hat_pos_f.min() + 1e-10)

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
        hat_f = kernel_density_estimate("S2",  Theta ,grid, kappa)[1].reshape(
            grid_resolution, grid_resolution
        )
        ax.pcolormesh(
            grid_phi - np.pi,
            np.pi / 2 - grid_theta,
            hat_f,
            alpha=0.8,
            shading="auto",
            cmap="Reds",
            vmin=0.5 if G.name == "uniform" else None,
            vmax=0.5 if G.name == "uniform" else None,
        )
    else:
        raise ValueError("Unsupported manifold type. Supported types are 'S1' and 'S2'.")


def plot_sims(manifold_type, results_mc, results_ocv, params, selected_sigma2 = None, showvars = ['excess_loss'], eps = 1e-5,
               absolute_excess_loss = False, CI= False, savefig = None):
    if selected_sigma2 is None:
        selected_sigma2 = results_ocv['sigma2'].unique()[-3]
    def _axis_all_positive_finite(ax) -> bool:
        """Return True iff *all* y-data on the axis is finite and strictly positive."""
        ys = []
        for line in ax.get_lines():
            y = line.get_ydata(orig=False)
            if y is None:
                continue
            y = np.asarray(y, dtype=float)
            if y.size:
                ys.append(y)
        if not ys:
            return False
        y_all = np.concatenate(ys)
        return np.all(np.isfinite(y_all)) and np.all(y_all > 0)

    LOSS_META = {
    "naive":  ("$\delta_N$",          "C0", ""),
    "oracle": ("$\delta_{\mathcal{T}}$", "C2", (1, 1)),
    }


    NMC = params.get('NMC', 1)
    G_sampler_ls = get_G_sampler_ls_from_params(params)

    K = len(G_sampler_ls)
    fig, axs = plt.subplots(2 + len(showvars), K, figsize=(20, 10/3*(2+len(showvars))), gridspec_kw={"hspace": 0.35, "wspace": 0.25})

    for idx, G in enumerate(G_sampler_ls):

        # FIRST ROW: plot the distribution G on the manfiold
        plot_G(manifold_type, G, fig, axs[0, idx])



        # SECOND ROW: plot oracle loss for increasing values of sigma2
        oracle_df_G = (
            results_ocv.loc[results_ocv["G"] == G.name]
            .groupby("sigma2", as_index=False)
            .agg(
                mean_naive_loss=("mean_naive_loss", "mean"),
                std_naive_loss=("std_naive_loss", "mean"),
                mean_oracle_loss=("mean_oracle_loss", "mean"),
                std_oracle_loss=("std_oracle_loss", "mean"),
                mean_cv_loss=("mean_cv_loss", "mean"),
                std_cv_loss=("std_cv_loss", "mean"),
                median_cv_excess_loss = ("mean_cv_excess_loss", "median")
            )
        )

        df_plot = pd.concat(
            [
                oracle_df_G[["sigma2", f"mean_{k}_loss", f"std_{k}_loss"]]
                .set_axis(["sigma2", "Loss", "Std"], axis=1)
                .assign(
                    **{"Loss Type": label, "color": color, "dash": [dash] * len(oracle_df_G)},
                    ci_lo=lambda d: d["Loss"] - 1.96 * d["Std"],
                    ci_hi=lambda d: d["Loss"] + 1.96 * d["Std"],
                )
                for k, (label, color, dash) in LOSS_META.items()
            ],
            ignore_index=True,
        )

        labels, palette, dashes = ([v[i] for v in LOSS_META.values()] for i in range(3))
        # axs[1, idx].set_yscale("log")
        
        sns.lineplot(
            data=df_plot, x="sigma2", y="Loss",
            hue="Loss Type", hue_order=labels, palette=dict(zip(labels, palette)),
            style="Loss Type", dashes=dict(zip(labels, dashes)),
            estimator=None, marker="o", 
            ax= axs[1, idx]
        )


        for label, color in zip(labels, palette):
            d = df_plot.loc[df_plot["Loss Type"] == label].sort_values("sigma2")
            axs[1, idx].fill_between(d.sigma2, d.ci_lo, d.ci_hi, color=color, alpha=0.15, linewidth=0)
            # axs[1, idx].fill_between(d.sigma2, d.ci_lo, d.ci_hi, color=color, alpha=0.15, linewidth=0)
            if idx == K-1:
                axs[1, idx].legend( loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, borderaxespad=0.0,fontsize = 16)
            else:
                axs[1, idx].legend([],[], frameon=False)
                axs[1, idx].set_ylabel('')
            axs[1, idx].set_xlabel("$\sigma^2$")




        # sample_sizes = sorted(results_ocv[results_ocv["G"] == G.name]["num_samples"].unique())
        # n = len(sample_sizes)

        # for rank, n_samples in enumerate(sample_sizes):
        #     df_cv_sigma = (
        #         results_ocv[(results_ocv["G"] == G.name) & (results_ocv["num_samples"] == n_samples)]
        #         .sort_values("sigma2")
        #     )
        #     off = max(2, round(12 - 10 * rank / max(n - 1, 1)))
        #     axs[1, idx].plot(
        #         df_cv_sigma["sigma2"],
        #         df_cv_sigma["mean_cv_loss"],
        #         color="C2",
        #         linewidth=1.5,
        #         linestyle=(0, (4, off)),
        #         alpha=0.75,
        #         label=f"CV $n$={int(n_samples)}" if idx == K - 1 else "_nolegend_",
        #     )


            # pos = axs[1, 0].get_position()  # any column works
            # y_sep = pos.y0 - 0.05  # bottom of 2nd row in figure coords (shifted down)
            # line = plt.Line2D([0, 1], [y_sep, y_sep], transform=fig.transFigure, color="black", linewidth=1)
            # fig.add_artist(line)
            # ----------------------
        # THIRD + ROWs: empirical (cv) excess loss and displacement error for increasing sample size
        for i, variable in enumerate(showvars):  
            if variable == "excess_loss":
                axs[2 + i, idx].axhline(results_ocv.loc[(results_ocv.G == G.name) & (results_ocv.sigma2 == selected_sigma2),'mean_naive_loss'].values.mean(),
                                         color='C0', linestyle='--', lw=2, alpha=0.7, label = '$R(\delta_N) - R(\delta_T)$')

            df_cv_G = results_ocv[(results_ocv.G == G.name) & (results_ocv.sigma2 == selected_sigma2)].copy().sort_values('num_samples')
            ax = axs[2 + i, idx]
            cv_col = f"mean_cv_{variable}"
            x_cv = df_cv_G["num_samples"].to_numpy(dtype=float)
            y_cv = df_cv_G[cv_col].to_numpy(dtype=float)
            if variable == "excess_loss" and absolute_excess_loss:
                y_cv = np.abs(y_cv)
            if eps is not None:
                y_cv_plot = y_cv + eps #np.clip(y_cv, eps, None)
                y_cv_upper = y_cv_plot + 1.96 * df_cv_G[f"std_cv_{variable}"].to_numpy(dtype=float)/np.sqrt(NMC)
                y_cv_lower = np.clip(y_cv_plot - 1.96 * df_cv_G[f"std_cv_{variable}"].to_numpy(dtype=float)/np.sqrt(NMC), eps, None)
                if CI:
                    ax.fill_between(x_cv, y_cv_lower, y_cv_upper, color='C2', alpha=0.15)
            else:
                y_cv_plot = y_cv

            ax.plot(x_cv, y_cv_plot, ':', color='C2', lw=2, label = '$R(\hat\delta_T) - R(\delta_T)$')
            for _, row in df_cv_G.iterrows():
                y = float(row[cv_col]) + eps
                ax.annotate(f"M={int(np.median(row.cv_Ms_star))}\nρ={row.cv_rhos_star.mean():.3f}",
                            xy=(row.num_samples, y),
                            fontsize=7, color='C2', ha='center', va='top')

            valid_cv = np.isfinite(x_cv) & np.isfinite(y_cv_plot) & (x_cv > 0) & (y_cv_plot > 0)
            if valid_cv.sum() > 1:
                b_cv = np.polyfit(np.log(x_cv[valid_cv]), np.log(y_cv_plot[valid_cv]), 1)[0]
            else: b_cv = np.nan

            if variable == "excess_loss":
                if manifold_type == 'S1':
                    ax.set_ylim(.75e-3, 1.25e-1)
                else:
                    ax.set_ylim(.99e-3, 1.75e-1)

            else:
                ax.set_ylabel(variable)
            ax.set_xlabel("n")
            if _axis_all_positive_finite(ax):
                ax.set_xscale("log")
                ax.set_yscale("log")

        # ── single shared legend, placed after both loops ──────────────────────────
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        if idx == K-1:
            axs[2, idx].legend( loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, borderaxespad=0.0,fontsize = 16)
        else:
            axs[2, idx].legend([],[], frameon=False)

    for ax in axs.ravel():
        ax.set_ylabel('')
    axs[2, 0].set_ylabel("Excess MSE")
    axs[1, 0].set_ylabel("MSE")
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')
    return None






try:
    import ipywidgets as widgets
    from IPython.display import display
    from functools import lru_cache
    import io
    def plot_mcratesims_interactive(manifold_type, results, results_ocv, params, extselected_Mrho=None, cvonly = False, eps = 1e-4):
        G_sampler_ls = get_G_sampler_ls_from_params(params)

        @lru_cache(maxsize=None)
        def _cached_G_image(manifold_type, G):
            """Render plot_G to a PNG bytes buffer and cache it."""
            fig_g, ax_g = plt.subplots(1, 1, figsize=(20/len(G_sampler_ls), 10 * 0.65 / 3))
            plot_G(manifold_type, G, fig_g, ax_g)
            buf = io.BytesIO()
            fig_g.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig_g)
            buf.seek(0)
            return buf.read()  # returns raw PNG bytes

        def _axis_all_positive_finite(ax) -> bool:
            """Return True iff *all* y-data on the axis is finite and strictly positive."""
            ys = []
            for line in ax.get_lines():
                y = line.get_ydata(orig=False)
                if y is None:
                    continue
                y = np.asarray(y, dtype=float)
                if y.size:
                    ys.append(y)
            if not ys:
                return False
            y_all = np.concatenate(ys)
            return np.all(np.isfinite(y_all)) and np.all(y_all > 0)

        M_options   = sorted(results.M.unique().tolist())
        rho_options = sorted(results.rho.unique().tolist())
        sigma2_options = sorted(results.sigma2.unique().tolist())

        M_slider   = widgets.SelectionSlider(options=M_options,   value=M_options[0],   description='M:',   continuous_update=False, layout=widgets.Layout(width='400px'))
        rho_slider = widgets.SelectionSlider(options=rho_options, value=rho_options[0], description='ρ:',   continuous_update=False, layout=widgets.Layout(width='400px'))
        sigma2_slider = widgets.SelectionSlider(options=sigma2_options, value=sigma2_options[0], description='σ²:', continuous_update=False, layout=widgets.Layout(width='400px'))
        ui = widgets.VBox([M_slider, rho_slider,sigma2_slider])

        def update(selected_M, selected_rho, selected_sigma2):
            plt.close('all')
            K = len(G_sampler_ls)
            fig, axs = plt.subplots(2, K, figsize=(20, 10 * (2/3)), gridspec_kw={"hspace": 0.35, "wspace": 0.25})

            for idx, G in enumerate(G_sampler_ls):
                df_G = results[(results.G == G.name)
                            & (results.sigma2 == selected_sigma2)].copy().sort_values('num_samples')
                df_rec = df_G[(df_G.M == selected_M)
                            & (df_G.rho == selected_rho)
                            & (df_G.sigma2 == selected_sigma2)].sort_values("num_samples")

                df_cv_G = None
                NMC = params.get('NMC', 1)
                if results_ocv.mean_cv_loss.values[0] ==results_ocv.mean_cv_loss.values[0]:
                    df_cv_G = results_ocv[(results_ocv.G == G.name) & (results_ocv.sigma2 == selected_sigma2)].copy().sort_values('num_samples')

                for i, variable in enumerate(['excess_loss', 'displacement']):
                    ax = axs[i, idx]
                    col = 'mean_' + variable

                    # Baseline
                    if i == 0:
                        ax.axhline(results_ocv.loc[(results_ocv.G == G.name), 'mean_naive_loss'].values.mean(), color='C0', linestyle='--', lw=2, alpha=0.7)

                    # 1) selected
                    if not cvonly:
                        x = df_rec["num_samples"].to_numpy(dtype=float)
                        y = df_rec[col].to_numpy(dtype=float)
                        ci = 1.96 * df_rec['std_' + variable].to_numpy(dtype=float) / np.sqrt(max(NMC, 1))
                        # pos = y[np.isfinite(y) & (y > 0)]
                        # eps = max(1e-10, (pos.min() / 10) if pos.size else 1e-10)
                        if eps is None:
                            y_plot = y
                            y_lo   = y - ci
                            y_hi   = y + ci
                        else:
                            y_plot = np.clip(y, eps, None)
                            y_lo   = np.clip(y - ci, eps, None)
                            y_hi   = np.clip(y + ci, eps, None)
                        ax.plot(x, y_plot, label=f"M={selected_M}, ρ={selected_rho}", color='C2', lw=2)
                        ax.fill_between(x, y_lo, y_hi, alpha=0.1, color='C2')
                        try:
                            valid = np.isfinite(x) & np.isfinite(y_plot) & (x > 0) & (y_plot > 0)

                            b = np.polyfit(np.log(x[valid]), np.log(y_plot[valid]), 1)[0] if valid.sum() > 1 else 0.0
                        except Exception:
                            b = np.nan

                        # 2) oracle 
                        best_idx = df_G.groupby('num_samples')[col].idxmin()
                        df_oc = df_G.loc[best_idx].sort_values('num_samples')
                        x_oc = df_oc["num_samples"].to_numpy(dtype=float)
                        y_oc = df_oc[col].to_numpy(dtype=float)
                        if eps is None:
                            y_oc_plot = y_oc
                        else:
                            y_oc_plot = np.clip(y_oc, eps, None)

                        valid_oc = np.isfinite(x_oc) & np.isfinite(y_oc_plot) & (x_oc > 0) & (y_oc_plot > 0)
                        if valid_oc.sum() > 1:
                            b_oc = np.polyfit(np.log(x_oc[valid_oc]), np.log(y_oc_plot[valid_oc]), 1)[0]
                        else:
                            b_oc = np.nan

                        ax.plot(x_oc, y_oc_plot, linestyle='--', color='black', alpha=0.6, label="Oracle (Best M,ρ)")
                        for _, row in df_oc.iterrows():
                            y = max(float(row[col]), eps) if eps is not None else float(row[col])
                            ax.annotate(f"M={int(row.M)}\nρ={row.rho:.3f}",
                                        xy=(row.num_samples, y),
                                        fontsize=7, color='black', ha='center', va='top')

                    # 3) CV 
                    b_cv = np.nan
                    if df_cv_G is not None and f"mean_cv_{variable}" in df_cv_G:
                        cv_col = f"mean_cv_{variable}"
                        x_cv = df_cv_G["num_samples"].to_numpy(dtype=float)
                        y_cv = df_cv_G[cv_col].to_numpy(dtype=float)
                        if eps is not None:
                            y_cv_plot = np.clip(y_cv, eps, None)
                        else:
                            y_cv_plot = y_cv

                        ax.plot(x_cv, y_cv_plot, ':', color='tab:red', lw=2, label="Score-Match CV")
                        for _, row in df_cv_G.iterrows():
                            y = max(float(row[cv_col]), eps) if eps is not None else float(row[cv_col])
                            ax.annotate(f"M={int(np.median(row.cv_Ms_star))}\nρ={row.cv_rhos_star.mean():.3f}",
                                        xy=(row.num_samples, y),
                                        fontsize=7, color='tab:red', ha='center', va='top')

                        valid_cv = np.isfinite(x_cv) & np.isfinite(y_cv_plot) & (x_cv > 0) & (y_cv_plot > 0)
                        if valid_cv.sum() > 1:
                            b_cv = np.polyfit(np.log(x_cv[valid_cv]), np.log(y_cv_plot[valid_cv]), 1)[0]

                    ax.set_title(f"{variable}\nSlope: {b_cv:.2f}")
                    # ax.set_title(f"{variable}\nSlope: Oracle ={b_oc:.2f} Sel={b:.2f} | CV={b_cv:.2f}")

                    # 8) outer selected 
                    if extselected_Mrho is not None:
                        _y = []
                        for _in, n in enumerate(df_G.num_samples.unique()):
                            _rho = extselected_Mrho[G.name]['rho'][_in]
                            _M = extselected_Mrho[G.name]['M'][_in]
                            val = df_G[(df_G.num_samples == n) & (df_G.M == _M) & (df_G.rho == _rho)][col].values[0]
                            _y.append(val)
                        _y = np.asarray(_y, dtype=float)
                        ax.plot(df_G.num_samples.unique(), np.clip(_y, eps, None),
                                linestyle='-.', color='tab:purple', lw=2, label="Outer Selected")

            # Set scales/legend once (after all plotting)
            handles, labels = [], []
            for ax in axs.ravel():
                ax.set_xscale("log")
                if _axis_all_positive_finite(ax):
                    ax.set_yscale("log")
                    # None
                else:
                    ax.set_yscale("symlog", linthresh=1e-10)
                    # ax.set_yscale("linear")  # or: 

                ax.grid(True, which="both", ls="--", alpha=0.3)

                h, l = ax.get_legend_handles_labels()
                for hh, ll in zip(h, l):
                    if ll and ll not in labels:
                        handles.append(hh)
                        labels.append(ll)
                if ax.get_legend() is not None:
                    ax.legend_.remove()

            # y-label on first column (if it ended up log)
            for i in range(axs.shape[0]):
                if axs[i, 0].get_yscale() == "log":
                    axs[i, 0].set_ylabel("Log Error")

            if handles:
                fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0),
                        ncol=len(labels), fontsize=12, frameon=False)
                fig.subplots_adjust(bottom=0.18)

            g_images = [widgets.Image(value=_cached_G_image(manifold_type, G), format='png', width=250) for G in G_sampler_ls]
            display(widgets.HBox(g_images, layout=widgets.Layout(justify_content='center')))
            plt.show()

        out = widgets.interactive_output(update, {'selected_M': M_slider, 'selected_rho': rho_slider, 'selected_sigma2': sigma2_slider})
        display(ui, out)
        return None

except ImportError:
    print("ipywidgets not found. Interactive plotting is disabled.")
    def plot_mcratesims_interactive(*args, **kwargs):
        print("Interactive plotting is disabled because ipywidgets is not installed.")
        return None
