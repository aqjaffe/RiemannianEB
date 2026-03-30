from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from .helpers import *
from .density_estimation import *
from .plotting import *

def plot_G(manifold_type, G, fig, ax):
    if manifold_type == "S1":
        manifold = Hypersphere(1)
        ss = ax.get_subplotspec()
        ax.remove()
        ax = fig.add_subplot(ss, polar=True)
        ax.set_title(f"'{G.name}'", fontsize=14)

        Theta = G.sample(1000)
        Theta = manifold.intrinsic_to_extrinsic_coords(manifold.extrinsic_to_intrinsic_coords(Theta) - np.pi / 12)

        f_scale = 0.3
        bottom = 0.105
        top = .5
        disk_r = 0.1
        grid_I = np.linspace(0, 2*np.pi, 50)
        on_X = manifold.intrinsic_to_extrinsic_coords( grid_I[:, None])
        hat_f = kernel_density_estimate("S1", Theta,  on_X, 9)[1]
        hat_pos_f = np.maximum(hat_f, 0)
        normalised_hat_f = (hat_pos_f - .9*hat_pos_f.min()) / (1.1*hat_pos_f.max() - .9*hat_pos_f.min() + 1e-10)
        
        verts = [[
                (grid_I[i], bottom),
                (grid_I[i], bottom + f_scale * hat_pos_f[i]), (grid_I[i+1], bottom + f_scale * hat_pos_f[i+1]),
                # (grid_I[i],top), (grid_I[i+1], bottom + top),
                (grid_I[i+1], bottom)
            ] for i in range(len(grid_I) - 1)] # Create polygon vertices for each segment
        poly = PolyCollection(verts, facecolors=plt.colormaps['Reds'](normalised_hat_f[:-1]), 
                            alpha=0.85, edgecolors='none')
        ax.add_collection(poly)

        ax.set_ylim(0, bottom + hat_f.max()*f_scale)  
        ax.set_yticks([])
        ax.bar(0, disk_r, width=2*np.pi, bottom=0, color="white", edgecolor="none", align="edge", zorder=3)
        ax.plot(grid_I, disk_r*np.ones_like(grid_I), color='black', linewidth=1.2, zorder=4)
        # S1_histogram(Theta, 30, ax, "Reds")

    elif manifold_type == "S2":
        manifold = Hypersphere(2)
        ss = ax.get_subplotspec()
        ax.remove()
        ax = fig.add_subplot(ss, projection="mollweide")
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        ax.set_title(f"'{G.name}'", fontsize=14)

        grid_resolution = 100
        grid, grid_theta, grid_phi = S2grid(grid_resolution)
        hat_f = kernel_density_estimate("S2", G.sample(1000),grid, 20)[1].reshape(
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


def plot_mcratesims(manifold_type, results_mc, M, rho, params):
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
    
    selected_sigma2 = params['sigma2']
    NMC = params.get('NMC', 1)
    G_sampler_ls = get_G_sampler_ls_from_params(params)

    K = len(G_sampler_ls)
    fig, axs = plt.subplots(3, K, figsize=(20, 10), gridspec_kw={"hspace": 0.35, "wspace": 0.25})

    for idx, G in enumerate(G_sampler_ls):
        
        plot_G(manifold_type, G, fig, axs[0, idx])

        df_G = results_mc[(results_mc.G == G.name)
                & (results_mc.sigma2 == selected_sigma2)
                    ].sort_values("num_samples")
    
        df_rec = results_mc[(results_mc.G == G.name)
                        & (results_mc.sigma2 == selected_sigma2)
                        & (df_G.rho == rho)
                        & (df_G.M == M)
                        ].sort_values("num_samples")

        for i, variable in enumerate(['excess_loss', 'displacement']):
            i+=1
            ax = axs[i, idx]
            col = 'mean_' + variable

            # # Baseline
            # if i == 0 and df_cv_G is not None and 'mean_naive_loss' in df_cv_G:
            #     ax.axhline(df_cv_G['mean_naive_loss'].values.mean(), color='C0', linestyle='--', lw=2, alpha=0.7)

            x = df_rec["num_samples"].to_numpy(dtype=float)
            y = df_rec[col].to_numpy(dtype=float)
            ci = 1.96 * df_rec['std_' + variable].to_numpy(dtype=float) / np.sqrt(max(NMC, 1))
            pos = y[np.isfinite(y) & (y > 0)]
            eps = max(1e-10, (pos.min() / 10) if pos.size else 1e-10)
            y_plot = np.clip(y, eps, None)
            y_lo   = np.clip(y - ci, eps, None)
            y_hi   = np.clip(y + ci, eps, None)

            ax.plot(x, y_plot, label=f"M={M}, ρ={rho}", color='C2', lw=2)
            ax.fill_between(x, y_lo, y_hi, alpha=0.1, color='C2')

            try:
                valid = np.isfinite(x) & np.isfinite(y_plot) & (x > 0) & (y_plot > 0)
                b = np.polyfit(np.log(x[valid]), np.log(y_plot[valid]), 1)[0] if valid.sum() > 1 else 0.0
            except Exception:
                b = np.nan

            best_idx = df_G.groupby('num_samples')[col].idxmin()
            df_oc = df_G.loc[best_idx].sort_values('num_samples')
            x = df_oc["num_samples"].to_numpy(dtype=float)
            y = df_oc[col].to_numpy(dtype=float)
            y_plot = np.clip(y, eps, None)
            try:
                valid = np.isfinite(x) & np.isfinite(y_plot) & (x > 0) & (y_plot > 0)
                b_o = np.polyfit(np.log(x[valid]), np.log(y_plot[valid]), 1)[0] if valid.sum() > 1 else 0.0
            except Exception:
                b_o = np.nan

            ax.plot(x, y_plot, linestyle='--', color='black', alpha=0.6, label="Oracle (Best M,ρ)")
            for _, row in df_oc.iterrows():
                ax.annotate(f"M={int(row.M)}\nρ={row.rho:.3f}",
                            xy=(row.num_samples, max(float(row[col]), eps)),
                            fontsize=7, color='black', ha='center', va='top')
            ax.set_title(f"{variable}\nMC rate slope: {b:.2f}, Oracle slope: {b_o:.2f}", fontsize=12)


    # Set scales/legend once (after all plotting)
    handles, labels = [], []
    for ax in axs.ravel():
        ax.set_xscale("log")
        if _axis_all_positive_finite(ax):
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=1e-10)  

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

    plt.show()


    return None


def plot_mcsims(manifold_type, df_sigma, df_N, G_sampler_ls, savefig=None):
    df_long, df_summary = df_sigma, df_N

    K = len(G_sampler_ls)

    # Make row 0 shorter than rows 1 and 2
    fig, axs = plt.subplots(
        3, K,
        figsize=(20, 10),
        gridspec_kw={"height_ratios": [0.65, 1.0, 1.0], "hspace": 0.35, "wspace": 0.25},
    )

    for idx, G in enumerate(G_sampler_ls):
        plot_G(manifold_type, G, fig, axs[0, idx])

        df_subset = df_long[df_long["G"] == G.name]
        sns.lineplot(
            data=df_subset,
            x="sigma2",
            y="Loss",
            hue="Loss Type",
            hue_order=["Naïve", "Oracle Denoised", "Oracle Bayes"],
            palette={"Naïve": "C0","Oracle Denoised": "C2","Oracle Bayes": "C4",},
            style="Loss Type",
            dashes={"Naïve": "","Oracle Denoised": (1, 1),"Oracle Bayes": (1, 1),},
            estimator="mean",
            errorbar=("ci", 68),
            marker="o",
            ax=axs[1, idx],
        )

       
        axs[1, idx].set_xlabel("σ²")
        axs[1, idx].set_ylabel("$\mathcal{R}(\delta)$")
        axs[1, idx].tick_params(axis="x", rotation=45)


        # ===== row 2: plot median diff vs N, colored by sigma2 =====
        ax = axs[2, idx] 
        df_g = df_summary[df_summary["G"] == G.name].copy()
        
        sigma2_vals = np.sort(df_summary["sigma2"].unique())
        norm = mpl.colors.Normalize(vmin=float(np.min(sigma2_vals)), vmax=float(np.max(sigma2_vals)))
        cmap = mpl.cm.get_cmap("Greens")
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        for sigma2 in sigma2_vals:
            d = df_g[df_g["sigma2"] == sigma2].sort_values("num_samples")
            if len(d) == 0:continue
            color = cmap(norm(float(sigma2)))
            x = d["num_samples"].to_numpy()
            try:
                y = d["mean_emp_minus_oracle"].to_numpy()
            except:
                y = d["median_emp_minus_oracle"].to_numpy()
            ax.plot(x, y, linestyle = '-.', color=color, linewidth=1.5, label = f"σ²={sigma2:.2f}")
        ax.legend()
        ax.axhline(0.0, color="k", linewidth=1, alpha=0.35)
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("$\mathbb{E}[\mathcal{R}(\hat\delta_T) - \mathcal{R}(\delta_T)]$")    
        ax.tick_params(axis="x", rotation=45)
        ax.set_xscale("log"); ax.set_yscale("log")

        # share y across two bottom row
        for r in [1,2]:
            base = axs[r, 0]
            for c in range(1, len(G_sampler_ls)):
                axs[r, c].sharey(base)
                axs[r, c].set_ylabel("") 
                axs[r, c].tick_params(labelleft=False) 
       

    # legend
    labels_mapper = { "Naïve": "$\delta_N$", "Empirical Denoised": "\hat\delta_T", "Oracle Denoised": "$\delta_T$", "Oracle Bayes": "$\delta_B$"}
    for idx, _ in enumerate(G_sampler_ls):
        handles1, labels1 = axs[1, idx].get_legend_handles_labels()
        labels1 = [labels_mapper[label] for label in labels1]
        axs[1, idx].get_legend().remove()

        handles2, labels2 = axs[2, idx].get_legend_handles_labels()
        axs[2, idx].get_legend().remove()
        axs[1, -1].legend(handles1,labels1,loc="center left",bbox_to_anchor=(1.02, 0.5),ncol=1,frameon=False,fontsize=14,borderaxespad=0.0,)
        axs[2, -1].legend(handles2,labels2, loc="center left",bbox_to_anchor=(1.02, 0.5),ncol=1,frameon=False,fontsize=14,borderaxespad=0.0,)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(f"{savefig}", bbox_inches="tight")
    plt.show()
    return None







def plot_mcsims_IncreasingSigma(manifold_type, df_long, G_sampler_ls, savefig=None):
    fig, axs = plt.subplots(2, len(G_sampler_ls), figsize=(20, 8))

    for idx, G in enumerate(G_sampler_ls):

        if manifold_type == "S1":
            manifold = Hypersphere(1)
            axs[0, idx].remove()
            axs[0, idx] = fig.add_subplot(2, len(G_sampler_ls), idx + 1, polar=True)
            Theta = G.sample(1000)
            Theta = manifold.intrinsic_to_extrinsic_coords(
                manifold.extrinsic_to_intrinsic_coords(Theta) - np.pi/12
            )
            S1_histogram(Theta, 30, axs[0, idx], "Reds")

        elif manifold_type == "S2":
            manifold = Hypersphere(2)
            axs[0, idx].remove()
            axs[0, idx] = fig.add_subplot(2, len(G_sampler_ls), 1 + idx, projection="mollweide")
            axs[0, idx].set_xticks([]); axs[0, idx].set_yticks([])
            axs[0, idx].grid(True, alpha=0.3)
            axs[0, idx].set_title(f"G '{G.name}'", fontsize=14)
            grid_resolution = 100
            grid, grid_theta, grid_phi = S2grid(grid_resolution)
            hat_f =  kernel_density_estimate("S2", G.sample(1000), 20, grid)[1].reshape(grid_resolution, grid_resolution)
            im = axs[0, idx].pcolormesh(
                grid_phi - np.pi,
                np.pi / 2 - grid_theta,
                hat_f,
                alpha=0.8,
                shading="auto",
                cmap="Reds",
                vmin= .5 if G.name == 'uniform' else None, vmax = .5 if G.name == 'uniform' else None
                )

        else:
            raise ValueError("Unsupported manifold type. Supported types are 'S1' and 'S2'.")

        # ... your second row plotting code unchanged ...
        df_subset = df_long[df_long["G"] == G.name]
        sns.lineplot(
            data=df_subset,
            x="sigma2",
            y="Loss",
            hue="Loss Type",
            hue_order=[
                "Naïve", 
                # "Empirical Denoised",
                "Oracle Denoised", 
                "Oracle Bayes"
                ],
            palette={
                "Naïve": "C0",
                # "Empirical Denoised": "C2",
                "Oracle Denoised": "C2",
                "Oracle Bayes": "C4",
            },
            style="Loss Type",
            dashes={
                "Naïve": "",
                # "Empirical Denoised": "",
                "Oracle Denoised": (1, 1),
                "Oracle Bayes": (1, 1),
            },
            estimator="mean",
            errorbar=("ci", 68),
            marker="o",
            ax=axs[1, idx],
        )

        handles, labels = axs[1, idx].get_legend_handles_labels()
        labels_mapper = {
            "Naïve": "$\delta_N$",
            "Empirical Denoised": "\hat\delta_T",
            "Oracle Denoised": "$\delta_T$",
            "Oracle Bayes": "$\delta_B$"
        }
        labels = [labels_mapper[label] for label in labels]
        axs[1, idx].get_legend().remove()
        if idx == len(G_sampler_ls) - 1:
            fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize = 14)

        axs[1, idx].set_xlabel("σ²")
        axs[1, idx].set_ylabel("Average Loss")
        axs[1, idx].tick_params(axis="x", rotation=45)

        base = axs[1, 0]
        for c in range(1, axs.shape[1]):
            axs[1, c].sharey(base)
            axs[1, c].set_ylabel("")
            axs[1, c].tick_params(labelleft=False)


    plt.tight_layout()
    if savefig is not None:
        plt.savefig(f"{savefig}", bbox_inches="tight")
    plt.show()
    return None


def plot_mcsims_IncreasingN(
    manifold_type,
    df_summary,
    G_sampler_ls,
    savefig=None,
    sigma_cmap="viridis",
    showG = True
):
    """
    Plot outcome of mcsims_IncreasingN_summary (median/std of Empirical-Oracle),
    with one column per G and sigma2 encoded by a colorscale + colorbar.

    Expected df_summary columns:
      - G
      - sigma2
      - num_samples
      - rho_star
      - median_emp_minus_oracle
      - std_emp_minus_oracle
    """
    fig, axs = plt.subplots(2 if showG else 1, len(G_sampler_ls), figsize=(20, 8 if showG else 4))

    # ---- color mapping for sigma2 (shared across all panels) ----
    sigma2_vals = np.sort(df_summary["sigma2"].unique())
    norm = mpl.colors.Normalize(vmin=float(np.min(sigma2_vals)), vmax=float(np.max(sigma2_vals)))
    cmap = mpl.cm.get_cmap(sigma_cmap)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    for idx, G in enumerate(G_sampler_ls):
        if showG:
            # ===== row 0: show the target distribution for this G (same style as your other plot) =====
            if manifold_type == "S1":
                manifold = Hypersphere(1)
                axs[0, idx].remove()
                axs[0, idx] = fig.add_subplot(2, len(G_sampler_ls), idx + 1, polar=True)
                Theta = G.sample(1000)
                Theta = manifold.intrinsic_to_extrinsic_coords(
                    manifold.extrinsic_to_intrinsic_coords(Theta) - np.pi / 12
                )
                S1_histogram(Theta, 30, axs[0, idx], "Reds")
                axs[0, idx].set_title(f"G '{G.name}'", fontsize=14)

            elif manifold_type == "S2":
                manifold = Hypersphere(2)
                axs[0, idx].remove()
                axs[0, idx] = fig.add_subplot(2, len(G_sampler_ls), 1 + idx, projection="mollweide")
                axs[0, idx].set_xticks([])
                axs[0, idx].set_yticks([])
                axs[0, idx].grid(True, alpha=0.3)
                axs[0, idx].set_title(f"G '{G.name}'", fontsize=14)

                grid_resolution = 100
                grid, grid_theta, grid_phi = S2grid(grid_resolution)
                hat_f = kernel_density_estimate("S2", G.sample(1000), 20, grid)[1].reshape(
                    grid_resolution, grid_resolution
                )
                axs[0, idx].pcolormesh(
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

        # ===== row 1: plot median diff vs N, colored by sigma2 =====
        ax = axs[1, idx] if showG else axs[idx]
        df_g = df_summary[df_summary["G"] == G.name].copy()

        for sigma2 in sigma2_vals:
            d = df_g[df_g["sigma2"] == sigma2].sort_values("num_samples")
            if len(d) == 0:
                continue

            color = cmap(norm(float(sigma2)))
            x = d["num_samples"].to_numpy()
            y = d["median_emp_minus_oracle"].to_numpy()


            ax.plot(x, y, color=color, marker="o", linewidth=2)
            # s = d["std_emp_minus_oracle"].to_numpy()
            # ax.fill_between(x, y - s, y + s, color=color, alpha=0.20, linewidth=0)

        ax.axhline(0.0, color="k", linewidth=1, alpha=0.35)
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("Median(Empirical − Oracle)")
        ax.tick_params(axis="x", rotation=45)

        # share y across bottom row
        base = axs[1, 0] if showG else axs[0]
        for c in range(1, len(G_sampler_ls)):
            axs[1, c].sharey(base) if showG else axs[c].sharey(base)
            axs[1, c].set_ylabel("") if showG else axs[c].set_ylabel("")
            axs[1, c].tick_params(labelleft=False) if showG else axs[c].tick_params(labelleft=False)

    plt.tight_layout()

    cax = fig.add_axes([0.25, -0.06, 0.50, 0.03])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("σ²")

    
    if savefig is not None:
        plt.savefig(f"{savefig}", bbox_inches="tight")
    plt.show()
    return None

# sns.lineplot(
#             data=df_long[df_long['num_modes'] == num_modes],
#             x="num_samples",
#             y="Loss",
#             hue="Loss Type",
#             hue_order=["Naïve","Empirical Denoised", "Oracle Denoised", "Oracle Bayes"],
#             palette={
#                 "Naïve": "C0",
#                 "Empirical Denoised": "C2",
#                 "Oracle Denoised": "C2",
#                 "Oracle Bayes": "C4",
#             },
#             style="Loss Type",  # map linestyle to hue categories
#             dashes={
#                 "Naïve": "",
#                 "Empirical Denoised": "",
#                 "Oracle Denoised": (1, 1),
#                 "Oracle Bayes": (1, 1),
#             },
#             estimator="mean",
#             errorbar=("ci", 68),  # 1-sigma style band
#             marker="o",
#             ax=axs[1, idx],
#         )