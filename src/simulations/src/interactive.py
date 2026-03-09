import ipywidgets as widgets
from IPython.display import display
from functools import lru_cache
import io
from utils import *
from .plot import *

import ipywidgets as widgets
from IPython.display import display
from functools import lru_cache
import io
from utils import *
from .plot import *


def plot_mcratesims_interactive(manifold_type, results, results_ocv, params, extselected_Mrho=None):
    selected_sigma2 = params['sigma2']
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

    M_slider   = widgets.SelectionSlider(options=M_options,   value=M_options[0],   description='M:',   continuous_update=False, layout=widgets.Layout(width='400px'))
    rho_slider = widgets.SelectionSlider(options=rho_options, value=rho_options[0], description='ρ:',   continuous_update=False, layout=widgets.Layout(width='400px'))

    ui = widgets.VBox([M_slider, rho_slider])

    def update(selected_M, selected_rho):
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
                df_cv_G = results_ocv[(results_ocv.G == G.name)].copy().sort_values('num_samples')

            for i, variable in enumerate(['excess_loss', 'displacement']):
                ax = axs[i, idx]
                col = 'mean_' + variable

                # Baseline
                if i == 0:
                    ax.axhline(results_ocv.loc[(results_ocv.G == G.name), 'mean_naive_loss'].values.mean(), color='C0', linestyle='--', lw=2, alpha=0.7)

                # 1) selected
                x = df_rec["num_samples"].to_numpy(dtype=float)
                y = df_rec[col].to_numpy(dtype=float)

                ci = 1.96 * df_rec['std_' + variable].to_numpy(dtype=float) / np.sqrt(max(NMC, 1))
                pos = y[np.isfinite(y) & (y > 0)]
                eps = max(1e-10, (pos.min() / 10) if pos.size else 1e-10)

                y_plot = np.clip(y, eps, None)
                y_lo   = np.clip(y - ci, eps, None)
                y_hi   = np.clip(y + ci, eps, None)

                ax.plot(x, y_plot, label=f"M={selected_M}, ρ={selected_rho}", color='C2', lw=2)
                ax.fill_between(x, y_lo, y_hi, alpha=0.1, color='C2')

                valid = np.isfinite(x) & np.isfinite(y_plot) & (x > 0) & (y_plot > 0)
                try:
                    b = np.polyfit(np.log(x[valid]), np.log(y_plot[valid]), 1)[0] if valid.sum() > 1 else 0.0
                except Exception:
                    b = np.nan

                # 2) oracle 
                best_idx = df_G.groupby('num_samples')[col].idxmin()
                df_oc = df_G.loc[best_idx].sort_values('num_samples')
                x_oc = df_oc["num_samples"].to_numpy(dtype=float)
                y_oc = df_oc[col].to_numpy(dtype=float)
                y_oc_plot = np.clip(y_oc, eps, None)

                valid_oc = np.isfinite(x_oc) & np.isfinite(y_oc_plot) & (x_oc > 0) & (y_oc_plot > 0)
                if valid_oc.sum() > 1:
                    b_oc = np.polyfit(np.log(x_oc[valid_oc]), np.log(y_oc_plot[valid_oc]), 1)[0]
                else:
                    b_oc = np.nan

                ax.plot(x_oc, y_oc_plot, linestyle='--', color='black', alpha=0.6, label="Oracle (Best M,ρ)")
                for _, row in df_oc.iterrows():
                    ax.annotate(f"M={int(row.M)}\nρ={row.rho:.3f}",
                                xy=(row.num_samples, max(float(row[col]), eps)),
                                fontsize=7, color='black', ha='center', va='top')

                # 3) CV 
                b_cv = np.nan
                if df_cv_G is not None and f"mean_cv_{variable}" in df_cv_G:
                    cv_col = f"mean_cv_{variable}"
                    x_cv = df_cv_G["num_samples"].to_numpy(dtype=float)
                    y_cv = df_cv_G[cv_col].to_numpy(dtype=float)
                    y_cv_plot = np.clip(y_cv, eps, None)

                    ax.plot(x_cv, y_cv_plot, ':', color='tab:red', lw=2, label="Score-Match CV")
                    for _, row in df_cv_G.iterrows():
                        ax.annotate(f"M={int(np.median(row.cv_Ms_star))}\nρ={row.cv_rhos_star.mean():.3f}",
                                    xy=(row.num_samples, max(float(row[cv_col]), eps)),
                                    fontsize=7, color='tab:red', ha='center', va='top')

                    valid_cv = np.isfinite(x_cv) & np.isfinite(y_cv_plot) & (x_cv > 0) & (y_cv_plot > 0)
                    if valid_cv.sum() > 1:
                        b_cv = np.polyfit(np.log(x_cv[valid_cv]), np.log(y_cv_plot[valid_cv]), 1)[0]

                ax.set_title(f"{variable}\nSlope: Oracle ={b_oc:.2f} Sel={b:.2f} | CV={b_cv:.2f}")

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
            else:
                ax.set_yscale("linear")  # or: ax.set_yscale("symlog", linthresh=1e-10)

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

    out = widgets.interactive_output(update, {'selected_M': M_slider, 'selected_rho': rho_slider})
    display(ui, out)
    return None


# def plot_mcratesims_interactive(manifold_type, results, results_ocv, params, extselected_Mrho=None):
#     ID, selected_sigma2 = float(params['ID']), params['sigma2']
#     G_sampler_ls = get_G_sampler_ls_from_params(params)
#     @lru_cache(maxsize=None)
#     def _cached_G_image(manifold_type, G):
#         """Render plot_G to a PNG bytes buffer and cache it."""
#         fig_g, ax_g = plt.subplots(1, 1, figsize=(20/len(G_sampler_ls), 10 * 0.65 / 3))
#         plot_G(manifold_type, G, fig_g, ax_g)
#         buf = io.BytesIO()
#         fig_g.savefig(buf, format='png', bbox_inches='tight')
#         plt.close(fig_g)
#         buf.seek(0)
#         return buf.read()  # returns raw PNG bytes

#     M_options   = sorted(results.M.unique().tolist())
#     rho_options = sorted(results.rho.unique().tolist())

#     M_slider   = widgets.SelectionSlider(options=M_options,   value=M_options[0],   description='M:',   continuous_update=False, layout=widgets.Layout(width='400px'))
#     rho_slider = widgets.SelectionSlider(options=rho_options, value=rho_options[0], description='ρ:',   continuous_update=False, layout=widgets.Layout(width='400px'))

#     ui = widgets.VBox([M_slider, rho_slider])

#     def update(selected_M, selected_rho):
#         plt.close('all')
#         K = len(G_sampler_ls)
#         fig, axs = plt.subplots(2, K, figsize=(20, 10 * (2/3)), gridspec_kw={"hspace": 0.35, "wspace": 0.25})

#         for idx, G in enumerate(G_sampler_ls):
#             # Filter main results for the interactive lines
#             df_G = results[(results.G == G.name)
#                             & (results.ID == ID)
#                             & (results.sigma2 == selected_sigma2)].copy().sort_values('num_samples')
#             df_rec = df_G[(df_G.M == selected_M)
#                           & (df_G.rho == selected_rho)
#                           & (df_G.sigma2 == selected_sigma2)].sort_values("num_samples")
#             # Filter CV results
#             df_cv_G = None
#             if results_ocv is not None:
#                 df_cv_G = results_ocv[(results_ocv.G == G.name) & (results_ocv.ID == ID)].copy().sort_values('num_samples')
#                 NMC = params['NMC']
#             for i, variable in enumerate(['excess_loss', 'displacement']):
#                 ax = axs[i, idx]
#                 if i == 0:
#                     ax.axhline(df_cv_G['mean_naive_loss'].values.mean(), color='C0', linestyle='--', lw=2, alpha=0.7)
#                 col = 'mean_' + variable
#                 # 1. INTERACTIVE SELECTION (Blue Line)
#                 x = df_rec["num_samples"].to_numpy(dtype=float)
#                 y = df_rec[col].to_numpy(dtype=float)
#                 ci = 1.96 * df_rec['std_' + variable].to_numpy(dtype=float) / np.sqrt(NMC)
#                 eps = max(1e-10, y[y > 0].min() / 10 if any(y > 0) else 1e-10)
#                 y_plot, y_lo, y_hi = np.clip(y, eps, None), np.clip(y - ci, eps, None), np.clip(y + ci, eps, None)
#                 ax.plot(x, y_plot, label=f"M={selected_M}, ρ={selected_rho}", color='C2', lw=2)
#                 ax.fill_between(x, y_lo, y_hi, alpha=0.1, color='C2')
#                 valid = (x > 0) & (y_plot > eps)
#                 try:
#                     b = np.polyfit(np.log(x[valid]), np.log(y_plot[valid]), 1)[0] if valid.sum() > 1 else 0.0
#                 except:
#                     b = np.nan
            
#                 # 2. ORACLE (Black Dashed - Best in Hindsight)
#                 best_idx = df_G.groupby('num_samples')[col].idxmin()
#                 df_oc = df_G.loc[best_idx].sort_values('num_samples')
#                 x_oc = df_oc["num_samples"].to_numpy(dtype=float)
#                 y_oc = df_oc[col].to_numpy(dtype=float)
#                 ax.plot(x_oc, y_oc, linestyle='--', color='black', alpha=0.6, label="Oracle (Best M,ρ)")
#                 for _, row in df_oc.iterrows():
#                     ax.annotate(f"M={int(row.M)}\nρ={row.rho:.3f}",
#                                 xy=(row.num_samples, row[col]),fontsize=7,color='black',ha='center',va='top')
                  
#                 # 3. SCORE-MATCHING CV (Red Dotted - CV Choice)
#                 b_cv = 0.0
#                 if results_ocv.mean_cv_loss.values[0] == results_ocv.mean_cv_loss.values[0]:
#                     cv_col = f"mean_cv_{variable}"
#                     x_cv = df_cv_G["num_samples"].to_numpy(dtype=float)
#                     y_cv = df_cv_G[cv_col].to_numpy(dtype=float)
#                     ax.plot(x_cv, y_cv, ':', color='tab:red', lw=2, label="Score-Match CV")
#                     for _, row in df_cv_G.iterrows():
#                         ax.annotate(f"M={int(np.median(row.cv_Ms_star))}\nρ={row.cv_rhos_star.mean():.3f}",
#                                     xy=(row.num_samples, row[cv_col]), fontsize=7, color='tab:red', ha='center', va='top')
#                     valid_cv = (x_cv > 0) & (y_cv > 0)
#                     if valid_cv.sum() > 1:
#                         b_cv = np.polyfit(np.log(x_cv[valid_cv]), np.log(y_cv[valid_cv]), 1)[0]
#                 else: b_cv = np.nan
#                 ax.set_title(f"{variable}\nSlope: Sel={b:.2f} | CV={b_cv:.2f}")


#                 # 8. OUTER SELECTED (Purple Dashed)
#                 if extselected_Mrho is not None:   
#                     _y = []
#                     for _in, n in enumerate(df_G.num_samples.unique()):
#                        _rho = extselected_Mrho[G.name]['rho'][_in]
#                        _M = extselected_Mrho[G.name]['M'] [_in]
#                        _y.append(df_G[(df_G.num_samples == n) & (df_G.M == _M) & (df_G.rho == _rho)][col].values[0])
#                     ax.plot(df_G.num_samples.unique(), _y, linestyle='-.', color='tab:purple', lw=2, label="Outer Selected")
                   

#                 handles, labels = [], []
#                 for ax in axs.ravel():
#                     ax.set_xscale("log"); 
#                     try: 
#                         ax.set_yscale("log")
#                         if idx == 0: ax.set_ylabel("Log Error")
#                     except:
#                         pass
#                     ax.grid(True, which="both", ls="--", alpha=0.3)
                   


#                     h, l = ax.get_legend_handles_labels()
#                     for hh, ll in zip(h, l):
#                         if ll and ll not in labels: handles.append(hh); labels.append(ll)
#                     ax.legend_.remove() if ax.get_legend() is not None else None
#                 if handles:
#                     fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(0.5, 0),ncol=len(labels),fontsize=12,frameon=False,)
#                     fig.subplots_adjust(bottom=0.18)
               
#         g_images = [widgets.Image(value=_cached_G_image(manifold_type, G), format='png', width=250) for G in G_sampler_ls]
#         display(widgets.HBox(g_images, layout=widgets.Layout(justify_content='center')))
#         plt.show()

#     out = widgets.interactive_output(update, {'selected_M': M_slider, 'selected_rho': rho_slider})
#     display(ui, out)
#     return None
