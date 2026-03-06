import ipywidgets as widgets
from IPython.display import display
from functools import lru_cache
import io
from utils import *
from .plot import *
def plot_mcratesims_interactive(manifold_type, results, results_ocv, G_sampler_ls, selected_NMC, selected_sigma2, extselected_Mrho=None):
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
            # Filter main results for the interactive lines
            df_G = results[(results.G == G.name)
                            & (results.NMC == selected_NMC)
                            & (results.sigma2 == selected_sigma2)].copy().sort_values('num_samples')
            df_rec = df_G[(df_G.M == selected_M)
                          & (df_G.rho == selected_rho)
                          & (results.sigma2 == selected_sigma2)].sort_values("num_samples")
            
            # Filter CV results
            df_cv_G = None
            if results_ocv is not None:
                df_cv_G = results_ocv[(results_ocv.G == G.name) & (results_ocv.NMC == selected_NMC)].copy().sort_values('num_samples')

            for i, variable in enumerate(['excess_loss', 'displacement']):
                ax = axs[i, idx]
                if i == 0:
                    ax.axhline(df_cv_G['mean_naive_loss'].values.mean(), color='gray', linestyle='--', lw=1, alpha=0.7)
                col = 'mean_' + variable
                
                # 1. INTERACTIVE SELECTION (Blue Line)
                if not df_rec.empty:
                    x = df_rec["num_samples"].to_numpy(dtype=float)
                    y = df_rec[col].to_numpy(dtype=float)
                    ci = 1.96 * df_rec['std_' + variable].to_numpy(dtype=float) / np.sqrt(selected_NMC)
                    eps = max(1e-10, y[y > 0].min() / 10 if any(y > 0) else 1e-10)
                    y_plot, y_lo, y_hi = np.clip(y, eps, None), np.clip(y - ci, eps, None), np.clip(y + ci, eps, None)
                    ax.plot(x, y_plot, label=f"M={selected_M}, ρ={selected_rho}", color='tab:blue', lw=2)
                    ax.fill_between(x, y_lo, y_hi, alpha=0.1, color='tab:blue')
                    valid = (x > 0) & (y_plot > eps)
                    b = np.polyfit(np.log(x[valid]), np.log(y_plot[valid]), 1)[0] if valid.sum() > 1 else 0.0
                
                # 2. ORACLE (Black Dashed - Best in Hindsight)
                best_idx = df_G.groupby('num_samples')[col].idxmin()
                df_oc = df_G.loc[best_idx].sort_values('num_samples')
                x_oc = df_oc["num_samples"].to_numpy(dtype=float)
                y_oc = df_oc[col].to_numpy(dtype=float)
                ax.plot(x_oc, y_oc, linestyle='--', color='black', alpha=0.6, label="Oracle (Best M,ρ)")
                for _, row in df_oc.iterrows():
                    ax.annotate(f"M={int(row.M)}\nρ={row.rho:.3f}",
                                xy=(row.num_samples, row[col]),fontsize=7,color='black',ha='center',va='top')
                  
                # 3. SCORE-MATCHING CV (Red Dotted - CV Choice)
                b_cv = 0.0
                if df_cv_G is not None:
                    cv_col = f"mean_cv_{variable}"
                    x_cv = df_cv_G["num_samples"].to_numpy(dtype=float)
                    y_cv = df_cv_G[cv_col].to_numpy(dtype=float)
                    ax.plot(x_cv, y_cv, ':', color='tab:red', lw=2, label="Score-Match CV")
                    for _, row in df_cv_G.iterrows():
                        ax.annotate(f"M={int(np.median(row.cv_Ms_star))}\nρ={row.cv_rhos_star.mean():.3f}",
                                    xy=(row.num_samples, row[cv_col]), fontsize=7, color='tab:red', ha='center', va='top')
                    valid_cv = (x_cv > 0) & (y_cv > 0)
                    if valid_cv.sum() > 1:
                        b_cv = np.polyfit(np.log(x_cv[valid_cv]), np.log(y_cv[valid_cv]), 1)[0]

                ax.set_title(f"{variable}\nSlope: Sel={b:.2f} | CV={b_cv:.2f}")


                # 8. OUTER SELECTED (Purple Dashed)
                if extselected_Mrho is not None:   
                    _y = []
                    for _in, n in enumerate(df_G.num_samples.unique()):
                       _rho = extselected_Mrho[G.name]['rho'][_in]
                       _M = extselected_Mrho[G.name]['M'] [_in]
                       _y.append(df_G[(df_G.num_samples == n) & (df_G.M == _M) & (df_G.rho == _rho)][col].values[0])
                    ax.plot(df_G.num_samples.unique(), _y, linestyle='-.', color='tab:purple', lw=2, label="Outer Selected")
                   

                handles, labels = [], []
                for ax in axs.ravel():
                    ax.set_xscale("log"); 
                    ax.set_yscale("log")
                    ax.grid(True, which="both", ls="--", alpha=0.3)
                    if idx == 0: ax.set_ylabel("Log Error")


                    h, l = ax.get_legend_handles_labels()
                    for hh, ll in zip(h, l):
                        if ll and ll not in labels: handles.append(hh); labels.append(ll)
                    ax.legend_.remove() if ax.get_legend() is not None else None
                if handles:
                    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(0.5, 0),ncol=len(labels),fontsize=12,frameon=False,)
                    fig.subplots_adjust(bottom=0.18)
               
        g_images = [widgets.Image(value=_cached_G_image(manifold_type, G), format='png', width=250) for G in G_sampler_ls]
        display(widgets.HBox(g_images, layout=widgets.Layout(justify_content='center')))
        plt.show()

    out = widgets.interactive_output(update, {'selected_M': M_slider, 'selected_rho': rho_slider})
    display(ui, out)
    return None
# def plot_mcratesims_interactive(manifold_type, results, G_sampler_ls, selected_NMC, cv_selected_M_df=None):
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

#         fig, axs = plt.subplots(2, K, figsize=(20, 10 * (2/3)),
#                         gridspec_kw={"hspace": 0.35, "wspace": 0.25},
#                         # constrained_layout=True
#                         )

#         for idx, G in enumerate(G_sampler_ls):

#             # --- oracle CV: best (M, rho) per n for each metric ---
#             df_G = results[
#                 (results.G == G.name) &
#                 (results.NMC == selected_NMC)
#             ].copy()
#             df_G['mean_excess_loss'] = df_G['mean_emp_loss'] - (df_G['mean_oracle_loss'] - df_G['std_oracle_loss'])

#             oracle_curves = {}
#             for variable in ['excess_loss', 'displacement']:
#                 col = 'mean_' + variable
#                 # for each n, pick the (M, rho) with the lowest mean value
#                 best_idx = df_G.groupby('num_samples')[col].idxmin()
#                 df_oracle = df_G.loc[best_idx].sort_values('num_samples')
#                 oracle_curves[variable] = df_oracle

#             # --- selected (M, rho) ---
#             df_rec = df_G[
#                 (df_G.M == selected_M) &
#                 (df_G.rho == selected_rho)
#             ].sort_values("num_samples").copy()

#             for i, variable in enumerate(['excess_loss', 'displacement']):
#                 ax = axs[i, idx]
#                 x  = df_rec["num_samples"].to_numpy(dtype=float)
#                 y  = df_rec["mean_" + variable].to_numpy(dtype=float)
#                 ci = 1.96 * df_rec["std_" + variable].to_numpy(dtype=float) / np.sqrt(selected_NMC)

#                 try:
#                     eps    = y[y > 0].min() / 10
#                     y_plot = np.clip(y, eps, None)
#                     y_lo   = np.clip(y - ci, eps, None)
#                     y_hi   = np.clip(y + ci, eps, None)

#                     ax.plot(x, y_plot, label=f"M={selected_M}, ρ={selected_rho}")
#                     ax.fill_between(x, y_lo, y_hi, alpha=0.2)
#                 except:
#                         ax.plot(x, y, label=f"M={selected_M}, ρ={selected_rho}")
#                         ax.fill_between(x, y - ci, y + ci, alpha=0.2)

#                 # oracle CV overlay
#                 df_oc  = oracle_curves[variable]
#                 x_oc   = df_oc["num_samples"].to_numpy(dtype=float)
#                 y_oc   = df_oc["mean_" + variable].to_numpy(dtype=float)
#                 ci_oc  = 1.96 * df_oc["std_" + variable].to_numpy(dtype=float) / np.sqrt(selected_NMC)
#                 eps_oc = y_oc[y_oc > 0].min() / 10
#                 y_oc_plot = np.clip(y_oc, eps_oc, None)
#                 y_oc_lo   = np.clip(y_oc - ci_oc, eps_oc, None)
#                 y_oc_hi   = np.clip(y_oc + ci_oc, eps_oc, None)

#                 ax.plot(x_oc, y_oc_plot, linestyle='--', color='black', label="oracle CV")
#                 ax.fill_between(x_oc, y_oc_lo, y_oc_hi, alpha=0.1, color='black')

#                 # annotate which (M, rho) was best at each n
#                 for _, row in df_oc.iterrows():
#                     ax.annotate(f"M={row.M}\nρ={row.rho}",
#                                 xy=(row.num_samples, np.clip(row["mean_" + variable], eps_oc, None)),
#                                 fontsize=6, ha='center', va='bottom', color='black', alpha=0.7)

#                 ax.set_xscale("log"); ax.set_yscale("log")
#                 ax.set_xlabel("n_samples")
#                 ax.grid(True, which="both", ls="--", alpha=0.4)
#                 ax.legend(fontsize=7)
                
#                 # --- data-driven CV curve ---
#                 if cv_selected_M_df is not None:
#                     df_cv_M = cv_selected_M_df[cv_selected_M_df.G == G.name].sort_values('n')

#                     cv_rows = []
#                     for _, cv_row in df_cv_M.iterrows():
#                         match = df_G[
#                             (df_G.num_samples == cv_row['n']) &
#                             (df_G.M == cv_row['M']) &
#                             (df_G.rho == selected_rho)
#                         ]
#                         if not match.empty:
#                             cv_rows.append(match.iloc[0])

#                     if cv_rows:
#                         df_cv = pd.DataFrame(cv_rows).sort_values('num_samples')
#                         x_cv    = df_cv["num_samples"].to_numpy(dtype=float)
#                         y_cv    = df_cv["mean_" + variable].to_numpy(dtype=float)
#                         ci_cv   = 1.96 * df_cv["std_" + variable].to_numpy(dtype=float) / np.sqrt(selected_NMC)
#                         eps_cv  = y_cv[y_cv > 0].min() / 10
#                         y_cv_plot = np.clip(y_cv, eps_cv, None)
#                         y_cv_lo   = np.clip(y_cv - ci_cv, eps_cv, None)
#                         y_cv_hi   = np.clip(y_cv + ci_cv, eps_cv, None)

#                         ax.plot(x_cv, y_cv_plot, linestyle=':', color='tab:red', label="data CV")
#                         ax.fill_between(x_cv, y_cv_lo, y_cv_hi, alpha=0.1, color='tab:red')

#                         # annotate selected M at each n
#                         for _, cv_row in df_cv_M.iterrows():
#                             match = df_G[(df_G.num_samples == cv_row['n']) & (df_G.M == cv_row['M']) & (df_G.rho == selected_rho)]
#                             if not match.empty:
#                                 y_ann = np.clip(match.iloc[0]["mean_" + variable], eps_cv, None)
#                                 ax.annotate(f"M={int(cv_row['M'])}",
#                                             xy=(cv_row['n'], y_ann),
#                                             fontsize=6, ha='center', va='top', color='tab:red', alpha=0.7)

#                         # slope
#                         valid_cv = np.log(y_cv_plot) > -np.inf
#                         if valid_cv.sum() >= 2:
#                             b_cv, _ = np.polyfit(np.log(x_cv)[valid_cv], np.log(y_cv_plot)[valid_cv], 1)
#                             current_title = ax.get_title()
#                             ax.set_title(current_title + f" | cv: {b_cv:.2f}")
#                 try:
#                     b, _ = np.polyfit(np.log(x)[np.log(y) == np.log(y)], np.log(y)[np.log(y) == np.log(y)], 1)
#                 except:
#                     b = 0

#                 valid_oc = np.log(y_oc_plot) > -np.inf
#                 b_oc, _ = np.polyfit(np.log(x_oc)[valid_oc], np.log(y_oc_plot)[valid_oc], 1)

#                 ax.set_title(f"{variable}  sel: {b:.2f} | oracle: {b_oc:.2f}")          
#                 # ax.set_aspect("equal", adjustable="datalim")

#         g_images = [widgets.Image(value=_cached_G_image(manifold_type, G), format='png') for G in G_sampler_ls]
#         g_row = widgets.HBox(g_images, layout=widgets.Layout(justify_content='space-around'))

#         out_fig = widgets.Output()
#         with out_fig:
#             plt.show()

#         display(widgets.VBox([g_row, out_fig]))

#     out = widgets.interactive_output(update, {'selected_M': M_slider, 'selected_rho': rho_slider})
#     display(ui, out)
#     return None