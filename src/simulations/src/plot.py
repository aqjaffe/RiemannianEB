from utils import *

def plot_mcsims_IncreasingSigma(manifold_type, df_long, num_modes_ls, tau2_ls, savefig = None):
    fig, axs = plt.subplots(2, len(num_modes_ls), figsize=(20, 8))
    for idx, (num_modes, tau2) in enumerate(zip(num_modes_ls, tau2_ls)):

        if manifold_type == 'S1':
            manifold = Hypersphere(1)
            axs[0, idx].remove()
            axs[0, idx] = fig.add_subplot( 2, len(num_modes_ls), idx + 1, polar=True)
            Theta = multimodal_sampler(  5000, manifold_type, {'tau2': tau2, 'num_modes': num_modes})
            Theta = manifold.intrinsic_to_extrinsic_coords( manifold.extrinsic_to_intrinsic_coords(Theta) - np.pi/12)
            S1_histogram(Theta, 30, axs[0, idx], 'Reds')

        elif manifold_type == 'S2':
            manifold = Hypersphere(2)
            axs[0, idx].remove()
            ax0 = fig.add_subplot(2, len(num_modes_ls), 1 + idx, projection="mollweide")
            axs[0, idx] = ax0
            axs[0, idx].set_xticks([])
            axs[0, idx].set_yticks([])
            axs[0, idx].grid(True, alpha=0.3)
            Theta = multimodal_sampler(  5000, manifold_type, {'tau2': tau2, 'num_modes': num_modes})
            grid_resolution = 100
            grid, grid_theta, grid_phi = S2grid(grid_resolution)
            _, hat_f, grad_hat_f, = kernel_density_estimate('S2', Theta, 20, grid)
            im = axs[0, idx].pcolormesh( grid_phi - np.pi, np.pi/2 - grid_theta, hat_f.reshape(grid_resolution,grid_resolution), alpha=0.8, shading='auto', cmap='Reds')

        else:
            ValueError ( "Unsupported manifold type. Supported types are 'S1' and 'S2'." )
        df_subset = df_long[df_long['num_modes'] == num_modes]

        sns.lineplot(
            data=df_subset,
            x="sigma2",
            y="Loss",
            hue="Loss Type",
            hue_order=["Naïve","Empirical Denoised", "Oracle Denoised", "Oracle Bayes"],
            palette={
                "Naïve": "C0",
                "Empirical Denoised": "C2",
                "Oracle Denoised": "C2",
                "Oracle Bayes": "C4",
            },
            style="Loss Type",  # map linestyle to hue categories
            dashes={
                "Naïve": "",
                "Empirical Denoised": "",
                "Oracle Denoised": (1, 1),
                "Oracle Bayes": (1, 1),
            },
            estimator="mean",
            errorbar=("ci", 68),  # 1-sigma style band; use 95 if you prefer
            marker="o",
            ax=axs[1, idx],
        )
        # single shared legend (one row) placed below all subplots
        handles, labels = axs[1, idx].get_legend_handles_labels()
        axs[1, idx].get_legend().remove()
        if idx == len(num_modes_ls) - 1:
            fig.legend(handles,labels,loc="lower center",ncol=len(labels),frameon=False,bbox_to_anchor=(0.5, -.02))
        # axs[1, idx].set_title(f"{num_modes} modes")
        axs[1, idx].set_xlabel("σ²")
        axs[1, idx].set_ylabel("Average Loss")
        axs[1, idx].tick_params(axis='x', rotation=45)
        

        # Share y-axes across rows and hide repeated y tick labels within each row
        base = axs[1, 0]
        for c in range(1, axs.shape[1]):
            axs[1, c].sharey(base)
            axs[1, c].set_ylabel("")
            axs[1, c].tick_params(labelleft=False)
    plt.tight_layout()
    if savefig is not None:
       plt.savefig('{}'.format(savefig), bbox_inches='tight')
    plt.show()
    return fig


def plot_mcsims_IncreasingN(manifold_type, df_long, num_modes_ls, tau2_ls, savefig = None):
    fig, axs = plt.subplots(2, len(num_modes_ls), figsize=(20, 8))

    for idx, (tau2, num_modes ) in enumerate(zip(tau2_ls, num_modes_ls)):

        if manifold_type == 'S1':
            manifold = Hypersphere(1)
            axs[0, idx].remove()
            axs[0, idx] = fig.add_subplot( 2, len(num_modes_ls), idx + 1, polar=True)
            Theta = multimodal_sampler(  5000, manifold_type, {'tau2': tau2, 'num_modes': num_modes})
            Theta = manifold.intrinsic_to_extrinsic_coords( manifold.extrinsic_to_intrinsic_coords(Theta) - np.pi/12)
            S1_histogram(Theta, 30, axs[0, idx], 'Reds')

        elif manifold_type == 'S2':
            manifold = Hypersphere(2)
            axs[0, idx].remove()
            ax0 = fig.add_subplot(2, len(num_modes_ls), 1 + idx, projection="mollweide")
            axs[0, idx] = ax0
            axs[0, idx].set_xticks([])
            axs[0, idx].set_yticks([])
            axs[0, idx].grid(True, alpha=0.3)
            Theta = multimodal_sampler(  5000, manifold_type, {'tau2': tau2, 'num_modes': num_modes})
            grid_resolution = 100
            grid, grid_theta, grid_phi = S2grid(grid_resolution)
            _, hat_f, grad_hat_f, = kernel_density_estimate('S2', Theta, 20, grid)
            im = axs[0, idx].pcolormesh( grid_phi - np.pi, np.pi/2 - grid_theta, hat_f.reshape(grid_resolution,grid_resolution), alpha=0.8, shading='auto', cmap='Reds')

        else:
            ValueError ( "Unsupported manifold type. Supported types are 'S1' and 'S2'." )
        sns.lineplot(
            data=df_long[df_long['num_modes'] == num_modes],
            x="num_samples",
            y="Loss",
            hue="Loss Type",
            hue_order=["Naïve","Empirical Denoised", "Oracle Denoised", "Oracle Bayes"],
            palette={
                "Naïve": "C0",
                "Empirical Denoised": "C2",
                "Oracle Denoised": "C2",
                "Oracle Bayes": "C4",
            },
            style="Loss Type",  # map linestyle to hue categories
            dashes={
                "Naïve": "",
                "Empirical Denoised": "",
                "Oracle Denoised": (1, 1),
                "Oracle Bayes": (1, 1),
            },
            estimator="mean",
            errorbar=("ci", 68),  # 1-sigma style band
            marker="o",
            ax=axs[1, idx],
        )
        # single shared legend (one row) placed below all subplots
        handles, labels = axs[1, idx].get_legend_handles_labels()
        axs[1, idx].get_legend().remove()
        if idx == len(num_modes_ls) - 1:
            fig.legend(handles,labels,loc="lower center",ncol=len(labels),frameon=False,bbox_to_anchor=(0.5, -.02))
        axs[1, idx].set_title(f"{num_modes} modes")
        axs[1, idx].set_xlabel("Sample Size")
        axs[1, idx].set_ylabel("Average Loss")
        axs[1, idx].tick_params(axis='x', rotation=45)
        
        # Share y-axes across rows and hide repeated y tick labels within each row
        base = axs[1, 0]
        for c in range(1, axs.shape[1]):
            axs[1, c].sharey(base)
            axs[1, c].set_ylabel("")
            axs[1, c].tick_params(labelleft=False)

    plt.tight_layout()

    if savefig is not None:
       plt.savefig('{}'.format(savefig), bbox_inches='tight')

    plt.show()
    return fig
