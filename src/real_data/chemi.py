import sys, os
sys.path.append(os.getcwd().split('src')[0] + 'src')
from utils import *
manifold_type = 'T2'
import Bio.PDB
warnings.simplefilter('ignore', Bio.BiopythonWarning)

fig, aaxs = plt.subplots(
    3,
    4,
    figsize=(12.5, 10),
    gridspec_kw={"width_ratios": [0.5, 1, 1, 1]},
)
for idx, file in enumerate(["6x8j.pdb1", "1ADO.pdb1", "1ENO.pdb1"]):
    print(f"\nProcessing {file}...")
    # Parse the PDB file and extract the phi and psi angles
    parser = Bio.PDB.PDBParser()
    file = 'data/'+ file
    structure = parser.get_structure('protein', file)
    polypeptide = Bio.PDB.PPBuilder().build_peptides(structure[0])

    phi = []
    psi = []
    for strand in polypeptide:
        phipsi = strand.get_phi_psi_list()
        for point in phipsi:
            try:
                phi_point = point[0] * 1.
                psi_point = point[1] * 1.
                phi.append(phi_point)
                psi.append(psi_point)
            except TypeError:
                pass
        
    X_phi, X_psi = np.asarray(phi), np.asarray(psi)
    X = np.asarray([[np.cos(phi),np.sin(phi)],
                    [np.cos(psi),np.sin(psi)]]).T
    

    
    # Calculate the mean B-factor and the corresponding sigma2 for the T2 manifold
    mean_b_factor = 0.
    count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    count += 1
                    mean_b_factor += atom.get_bfactor()
    mean_b_factor /= count
    sigma2 = np.arctan(np.sqrt(mean_b_factor/(8*np.pi**2))/1.5)**2
    print(f'    sigma2 = {sigma2:.4f}')

    # Perform cross-validation to select the optimal parameters for density estimation
    M_grid=np.arange(1, 7)
    rhoperc_grid = 5#np.arange(1,10,0.5)
    criterion = 'cv'

    # params, scores = scoreMatchingKFoldCV(manifold_type, X, M_grid, rhoperc_grid, n_splits=10, return_scores = True, random_state=42)
    # params, scores = DensityKFoldCV(manifold_type, X, M_grid, rhoperc_grid, n_splits=10, return_scores = True, random_state=42)
    # if True: plot_cv_scores(scores[criterion], M_grid, rhoperc_grid, title=f"Density cv {params[criterion]}", ax = aaxs[idx, 0])
    # M, rho = params[criterion]
    M,rho = 3, 1e-2
    if True:
        aaxs[idx, 0].cla()
        aaxs[idx, 0].axis("off")
        protein_name = os.path.splitext(os.path.basename(file))[0].split(".")[0]
        aaxs[idx, 0].text(
            0.5,
            0.5,
            protein_name,
            transform=aaxs[idx, 0].transAxes,
            ha="center",
            va="center",
            fontsize=18,
        )
    print(f'    optimal M = {M}, optimal rho = {rho:.4f}')

    # Estimate the density using the optimal parameters
    res = 50   
    grid, grid_phi, grid_psi = T2grid(res)
    grid, hat_f, hat_grad_f = density_estimate(manifold_type, X, M, grid, sigma2)
    
    # denoiser
    delta = denoiser(manifold_type, X, M, rho, sigma2, X)


    # --------- --------- ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  --------- 
    # --------- plotting ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  --------- 
    axs = aaxs[idx, 1:]

    # Top-left: X
    axs[0].grid(True, color="gray", lw=0.5)
    T2_scatter(X, ax=axs[0])
    axs[0].set_title(r"$X_i$")

    # right: denoised delta
    axs[2].grid(True, color="gray", lw=0.5)
    T2_scatter(delta, ax=axs[2], color = 'C2' )
    axs[2].set_title(r"$\hat{\delta}_{\mathcal{T}}(X_i)$")

    # center: score field 
    _T2_grid = T2grid(30)[0]
    _T2_grid, _hat_f, _hat_grad_f = density_estimate(manifold_type, X, int(M), _T2_grid, sigma2)
    _score = _hat_grad_f / np.maximum(_hat_f[:, None, None], rho)
    vals_to_plot = {
        "gradient": {"vals": _hat_grad_f,"cmap": "Greys","title": r"$\nabla \hat f$",},
        "score": {"vals": _score,"cmap": "Greens","title": r"$\nabla \log \hat f$",
        },
    }
    WHICH = "score"  # "gradient" or "score"
    _lo, _hi = np.percentile(_hat_f, [5, 95])
    _hat_f = np.clip(_hat_f, _lo, _hi)
    T2plot_quiver(
        _T2_grid,
        vals_to_plot[WHICH]["vals"],
        figax=(fig, axs[1]),
        scale=2*M,
        cmap=vals_to_plot[WHICH]["cmap"],
        cvals = _hat_f,
    )
    axs[1].set_title(vals_to_plot[WHICH]["title"])

    # axis formatting 
    for ax in axs.flatten():
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        ax.set_aspect("equal", adjustable="box")
os.makedirs("fig", exist_ok=True)
plt.savefig("fig/chemi.png", bbox_inches="tight")