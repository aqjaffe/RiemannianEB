import sys, os
sys.path.append(os.getcwd().split('src')[0] + 'src')
from utils import *
manifold_type = 'S2'
manifold = get_manifold(manifold_type)

fig, axs = plt.subplots(1, 3, figsize=(20, 8), subplot_kw={'projection': 'mollweide'})

# read data
df = pd.read_csv('real_data/data/BATSE_4B.txt', header=None, sep='\s+', encoding='utf-8')

# get spherical coordinates of observations X
ra = df[5].values
dec = df[6].values
phi = np.deg2rad(ra)           # longitude
theta = np.pi/2 - np.deg2rad(dec)  # colatitude
X_sph = np.column_stack([theta, phi])
X = manifold.spherical_to_extrinsic(X_sph)

# sigma2 = np.mean(np.power(df[9].values * np.pi/180.,2))
sigma2 = 3.046*1e-2
print('Average angular variance (in rad^2): ', sigma2)

# crossvalidation for hyperparameter selection
M_grid=np.arange(1, 25)
rhoperc_grid = 1.5 # np.arange(1,10,0.5)
criterion = 'AIC'
params, scores = scoreMatchingKFoldCV(manifold_type, X, M_grid, rhoperc_grid, n_splits=5, return_scores = True, random_state=42)
if False:
    plot_cv_scores(scores[criterion], M_grid, rhoperc_grid, title=f"Score Matching cv {params[criterion]}", ax = axs[0])
    axs = axs[1:]

M, rho = params[criterion]

# compute denoised estimates
delta = denoiser('S2', X, M, rho, sigma2, X)

# --------- --------- ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  --------- 
# --------- plotting ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  --------- 

# left plot: X
axs[0].grid(True, color='gray', lw=0.5)
axs[0].set_title('$X_i$', y =1.1)
S2scatter(X, ax=axs[0], color='C0', alpha=0.25, s=10)

# right plot: $\hat{\delta}_T(X_i)$
axs[2].set_title(r'$\hat{\delta}_{\mathcal{T}}(X_i)$', y=1.1)

S2scatter(delta, ax=axs[2], color='C2', alpha=0.25, s=10)


# center plot: $\nabla \log \hat{f}$
_X_grid, _hat_f, _hat_grad_f = density_estimate('S2', X, M, S2grid_fib(50)[0])
_score = _hat_grad_f/np.maximum(_hat_f, rho)[:, None]
vals_to_plot =  {   'gradient': {'vals': _hat_grad_f, 'cmap' : 'Greys', 'title' : r'$\nabla \hat f$' },
                    'score': {'vals': _score, 'cmap' : 'Greens', 'title' : r'$\nabla \log \hat f$' }}
WHICH = 'score' # 'gradient' or 'score'

S2plot_quiver(_X_grid , vals_to_plot[WHICH]['vals'], skip = 1,
              figax= (fig,axs[1]), scale=50, cmap = vals_to_plot[WHICH]['cmap'], cvals = _hat_f)
axs[1].set_title(vals_to_plot[WHICH]['title'], y =1.1)

for ax in axs.flatten():   
    ax.set_xticks(np.linspace(-np.pi, np.pi, 7)[1:-1])
    ax.set_xticklabels([ '120°', '60°', '0°', '60°', '120°'])
    ax.set_yticks(np.linspace(-np.pi/2, np.pi/2, 5))
    ax.set_yticklabels(['90°', '45°', '0°', '45°', '90°'])

plt.tight_layout()
plt.savefig('fig/astro.png', bbox_inches='tight')
plt.show()




