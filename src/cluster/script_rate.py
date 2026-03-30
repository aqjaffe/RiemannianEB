try:
    from utils import *
    from params import *
except:
    import sys, os
    sys.path.append(os.getcwd().split('src')[0] + 'src')
    from utils import *
    from cluster.params import *

import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from contextlib import contextmanager
timenow = str(time.time())



manifold_type = 'S1'
manifold = get_manifold(manifold_type)
n_samples_ls, G_sampler_ls, M_grid, rho_grid, sigma2s, NMC, test_size, num_oracle_samples = getparams(manifold_type)


# ------- # ------- # ------- # ------- # ------- # ------- # ------- # ------- 

@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def run_one_mc(imc, manifold_type, G, n_samples_ls, M_grid, rho_grid, sigma2, test_size,
               oracle_samples, cv, light = True):
    """Single Monte Carlo replicate — fully self-contained, safe to run in a worker."""
    manifold = get_manifold(manifold_type)

    test_Theta = G.sample(test_size)
    test_X     = manifold.random_riemannian_normal(test_Theta, 1.0 / sigma2, test_size)

    naive_loss_val  = sq_loss(manifold, test_X, test_Theta)
    oracle_delta    =  oracle_denoiser(manifold_type, oracle_samples, sigma2, test_X)
    # oracle_delta    =  oracle_denoiser__kernel(manifold_type, oracle_samples, sigma2, test_X)
    oracle_loss_val = sq_loss(manifold, oracle_delta, test_Theta)

    emp_loss      = np.zeros((len(n_samples_ls), len(M_grid), len(rho_grid)), dtype=float)
    displacements = np.zeros_like(emp_loss)

    cv_loss          = np.zeros(len(n_samples_ls), dtype=float)  if cv else None
    cv_Ms_star       = np.zeros(len(n_samples_ls), dtype=int)    if cv else None
    cv_rhos_star     = np.zeros(len(n_samples_ls), dtype=float)  if cv else None
    cv_displacements = np.zeros(len(n_samples_ls), dtype=float)  if cv else None
    cv_excess_loss   = np.zeros(len(n_samples_ls), dtype=float)  if cv else None

    for ixn, n_samples in enumerate(n_samples_ls):
        train_Theta = G.sample(n_samples)
        train_X     = manifold.random_riemannian_normal(train_Theta, 1.0 / sigma2, n_samples)

        if not light:
            for ixM, M in enumerate(M_grid):
                density_onX = density_estimate(manifold_type, train_X, M, test_X)
                for ixrho, rho in enumerate(rho_grid):
                    delta = denoiser(manifold_type, train_X, M, rho, sigma2, test_X,
                                    densityIn=(density_onX[1], density_onX[2]))
                    emp_loss[ixn, ixM, ixrho]     = sq_loss(manifold, delta, test_Theta)
                    displacements[ixn, ixM, ixrho] = sq_loss(manifold, oracle_delta, delta)
                
        if cv:
            _M_grid = M_grid[M_grid <= n_samples//100 + 4] 
            M_star, rho_star = scoreMatchingKFoldCV(
                manifold_type, train_X, _M_grid, np.arange(3,10,1), n_splits=10, display_tqdm=False)['AIC']
            cv_Ms_star[ixn]       = M_star
            cv_rhos_star[ixn]     = rho_star

            if light:
                delta = denoiser(manifold_type, train_X, M_star, rho_star, sigma2, test_X)
                cv_loss[ixn]          = sq_loss(manifold, delta, test_Theta)
                cv_displacements[ixn] = sq_loss(manifold, oracle_delta, delta)
                cv_excess_loss[ixn]   = cv_loss[ixn] - oracle_loss_val

            else: 
                ixM_star   = int(np.argmin(np.abs(M_grid   - M_star)))
                ixrho_star = int(np.argmin(np.abs(rho_grid - rho_star)))
                cv_loss[ixn]          = emp_loss[ixn, ixM_star, ixrho_star]
                cv_displacements[ixn] = displacements[ixn, ixM_star, ixrho_star]

    return {
        'naive_loss':       naive_loss_val,
        'oracle_loss':      oracle_loss_val,
        'emp_loss':         emp_loss,
        'displacements':    displacements,
        'cv_loss':          cv_loss,
        'cv_Ms_star':       cv_Ms_star,
        'cv_rhos_star':     cv_rhos_star,
        'cv_displacements': cv_displacements,
        'cv_excess_loss':   cv_excess_loss,
    }


def converenge_rate_experiment(manifold_type, G, n_samples_ls, M_grid, rho_grid,
                                sigma2s, test_size, num_oracle_samples, NMC, timenow, cv=False):

    oracle_samples = G.sample(num_oracle_samples)
    sigma2s        = np.atleast_1d(sigma2s)
    
    all_records    = []
    all_oracleandcv_results = []

    for sigma2 in sigma2s:
        inner_per_mc = len(n_samples_ls) * len(M_grid) * len(rho_grid)
        desc = f'G="{G.name}", σ²={sigma2}'

        with tqdm_joblib(tqdm(
            total=NMC,
            desc=desc,
            dynamic_ncols=True,
            bar_format=(
                '{l_bar}{bar}| {n_fmt}/{total_fmt} replicates'
                ' [{elapsed}<{remaining}, {rate_fmt}]'
            )
        )):
            mc_results = Parallel(n_jobs=-1)(
                delayed(run_one_mc)(
                    imc, manifold_type, G, n_samples_ls, M_grid, rho_grid,
                    sigma2, test_size, oracle_samples, cv
                )
                for imc in range(NMC)
            )

        # ---- unpack results ----
        naive_loss    = np.array([r['naive_loss']  for r in mc_results])
        oracle_loss   = np.array([r['oracle_loss'] for r in mc_results])
        emp_loss      = np.stack([r['emp_loss']      for r in mc_results])  # (NMC, n, M, rho)
        displacements = np.stack([r['displacements'] for r in mc_results])

        condition =  cv and (sigma2 == sigma2s[3])
        if condition:
            cv_loss          = np.stack([r['cv_loss']          for r in mc_results])
            cv_Ms_star       = np.stack([r['cv_Ms_star']       for r in mc_results])
            cv_rhos_star     = np.stack([r['cv_rhos_star']     for r in mc_results])
            cv_displacements = np.stack([r['cv_displacements'] for r in mc_results])
            cv_excess_loss   = np.stack([r['cv_excess_loss']   for r in mc_results])

        # ---- aggregation ----
        for ixn, n_samples in enumerate(n_samples_ls):
            all_oracleandcv_results.append({
                'ID':                   float(timenow),
                'G':                    str(G.name),
                'sigma2':               float(sigma2),
                'num_samples':          int(n_samples),
                'mean_naive_loss':      float(np.mean(naive_loss)),
                'median_naive_loss':      float(np.median(naive_loss)),
                'std_naive_loss':       float(naive_loss.std()),
                'mean_oracle_loss':     float(np.mean(oracle_loss)),
                'median_oracle_loss':     float(np.median(oracle_loss)),
                'std_oracle_loss':      float(oracle_loss.std()),
                'median_cv_loss':         float(np.median(cv_loss[:, ixn]))             if condition else None,
                'mean_cv_loss':         float(np.mean(cv_loss[:, ixn]))                 if condition else None,
                'std_cv_loss':          float(cv_loss[:, ixn].std())                    if condition else None,
                'median_cv_excess_loss':  float(np.median(cv_excess_loss[:, ixn]))      if condition else None,
                'mean_cv_excess_loss':  float(np.mean(cv_excess_loss[:, ixn]))          if condition else None,
                'std_cv_excess_loss':   float(cv_excess_loss[:, ixn].std())             if condition else None,
                'median_cv_displacement': float(np.median(cv_displacements[:, ixn]))    if condition else None,
                'mean_cv_displacement': float(np.mean(cv_displacements[:, ixn]))        if condition else None,
                'std_cv_displacement':  float(cv_displacements[:, ixn].std())           if condition else None,
                'cv_Ms_star':           cv_Ms_star[:, ixn]                              if condition else None,
                'cv_rhos_star':         cv_rhos_star[:, ixn]                            if condition else None,
            })

            for ixM, M in enumerate(M_grid):
                for ixrho, rho in enumerate(rho_grid):
                    all_records.append({
                        'ID':                float(timenow),
                        'G':                 G.name,
                        'sigma2':            float(sigma2),
                        'num_samples':       int(n_samples),
                        'M':                 M,
                        'rho':               rho,
                        'NMC':               NMC,
                        'mean_emp_loss':     float(np.median(emp_loss[:, ixn, ixM, ixrho])),
                        'std_emp_loss':      float(emp_loss[:, ixn, ixM, ixrho].std()),
                        'mean_displacement': float(np.median(displacements[:, ixn, ixM, ixrho])),
                        'std_displacement':  float(displacements[:, ixn, ixM, ixrho].std()),
                        'mean_excess_loss':  float(np.median(emp_loss[:, ixn, ixM, ixrho] - oracle_loss)),
                        'std_excess_loss':   float(np.median(emp_loss[:, ixn, ixM, ixrho] - oracle_loss)),
                    })

    return pd.DataFrame(all_records), pd.DataFrame(all_oracleandcv_results)


results_mc  = []
results_ocv = []
for G in G_sampler_ls:
    dfmc, dforaclecv = converenge_rate_experiment(
        manifold_type, G, n_samples_ls, M_grid, rho_grid,
        sigma2s, test_size, num_oracle_samples, NMC, timenow, cv=True
    )
    results_mc.append(dfmc)
    results_ocv.append(dforaclecv)

params_ = {
    'ID':                 timenow,
    'manifold_type':      manifold_type,
    'n_samples_ls':       n_samples_ls,
    'M_grid':             M_grid,
    'rho_grid':           rho_grid,
    'sigma2s':            sigma2s,
    'test_size':          test_size,
    'num_oracle_samples': num_oracle_samples,
    'NMC':                NMC,
    'G_names':            [G.name   for G in G_sampler_ls],
    'G_params':           [G.params for G in G_sampler_ls],
}

out_dir = os.path.join('data', str(manifold_type), str(timenow))
os.makedirs(out_dir, exist_ok=True)

for name, results in zip(
    ['mc', 'ocv', 'params'],
    [
        pd.concat(results_mc,  ignore_index=True),
        pd.concat(results_ocv, ignore_index=True),
        params_,
    ]
):
    ext      = 'pkl' if name == 'params' else 'csv'
    filepath = os.path.join(out_dir, f'rate_{name}.{ext}')
    if name == 'params':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    else:
        results.to_csv(filepath, index=False)