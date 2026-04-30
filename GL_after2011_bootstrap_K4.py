# Import 
import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
from itertools import permutations
import time
import pandas as pd
from joblib import dump, load
from pathlib import Path
import sys

## sourceXray
from src.sourceXray_BJ import sourceXray, compute_C, solve_H_right_inverse, log_intrinsic_volume_score
from src.utils import *

## N-FINDR
from src.NFINDR import nfindr_BJ

## LS-NMF
from sklearn.decomposition import NMF
#-----------------------------------------------------------------------------------------------------------------------------#

# GL data 
df = pd.read_csv("data/GL_reduced_measurements_a2011_scaled.csv") 
df["SampleDate"] = pd.to_datetime(df["SampleDate"])
Y = df.drop(columns="SampleDate")
#-----------------------------------------------------------------------------------------------------------------------------#

# path 
outdir = Path("results/GL")
outdir.mkdir(parents=True, exist_ok=True)

# bootstrap settings
seed = 1
K = 4
max_K = 100*K
n, J = Y.shape

# reference H 
H_star = load(outdir/f"GL_a2011_scaled_K{K}_maxK{max_K}_results.joblib")[0]

# allocate save arrays
results_boots = {
    "seed": np.nan,

    "time": np.nan,
    "logvol": np.nan, 
    "Phi_hat": np.empty((J, K), dtype=float),
    # to recover W later
    "H_star_hat": np.empty((K, J), dtype=float),
    # estimation error
    "mse_by_pollutant": np.full(J, np.nan),
    "mse_overall": np.nan,

    # N-FINDR results
    "time_nfindr": np.nan,
    "logvol_nfindr": np.nan,
    "Phi_hat_nfindr": np.empty((J, K), dtype=float),
    "H_star_hat_nfindr": np.empty((K, J), dtype=float),
    "mse_by_pollutant_nfindr": np.full(J, np.nan),
    "mse_overall_nfindr": np.nan,

    # LS-NMF results
    "time_lsnmf": np.nan,       
    "logvol_lsnmf": np.nan,
    "Phi_hat_lsnmf": np.empty((J, K), dtype=float),
    "H_star_hat_lsnmf": np.empty((K, J), dtype=float),
    "mse_by_pollutant_lsnmf": np.full(J, np.nan),
    "mse_overall_lsnmf": np.nan,
}
#-----------------------------------------------------------------------------------------------------------------------------#

# Bootstrap 
rep_env = os.environ.get("SLURM_ARRAY_TASK_ID")
rep = int(rep_env) if rep_env else 1
file_rep = outdir/f"K{K}/GL_a2011_scaled_K{K}_maxK{max_K}_bootstrap_rep{rep}.joblib"
(outdir/f"K{K}").mkdir(parents=True, exist_ok=True)
if file_rep.exists():
    sys.exit(0)  # skip if already done
seed_rep = seed+rep
rng = np.random.default_rng(seed_rep)
results_boots["seed"] = seed_rep

# resample
idx = rng.integers(0, n, size=n)
Yb = np.asarray(Y)[idx] 
rb = Yb.sum(axis=1, keepdims=True)
Yb_star = Yb / rb

# sourceXray
start = time.time()
H_star_hat, W_tilde_hat, mu_tilde_hat, Phi_hat, logvol_hat = sourceXray(Yb, K, seed=seed_rep, tol=1e-12,
                                                                        candidate_method="random", # "random" (random directions)
                                                                        T=20000, topk=1, max_K=max_K, # for random candidate method
                                                                        verbose=True)[0]
end = time.time()
results_boots["time"] = end - start
results_boots["logvol"] = logvol_hat

## estimation error
Yhat = W_tilde_hat @ H_star_hat
e = Yb - Yhat
results_boots["mse_by_pollutant"] = np.mean(e**2, axis=0)
results_boots["mse_overall"] = np.mean(e**2)

## permute
H_star_hat_perm, mu_tilde_hat_perm, Phi_hat_perm, order = permute_estimates_to_match_truth(H_star, H_star_hat, mu_tilde_hat, Phi_hat)
results_boots["Phi_hat"] = np.asarray(Phi_hat_perm)
results_boots["H_star_hat"] = np.asarray(H_star_hat_perm)

# N-FINDR
start = time.time()
H_star_hat_nfindr, *_ = nfindr_BJ(Yb_star, K, max_iter=5, seed=seed_rep, normalize=False, init='atgp')
W_star_hat_nfindr, _, _ = solve_H_right_inverse(Yb_star, H_star_hat_nfindr)
W_tilde_hat_nfindr = W_star_hat_nfindr * rb
mu_tilde_hat_nfindr = W_tilde_hat_nfindr.mean(axis=0)
Phi_hat_nfindr = compute_C(mu_tilde_hat_nfindr, H_star_hat_nfindr)
end = time.time()
results_boots["time_nfindr"] = end - start
logvol_nfindr, _ = log_intrinsic_volume_score(H_star_hat_nfindr)
results_boots["logvol_nfindr"] = logvol_nfindr

## estimation error
Yhat_nfindr = W_tilde_hat_nfindr @ H_star_hat_nfindr
e_nfindr = Yb - Yhat_nfindr
results_boots["mse_by_pollutant_nfindr"] = np.mean(e_nfindr**2, axis=0)
results_boots["mse_overall_nfindr"] = np.mean(e_nfindr**2)

## permute
H_star_hat_perm_nfindr, mu_tilde_hat_perm_nfindr, Phi_hat_perm_nfindr, order_nfindr = permute_estimates_to_match_truth(H_star, H_star_hat_nfindr, mu_tilde_hat_nfindr, Phi_hat_nfindr)
results_boots["Phi_hat_nfindr"] = np.asarray(Phi_hat_perm_nfindr)
results_boots["H_star_hat_nfindr"] = np.asarray(H_star_hat_perm_nfindr)

# LS-NMF
start = time.time()
nmf_model = NMF(
    n_components=K,
    init="nndsvda",          # good default for nonnegative data
    random_state=seed,   # tie to bootstrap seed for reproducibility
    max_iter=1000,
    tol=1e-4
)

W_nmf = nmf_model.fit_transform(Yb_star)  # shape: (n, K)
H_nmf = nmf_model.components_            # shape: (K, J)

## make H_nmf row-stochastic and adjust W_nmf accordingly 
H_rs = H_nmf.sum(axis=1, keepdims=True)  # shape (K, 1)
H_star_hat_lsnmf = H_nmf/H_rs            # each row sums to 1
W_star_hat_lsnmf = W_nmf*H_rs.T          # scale columns of W_nmf

## revert back to original scale
W_tilde_hat_lsnmf = W_star_hat_lsnmf * rb     
mu_tilde_hat_lsnmf = W_tilde_hat_lsnmf.mean(axis=0)
Phi_hat_lsnmf = compute_C(mu_tilde_hat_lsnmf, H_star_hat_lsnmf)
end = time.time()
results_boots["time_lsnmf"] = end - start
logvol_lsnmf, _ = log_intrinsic_volume_score(H_star_hat_lsnmf)
results_boots["logvol_lsnmf"] = logvol_lsnmf

## estimation error
Yhat_lsnmf = W_tilde_hat_lsnmf @ H_star_hat_lsnmf
e_lsnmf = Yb - Yhat_lsnmf
results_boots["mse_by_pollutant_lsnmf"] = np.mean(e_lsnmf**2, axis=0)
results_boots["mse_overall_lsnmf"] = np.mean(e_lsnmf**2)

## permute
H_star_hat_perm_lsnmf, mu_tilde_hat_perm_lsnmf, Phi_hat_perm_lsnmf, order_lsnmf = permute_estimates_to_match_truth(H_star, H_star_hat_lsnmf, mu_tilde_hat_lsnmf, Phi_hat_lsnmf)
results_boots["Phi_hat_lsnmf"] = np.asarray(Phi_hat_perm_lsnmf)
results_boots["H_star_hat_lsnmf"] = np.asarray(H_star_hat_perm_lsnmf)

dump(results_boots, file_rep)   

