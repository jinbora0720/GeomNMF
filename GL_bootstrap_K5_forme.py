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
df = pd.read_csv("data/GL_measurements_scaled.csv") 
df["SampleDate"] = pd.to_datetime(df["SampleDate"])
Y = df.drop(columns="SampleDate")
#-----------------------------------------------------------------------------------------------------------------------------#

# path 
outdir = Path("results/GL")
outdir.mkdir(parents=True, exist_ok=True)

# bootstrap settings
seed = 1
K = 5
min_K = 20*K
n, J = Y.shape

# reference H 
H_star = load(outdir/f"GL_scaled_K{K}_minK{min_K}_results.joblib")[0]

#-----------------------------------------------------------------------------------------------------------------------------#

# Bootstrap 
rep_env = os.environ.get("SLURM_ARRAY_TASK_ID")
rep = int(rep_env) if rep_env else 1
file_rep = outdir/f"v3_K{K}/GL_scaled_K{K}_minK{min_K}_bootstrap_rep{rep}.joblib"
(outdir/f"v3_K{K}").mkdir(parents=True, exist_ok=True)
results_boots = load(file_rep)
seed_rep = seed+rep
rng = np.random.default_rng(seed_rep)

# resample
idx = rng.integers(0, n, size=n)
Yb = np.asarray(Y)[idx] 
rb = Yb.sum(axis=1, keepdims=True)
Yb_star = Yb / rb

# LS-NMF
start = time.time()
nmf_model = NMF(
    n_components=K,
    init="nndsvda",          # good default for nonnegative data
    random_state=seed_rep,   # tie to bootstrap seed for reproducibility
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

# LS-NMF w/ regularization
start = time.time()
nmf_model2 = NMF(
    n_components=K,
    init="nndsvda",          # good default for nonnegative data
    random_state=seed_rep,   # tie to bootstrap seed for reproducibility
    max_iter=1000,
    tol=1e-4, 
    alpha_W = 0.01, 
    alpha_H = 0.01
)

W_nmf2 = nmf_model2.fit_transform(Yb_star)  # shape: (n, K)
H_nmf2 = nmf_model2.components_            # shape: (K, J)

## make H_nmf row-stochastic and adjust W_nmf accordingly 
H_rs = H_nmf2.sum(axis=1, keepdims=True)  # shape (K, 1)
H_star_hat_lsnmf2 = H_nmf2/H_rs            # each row sums to 1
W_star_hat_lsnmf2 = W_nmf2*H_rs.T          # scale columns of W_nmf2

## revert back to original scale
W_tilde_hat_lsnmf2 = W_star_hat_lsnmf2 * rb     
mu_tilde_hat_lsnmf2 = W_tilde_hat_lsnmf2.mean(axis=0)
Phi_hat_lsnmf2 = compute_C(mu_tilde_hat_lsnmf2, H_star_hat_lsnmf2)
end = time.time()
results_boots["time_lsnmf2"] = end - start
logvol_lsnmf2, _ = log_intrinsic_volume_score(H_star_hat_lsnmf2)
results_boots["logvol_lsnmf2"] = logvol_lsnmf2

## estimation error
Yhat_lsnmf2 = W_tilde_hat_lsnmf2 @ H_star_hat_lsnmf2
e_lsnmf2 = Yb - Yhat_lsnmf2
results_boots["mse_by_pollutant_lsnmf2"] = np.mean(e_lsnmf2**2, axis=0)
results_boots["mse_overall_lsnmf2"] = np.mean(e_lsnmf2**2)

## permute
H_star_hat_perm_lsnmf2, mu_tilde_hat_perm_lsnmf2, Phi_hat_perm_lsnmf2, order_lsnmf2 = permute_estimates_to_match_truth(H_star, H_star_hat_lsnmf2, mu_tilde_hat_lsnmf2, Phi_hat_lsnmf2)
results_boots["Phi_hat_lsnmf2"] = np.asarray(Phi_hat_perm_lsnmf2)
results_boots["H_star_hat_lsnmf2"] = np.asarray(H_star_hat_perm_lsnmf2)

dump(results_boots, file_rep)   

