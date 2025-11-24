import math
import numpy as np
import pandas as pd

def _pca_reduce(Y, d):
    """
    Center Y and project to top-d principal components using SVD.
    Y: (N, J)
    returns Z: (N, d)
    """
    
    Yc = Y - Y.mean(axis=0, keepdims=True)
    
    # economy SVD
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    # project onto top d components
    Z = U[:, :d] * S[:d]
    return Z

def _simplex_volume(points):
    """
    points: (K, K-1) array; each row is a vertex in R^{K-1}.
    Returns volume of the (K-1)-simplex spanned by these K points.
    Volume = |det([points^T; 1 ... 1])| / (K-1)!
    """
    K, d = points.shape
    assert d == K - 1, "Each point must be in R^{K-1}."
    M = np.vstack([points.T, np.ones((1, K))])  # (K, K)
    return abs(np.linalg.det(M)) / math.factorial(K - 1)

def _init_indices_random(N, K, rng):
    return rng.choice(N, size=K, replace=False)

def _init_indices_atgp(Z, K):
    """
    ATGP-style initialization in the reduced space Z (N x (K-1)).
    Greedily picks points that are most orthogonal to current subspace.
    """
    N, d = Z.shape
    idx = []
    norms = np.sum(Z * Z, axis=1)
    idx.append(int(np.argmax(norms)))

    for _ in range(1, K):
        # Build orthonormal basis of current subspace
        Q, _ = np.linalg.qr(Z[idx, :].T)  # (d, m) with m=len(idx)
        # Projection onto subspace spanned by Q
        proj = Z @ (Q @ Q.T)              # (N, d)
        resid = Z - proj
        rnorm = np.sum(resid * resid, axis=1)
        rnorm[idx] = -np.inf              # exclude already selected
        idx.append(int(np.argmax(rnorm)))
    return np.array(idx, dtype=int)

def nfindr_BJ(Y, K, max_iter=5, seed=None, normalize=True, init='atgp'):
    """
    SciPy-free N-FINDR (NumPy only).

    Parameters
    ----------
    Y : array
        (N, J) or (rows, cols, bands). Must be nonnegative (common in spectral data).
    K : int
        Number of endmembers.
    max_iter : int
        Number of N-FINDR replacement sweeps.
    seed : int or None
        RNG seed for reproducibility (used if init='random').
    normalize : bool
        If True, L2-normalize each spectrum (row) before PCA (common practice).
    init : {'atgp', 'random'}
        Initialization method.

    Returns
    -------
    E : (K, J) array
        Extracted endmember spectra (rows).
    idx : (K,) array of ints
        Indices of selected pixels in the flattened data.
    """
    if isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()
    
    # Accept cube or matrix
    if Y.ndim == 3:
        rows, cols, J = Y.shape
        Y2 = Y.reshape(rows * cols, J)
    elif Y.ndim == 2:
        Y2 = Y
        rows = cols = None
    else:
        raise ValueError("Y must be (N, J) or (rows, cols, bands).")

    N, J = Y2.shape
    if normalize:
        norms = np.linalg.norm(Y2, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Yn = Y2 / norms
    else:
        Yn = Y2

    if K < 2 or K > J + 1:
        raise ValueError("K must satisfy 2 ≤ K ≤ J+1 (after PCA to K-1 dims).")

    # Reduce to (K-1)-dim space
    Z = _pca_reduce(Yn, d=K - 1)  # (N, K-1)

    # Initialize K indices
    rng = np.random.default_rng(seed)
    if init == 'atgp':
        idx = _init_indices_atgp(Z, K)
    elif init == 'random':
        idx = _init_indices_random(N, K, rng)
    else:
        raise ValueError("init must be 'atgp' or 'random'.")

    best_vol = _simplex_volume(Z[idx, :])

    # Replacement sweeps
    for _ in range(max_iter):
        improved = False
        for j in range(K):
            base = idx.copy()
            best_j = base[j]
            best_v = best_vol

            # Try replacing j-th vertex with every pixel
            for n in range(N):
                if n in base:
                    continue
                base[j] = n
                v = _simplex_volume(Z[base, :])
                if v > best_v:
                    best_v = v
                    best_j = n

            if best_j != idx[j]:
                idx[j] = best_j
                best_vol = best_v
                improved = True

        if not improved:
            break

    # Endmembers in original spectral space
    E = Y2[idx, :]
    return E, idx

# # --- Method 1: Nonnegative Constrained Least Squares
# def nnls(Y, H):
#     """
#     Solve min 0.5||y - wH||^2 s.t. w>=0, sum(w)=1 for each row y of Y
#     Y: (N, J) data, H: (K, J) endmembers (your E)
#     Returns W: (N, K) (nonneg)
#     """
#     N, J = Y.shape
#     K, J2 = H.shape
#     assert J == J2

#     # Unconstrained least squares: W_ls = Y H^T (H H^T)^{-1}
#     G = H @ H.T                       # (K,K)
#     # Regularize a bit in case G is near-singular
#     reg = 1e-10 * np.trace(G) / K
#     G_inv = np.linalg.pinv(G + reg * np.eye(K))
#     W_ls = (Y @ H.T) @ G_inv          # (N,K)

#     # Project each row onto the nonnegative space
#     W = np.vstack([np.maximum(w, 0.0) for w in W_ls])
#     return W

# # --- Method 2: Nonnegative Constrained Least Squares via Projected Gradient Descent (NNLS-PGD)
# def nnls_pgd(Y, H, max_iter=500, tol=1e-6):
#     """
#     Solve min 0.5||y - wH||^2 s.t. w>=0, sum(w)=1 for each row y of Y
#     using projected gradient with fixed step 1/L, L = ||H H^T||_2.
#     """
#     N, J = Y.shape
#     K, J2 = H.shape
#     assert J == J2

#     G = H @ H.T                              # (K,K)
#     # Lipschitz constant of gradient (spectral norm of G)
#     L = np.linalg.norm(G, 2)
#     step = 1.0 / (L + 1e-12)

#     B = Y @ H.T                              # (N,K) with rows b_i = y_i H^T
#     W = np.full((N, K), 1.0 / K)             # start at uniform simplex point

#     for i in range(N):
#         w = W[i]
#         b = B[i]
#         # objective: 0.5 w^T G w - b^T w + const; grad = G w - b
#         for _ in range(max_iter):
#             grad = G @ w - b
#             w_new = np.maximum(w - step * grad, 0.0)
#             if np.linalg.norm(w_new - w) < tol * (1 + np.linalg.norm(w)):
#                 w = w_new
#                 break
#             w = w_new
#         W[i] = w
#     return W
