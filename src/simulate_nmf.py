from scipy.spatial import ConvexHull
import numpy as np
from src.sourceXray_BJ import compute_C

# source functions to simulate data
def generate_convex_independent_H(K, J, batch=200, max_iter=100, seed=None):
    rng = np.random.default_rng(None if seed is None else seed)
    for _ in range(max_iter):
        points = rng.exponential(scale=1.0, size=(batch, J))
        try:
            hull = ConvexHull(points)
        except Exception:
            continue
        hull_verts = hull.vertices
        if len(hull_verts) >= K:
            chosen = rng.choice(hull_verts, size=K, replace=False)
            return points[chosen]
    raise RuntimeError(f"Failed to find {K} convex-independent points in {max_iter} tries.")

# ---------- mixture of log normals ----------
def generate_W_iid(n, K, seed=None):
    """
    Mixture of lognormal W, independent across (i,k).
    Returns W (n x K) and population mean mu (length K).
    """
    rng = np.random.default_rng(None if seed is None else seed)

    # per-k mixture parameters
    C_list   = [rng.poisson(lam=3) + 1 for _ in range(K)]
    pis_list = [rng.dirichlet(np.ones(C)) for C in C_list]
    means    = [rng.uniform(-1.0, 1.0, size=C) for C in C_list]
    sds      = [rng.uniform(0.1, 1.0, size=C) for C in C_list]

    W = np.empty((n, K), dtype=float)
    for k in range(K):
        Ck, pik, mk, sk = C_list[k], pis_list[k], means[k], sds[k]
        z = rng.choice(Ck, size=n, p=pik)
        gk = rng.normal(loc=mk[z], scale=sk[z])
        W[:, k] = np.exp(gk)

    # population mean for each k under the mixture
    mu = np.array([
        np.sum(pis_list[k] * np.exp(means[k] + 0.5 * sds[k]**2))
        for k in range(K)
    ])
    return W, mu

# ---------- AR(1) on log-W ----------
def generate_W_ar1(n, K, phi=0.8, mu_log=None, sigma_eps=None, seed=None):
    """
    Generate W with log W following independent AR(1) per coordinate k.

      g_{i,k} = mu_log[k] + phi_k*(g_{i-1,k} - mu_log[k]) + eps_{i,k}
      eps_{i,k} ~ N(0, sigma_eps[k]^2),  |phi_k| < 1

    Returns:
      W  : (n x K) positive matrix
      mu : (K,) population mean of W under the stationary distribution
    """
    rng = np.random.default_rng(None if seed is None else seed)

    # allow scalar or vector params
    if np.isscalar(phi):
        phi = np.full(K, float(phi))
    else:
        phi = np.asarray(phi, float)
        assert phi.shape == (K,)
    if mu_log is None:
        mu_log = rng.uniform(-0.5, 0.5, size=K)          # mean of g
    else:
        mu_log = np.asarray(mu_log, float); assert mu_log.shape == (K,)
    if sigma_eps is None:
        sigma_eps = rng.uniform(0.15, 0.5, size=K)       # innovation sd of g
    else:
        sigma_eps = np.asarray(sigma_eps, float); assert sigma_eps.shape == (K,)

    if np.any(np.abs(phi) >= 1):
        raise ValueError("All |phi_k| must be < 1 for stationarity.")

    # stationary variance of g: var_g = sigma_eps^2 / (1 - phi^2)
    var_g = sigma_eps**2 / (1.0 - phi**2)
    sd_g0 = np.sqrt(var_g)

    g = np.empty((n, K), dtype=float)

    # initialize from the stationary distribution
    g[0, :] = rng.normal(loc=mu_log, scale=sd_g0)

    # iterate AR(1)
    eps = rng.normal(loc=0.0, scale=sigma_eps, size=(n-1, K))
    for t in range(1, n):
        g[t, :] = mu_log + phi*(g[t-1, :] - mu_log) + eps[t-1, :]

    W = np.exp(g)

    # population mean of W under stationary g ~ N(mu_log, var_g)
    mu = np.exp(mu_log + 0.5*var_g)
    return W, mu

# ---------- top-level simulator ----------
def simulate_dataset(n, K, J, *, process="iid", seed=None, **kwargs):
    """
    Simulate Y = W H with row-normalized versions and helper quantities.

    Args
    ----
    n, K, J : int
        n samples, K sources, J features.
    process : {"ar1", "iid"}
        How to generate W.
        - "ar1": log W follows AR(1) per k (use kwargs: phi, mu_log, sigma_eps).
        - "iid"  : your original iid mixture generator.
    seed : int | None
        Seed for reproducibility.
    **kwargs : passed to the chosen W generator.

    Returns
    -------
    (Y, Y_star, r,
     H, H_star,
     W, mu, W_tilde, mu_tilde, W_star,
     C, C_alt)
    """

    # H: K x J (convex-independent rows)
    H = generate_convex_independent_H(K, J, seed=seed)

    # W: n x K and population mean mu: length K
    if process == "ar1":
        W, mu = generate_W_ar1(n, K, seed=seed, **kwargs)
    elif process == "iid":
        W, mu = generate_W_iid(n, K, seed=seed)
    else:
        raise ValueError(f"Unknown process '{process}'")

    # Observed matrix
    Y = W @ H                             # (n x J)

    # Normalize H rows to sum 1
    d = H.sum(axis=1, keepdims=True)     # (K x 1)
    H_star = H / d

    # Scale-aware W, mu
    d_flat = d.flatten()               # length K
    W_tilde = W * d_flat               # (n x K)
    mu_tilde = mu * d_flat             # (K,)

    # Row-normalize Y to simplex and its weights
    r = Y.sum(axis=1, keepdims=True)      # (n x 1)
    Y_star = Y / r
    W_star = W_tilde / r

    # Composition matrices from population means
    C     = compute_C(mu, H)
    C_alt = compute_C(mu_tilde, H_star)

    return Y, Y_star, r, H, H_star, W, mu, W_tilde, mu_tilde, W_star, C, C_alt

# ---------- smoke test ----------
if __name__ == "__main__":
    # AR(1) log-normal W
    out = simulate_dataset(n=200, K=4, J=6, process="ar1", seed=0, phi=0.9)
    Y, Y_star, r, H, H_star, W, mu, W_tilde, mu_tilde, W_star, C, C_alt = out
    print("AR(1) shapes:", Y.shape, H.shape, W.shape)

    # iid mixture W (your original behavior)
    out2 = simulate_dataset(n=200, K=4, J=6, process="iid", seed=1)
    print("IID mixture shapes:", out2[0].shape, out2[3].shape, out2[5].shape)