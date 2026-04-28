# fea_utl/fsi_interp.py
import numpy as np

def structural_reduced_mats(M_global, C_global, K_global, Psi):
    """Riduci le matrici strutturali (secche) nei primi r modi."""
    M_r = Psi.T @ M_global @ Psi
    C_r = Psi.T @ C_global @ Psi if C_global is not None else np.zeros_like(M_r)
    K_r = Psi.T @ K_global @ Psi
    return M_r, C_r, K_r

def _symmetrize(A):
    return 0.5*(A + A.T)

def make_matrix_interp_in_k(k_samples, mats_samples):
    """
    Interpolatore lineare in k per una matrice r×r.
    - k_samples: (n_k,)
    - mats_samples: (n_k, r, r)
    Ritorna una funzione f(kq) -> (r,r) (o stack per array di kq).
    """
    k = np.array(k_samples, dtype=float)
    M = np.array(mats_samples, dtype=float)
    assert M.ndim == 3 and len(k)==M.shape[0], "shape mismatch"

    def interp_one(kq):
        if kq <= k[0]:  return M[0]
        if kq >= k[-1]: return M[-1]
        i = np.searchsorted(k, kq) - 1
        t = (kq - k[i]) / (k[i+1] - k[i] + 1e-16)
        A = (1.0 - t)*M[i] + t*M[i+1]
        return _symmetrize(A)

    def f(k_query):
        if np.isscalar(k_query):
            return interp_one(float(k_query))
        k_query = np.array(k_query, dtype=float).ravel()
        outs = [interp_one(x) for x in k_query]
        return np.stack(outs, axis=0)

    return f

def build_fluid_interps_for_group(results_group, Psi):
    """
    Dato un gruppo di risultati (stessa alpha e V0, k diversi),
    proietta M_added, C_added in spazio modale e crea due interpolatori in k.
    """
    k_list, MW_list, CW_list = [], [], []
    for res in results_group:
        k_list.append(float(res["k"]))
        MW_list.append(_symmetrize(Psi.T @ res["M_added"] @ Psi))
        CW_list.append(_symmetrize(Psi.T @ res["C_added"] @ Psi))

    k_arr = np.array(k_list, dtype=float)
    idx = np.argsort(k_arr)
    k_arr = k_arr[idx]
    MW_stack = np.stack([MW_list[i] for i in idx], axis=0)
    CW_stack = np.stack([CW_list[i] for i in idx], axis=0)

    # deduplica k identici
    keep = np.ones_like(k_arr, dtype=bool)
    keep[1:] = np.diff(k_arr) > 1e-12
    k_arr = k_arr[keep]
    MW_stack = MW_stack[keep]
    CW_stack = CW_stack[keep]

    MW_interp = make_matrix_interp_in_k(k_arr, MW_stack)
    CW_interp = make_matrix_interp_in_k(k_arr, CW_stack)
    return MW_interp, CW_interp, k_arr
