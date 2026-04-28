import numpy as np
from numpy.linalg import eig, solve

def rayleigh_ab_from_targets(omegas, zetas):
    # omegas: [w1, w2] circular frequencies (rad/s)
    # zetas:  [z1, z2] target damping ratios
    A = np.array([[1/omegas[0], omegas[0]],
                  [1/omegas[1], omegas[1]]], dtype=float)
    b = np.array([2*zetas[0], 2*zetas[1]], dtype=float)
    alpha, beta = solve(A, b)
    return alpha, beta

def modal_frequencies(M, K, nmodes=None):
    # Solve generalized eigenproblem K φ = ω^2 M φ
    w2, Phi = eig(np.linalg.inv(M) @ K)
    # keep real positive part, sort
    w = np.sqrt(np.real(w2))
    idx = np.argsort(w)
    w = w[idx]
    Phi = Phi[:, idx]
    if nmodes: 
        w, Phi = w[:nmodes], Phi[:, :nmodes]
    return w, Phi

def rayleigh_damping_matrix(M, K, omegas=None, zetas=None, zeta_target=0.02, mode_ids=(0,1)):
    """
    If omegas/zetas are given (length 2), use them.
    Otherwise: compute modes, pick two mode IDs, assume same zeta_target.
    """
    if omegas is None:
        w, _ = modal_frequencies(M, K)
        i, j = mode_ids
        omegas = [w[i], w[j]]
        zetas  = [zeta_target, zeta_target]
    alpha, beta = rayleigh_ab_from_targets(omegas, zetas)
    C = alpha*M + beta*K
    return C, alpha, beta