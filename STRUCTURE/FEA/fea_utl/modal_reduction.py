from scipy.linalg import eig, solve
import numpy as np

def reduce_matrices(M, C=None, K=None, Psi=None):
    
    M_hat = Psi.T @ M @ Psi
    
    K_hat = Psi.T @ K @ Psi
    
    if C is not None:
        C_hat = Psi.T @ C @ Psi
    else:
        C_hat = None
    
    return M_hat, K_hat, C_hat

def solve_modes_state_space(M_hat, C_hat, K_hat, num_modes=6):

    n = M_hat.shape[0]
    I = np.eye(n)
    Z = np.zeros((n, n))

    # Handle undamped case
    if C_hat is None:
        C_hat = Z

    # State-space generalized eigenproblem
    A = np.block([[Z, I],
                  [-K_hat, -C_hat]])
    B = np.block([[I, Z],
                  [Z, M_hat]])

    eigvals, eigvecs = eig(A, B)

    # Keep oscillatory roots (positive imaginary part)
    keep = np.imag(eigvals) > 1e-12
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]

    # Sort by increasing frequency (|Im(λ)|)
    order = np.argsort(np.imag(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Optionally truncate to first 2*num_modes structural eigenvalues
    m_keep = min(2 * num_modes, eigvals.size)
    eigvals = eigvals[:m_keep]
    eigvecs = eigvecs[:, :m_keep]

    # Frequencies & damping from eigenvalues
    omega = np.imag(eigvals)                      # rad/s
    freqs = omega / (2.0 * np.pi)                 # Hz
    zeta  = -np.real(eigvals) / omega

    # Physical mode shapes = q block (first n rows), mass-normalized
    modes = eigvecs[:n, :].copy()
    for j in range(modes.shape[1]):
        mj = np.real(modes[:, j].conj().T @ (M_hat @ modes[:, j]))
        if mj > 1e-14:
            modes[:, j] /= np.sqrt(mj)

    return freqs, zeta, eigvals, modes, eigvecs
    