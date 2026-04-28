import numpy as np
from scipy.linalg import eigh, solve, cholesky, LinAlgError
from scipy.sparse.linalg import eigsh
import warnings

def diagnose_matrices(M, K, expected_frequencies=None):
    """
    Diagnose structural matrices and compare with expected frequencies
    """
    print("\n" + "="*50)
    print("MATRIX DIAGNOSTICS")
    print("="*50)
    
    # Check matrix properties
    print(f"Matrix size: {M.shape[0]} x {M.shape[1]}")
    print(f"Mass matrix condition number: {np.linalg.cond(M):.2e}")
    print(f"Stiffness matrix condition number: {np.linalg.cond(K):.2e}")
    
    # Check for obvious issues
    M_eigs = np.real(np.linalg.eigvals(M))
    K_eigs = np.real(np.linalg.eigvals(K))
    
    print(f"Mass matrix eigenvalues range: [{M_eigs.min():.2e}, {M_eigs.max():.2e}]")
    print(f"Stiffness matrix eigenvalues range: [{K_eigs.min():.2e}, {K_eigs.max():.2e}]")
    
    # Quick eigenvalue check
    try:
        test_eigs = eigh(K, M, eigvals_only=True)
        test_freqs = np.sqrt(np.abs(test_eigs)) / (2 * np.pi)
        test_freqs = test_freqs[test_freqs > 0.1]  # Filter very low frequencies
        
        print(f"Quick frequency check - first 10 modes above 0.1 Hz:")
        print([f'{f:.3f}' for f in test_freqs[:10]])
        
        if expected_frequencies is not None:
            print(f"Expected frequencies: {expected_frequencies}")
            
    except Exception as e:
        print(f"Quick eigenvalue test failed: {e}")
    
    print("="*50)

def compute_dry_modes_full(M, K, num_modes=10, freq_threshold=1.0):
    """
    Compute dry modes using full matrices (reference solution)
    """
    print("\n" + "="*60)
    print("STEP 1: FULL DRY MODAL ANALYSIS (Reference)")
    print("="*60)
    
    try:
        # Solve full generalized eigenvalue problem
        dry_eigenvals, dry_eigenvecs = eigh(K, M)
        
        # Convert to frequencies
        all_frequencies = np.sqrt(np.abs(dry_eigenvals)) / (2 * np.pi)
        
        # Filter out very low frequency modes
        valid_mask = all_frequencies > freq_threshold
        
        if not np.any(valid_mask):
            print(f"Warning: No modes above {freq_threshold} Hz. Lowering threshold to 0.1 Hz")
            freq_threshold = 0.1
            valid_mask = all_frequencies > freq_threshold
            
        dry_frequencies = all_frequencies[valid_mask]
        dry_modes = dry_eigenvecs[:, valid_mask]
        
        # Sort by frequency
        idx = np.argsort(dry_frequencies)
        dry_frequencies = dry_frequencies[idx]
        dry_modes = dry_modes[:, idx]
        
        print(f"Found {len(dry_frequencies)} modes above {freq_threshold} Hz")
        print(f"First 10 dry frequencies: {[f'{f:.3f}' for f in dry_frequencies[:10]]} Hz")
        
        return dry_frequencies, dry_modes
        
    except Exception as e:
        print(f"Error in full dry modal analysis: {e}")
        return None, None

def test_modal_reduction_dry(M, K, num_modes=5, freq_threshold=1.0):
    """
    Test modal reduction technique using only dry matrices (no fluid)
    """
    print("\n" + "="*60)
    print("STEP 2: MODAL REDUCTION TEST (Dry matrices only)")
    print("="*60)
    
    # First get reference solution
    dry_freq_full, dry_modes_full = compute_dry_modes_full(M, K, num_modes*3, freq_threshold)
    
    if dry_freq_full is None:
        return None, None, None
    
    # Now test modal reduction
    print(f"\nTesting modal reduction with {num_modes} modes...")
    
    # Step 1: Get the first num_modes for modal basis
    Psi_k = dry_modes_full[:, :num_modes]
    basis_frequencies = dry_freq_full[:num_modes]
    
    print(f"Modal basis frequencies: {[f'{f:.3f}' for f in basis_frequencies]} Hz")
    
    # Step 2: Project matrices onto modal space
    M_hat = Psi_k.T @ M @ Psi_k
    K_hat = Psi_k.T @ K @ Psi_k
    
    print(f"Reduced system size: {M_hat.shape[0]} x {M_hat.shape[1]}")
    
    # Step 3: Solve reduced eigenvalue problem
    try:
        eigenvals_reduced, eigenvecs_reduced = eigh(K_hat, M_hat)
        
        # Convert to frequencies
        freq_reduced = np.sqrt(np.abs(eigenvals_reduced)) / (2 * np.pi)
        
        # Sort by frequency
        idx = np.argsort(freq_reduced)
        freq_reduced = freq_reduced[idx]
        eigenvecs_reduced = eigenvecs_reduced[:, idx]
        
        # Transform back to full space
        modes_reduced = Psi_k @ eigenvecs_reduced
        
        print(f"Reduced system frequencies: {[f'{f:.3f}' for f in freq_reduced]} Hz")
        
        # Compare with reference
        print("\nComparison with full analysis:")
        print(f"{'Mode':<6} {'Full (Hz)':<12} {'Reduced (Hz)':<15} {'Error (%)':<10}")
        print("-" * 50)
        
        for i in range(min(len(freq_reduced), len(basis_frequencies))):
            error = abs(freq_reduced[i] - basis_frequencies[i]) / basis_frequencies[i] * 100
            print(f"{i+1:<6} {basis_frequencies[i]:<12.3f} {freq_reduced[i]:<15.3f} {error:<10.2f}")
        
        return freq_reduced, modes_reduced, basis_frequencies
        
    except Exception as e:
        print(f"Error in reduced eigenvalue problem: {e}")
        return None, None, None

def test_modal_reduction_wet(M, C, K, M_W, C_W, num_modes=5, freq_threshold=1.0):
    """
    Test modal reduction technique with fluid matrices added
    """
    print("\n" + "="*60)
    print("STEP 3: MODAL REDUCTION WITH FLUID MATRICES")
    print("="*60)
    
    # First get dry modal basis (same as before)
    dry_freq_full, dry_modes_full = compute_dry_modes_full(M, K, num_modes*3, freq_threshold)
    
    if dry_freq_full is None:
        return None, None, None
    
    # Use first num_modes as basis
    Psi_k = dry_modes_full[:, :num_modes]
    basis_frequencies = dry_freq_full[:num_modes]
    
    print(f"Using dry modal basis with frequencies: {[f'{f:.3f}' for f in basis_frequencies]} Hz")
    
    # Project matrices including fluid effects
    M_total = M + M_W
    C_total = C + C_W
    
    M_hat = Psi_k.T @ M_total @ Psi_k
    C_hat = Psi_k.T @ C_total @ Psi_k
    K_hat = Psi_k.T @ K @ Psi_k
    
    print(f"Fluid effects on mass matrix magnitude: {np.linalg.norm(M_W) / np.linalg.norm(M):.3f}")
    print(f"Fluid effects on damping matrix magnitude: {np.linalg.norm(C_W) / np.linalg.norm(C):.3f}")
    
    # Solve wet eigenvalue problem
    try:
        # Method 1: Undamped approximation (ignore damping for frequency calculation)
        eigenvals_wet, eigenvecs_wet = eigh(K_hat, M_hat)
        
        # Filter positive eigenvalues
        positive_mask = eigenvals_wet > 1e-10
        eigenvals_wet = eigenvals_wet[positive_mask]
        eigenvecs_wet = eigenvecs_wet[:, positive_mask]
        
        # Sort by frequency
        idx = np.argsort(eigenvals_wet)
        eigenvals_wet = eigenvals_wet[idx]
        eigenvecs_wet = eigenvecs_wet[:, idx]
        
        # Convert to frequencies
        wet_frequencies = np.sqrt(eigenvals_wet) / (2 * np.pi)
        
        # Transform back to full space
        wet_modes = Psi_k @ eigenvecs_wet
        
        print(f"Wet frequencies: {[f'{f:.3f}' for f in wet_frequencies]} Hz")
        
        # Compare dry vs wet
        print("\nDry vs Wet frequency comparison:")
        print(f"{'Mode':<6} {'Dry (Hz)':<12} {'Wet (Hz)':<12} {'Reduction (%)':<15}")
        print("-" * 55)
        
        for i in range(min(len(wet_frequencies), len(basis_frequencies))):
            reduction = (basis_frequencies[i] - wet_frequencies[i]) / basis_frequencies[i] * 100
            print(f"{i+1:<6} {basis_frequencies[i]:<12.3f} {wet_frequencies[i]:<12.3f} {reduction:<15.1f}")
        
        return wet_frequencies, wet_modes, basis_frequencies
        
    except Exception as e:
        print(f"Error in wet eigenvalue problem: {e}")
        
        # Try alternative method with damping
        try:
            print("Trying complex eigenvalue approach...")
            wet_freq_complex, wet_modes_complex = solve_damped_system(M_hat, C_hat, K_hat, Psi_k)
            return wet_freq_complex, wet_modes_complex, basis_frequencies
            
        except Exception as e2:
            print(f"Complex eigenvalue method also failed: {e2}")
            return None, None, None

def solve_damped_system(M_hat, C_hat, K_hat, Psi_k):
    """
    Solve damped system using state-space approach
    """
    n = M_hat.shape[0]
    
    # State-space formulation: [M 0; 0 I][ẍ; ẋ] + [C K; -I 0][ẋ; x] = 0
    A = np.zeros((2*n, 2*n))
    B = np.zeros((2*n, 2*n))
    
    A[:n, :n] = C_hat
    A[:n, n:] = K_hat
    A[n:, n:] = -np.eye(n)
    
    B[:n, :n] = M_hat
    B[n:, n:] = np.eye(n)
    
    # Solve generalized eigenvalue problem
    eigenvals, eigenvecs = eigh(A, B)
    
    # Extract physical frequencies (positive imaginary parts)
    frequencies = []
    mode_shapes = []
    
    for i, lam in enumerate(eigenvals):
        if np.imag(lam) > 0 and np.abs(np.imag(lam)) > 1e-6:
            freq = np.abs(np.imag(lam)) / (2 * np.pi)
            frequencies.append(freq)
            # Extract displacement part
            mode_shape = np.real(eigenvecs[n:, i])
            original_mode = Psi_k @ mode_shape
            mode_shapes.append(original_mode)
    
    if len(frequencies) == 0:
        raise ValueError("No physical modes found in damped analysis")
    
    frequencies = np.array(frequencies)
    mode_shapes = np.column_stack(mode_shapes) if mode_shapes else np.array([])
    
    # Sort by frequency
    idx = np.argsort(frequencies)
    frequencies = frequencies[idx]
    if mode_shapes.size > 0:
        mode_shapes = mode_shapes[:, idx]
    
    return frequencies, mode_shapes

def complete_modal_analysis_test(M, C, K, M_W, C_W, num_modes=5, expected_dry_freq=None):
    """
    Complete step-by-step modal analysis test
    """
    print("="*80)
    print("COMPLETE MODAL REDUCTION VALIDATION")
    print("="*80)
    
    # Run diagnostics
    diagnose_matrices(M, K, expected_dry_freq)
    
    # Step 1: Test modal reduction on dry system
    freq_dry_reduced, modes_dry_reduced, freq_dry_reference = test_modal_reduction_dry(
        M, K, num_modes, freq_threshold=1.0
    )
    
    if freq_dry_reduced is None:
        print("❌ Dry modal reduction test failed!")
        return None, None, None
    else:
        print("✅ Dry modal reduction test successful!")
    
    # Step 2: Test with fluid matrices
    freq_wet, modes_wet, freq_dry_basis = test_modal_reduction_wet(
        M, C, K, M_W, C_W, num_modes, freq_threshold=1.0
    )
    
    if freq_wet is None:
        print("❌ Wet modal analysis failed!")
        return freq_dry_reduced, None, freq_dry_reference
    else:
        print("✅ Wet modal analysis successful!")
    
    return freq_wet, modes_wet, freq_dry_reference

# Example usage function
def test_with_simple_system():
    """
    Test with a simple 2-DOF system to verify the approach
    """
    print("Testing with simple 2-DOF system...")
    
    # Simple 2-DOF system
    M = np.array([[2.0, 0.0], [0.0, 1.0]])
    K = np.array([[3.0, -1.0], [-1.0, 2.0]])
    C = 0.01 * (M + K)
    
    # Added fluid matrices
    M_W = np.array([[0.5, 0.1], [0.1, 0.3]])
    C_W = np.array([[0.2, 0.05], [0.03, 0.15]])
    
    return complete_modal_analysis_test(M, C, K, M_W, C_W, num_modes=2)


def compute_mac_matrix(dry_modes, wet_modes):
    """
    Compute the MAC (Modal Assurance Criterion) between dry and wet modes.

    Parameters:
    -----------
    dry_modes : ndarray (n_dof, n_dry_modes)
        Matrix of dry mode shapes (columns are mode vectors).
    wet_modes : ndarray (n_dof, n_wet_modes)
        Matrix of wet mode shapes (same DOFs as dry_modes).
        
    Returns:
    --------
    MAC : ndarray (n_dry_modes, n_wet_modes)
        MAC matrix, where MAC[i, j] is the correlation between dry mode i and wet mode j.
    """
    n_dry = dry_modes.shape[1]
    n_wet = wet_modes.shape[1]
    MAC = np.zeros((n_dry, n_wet))

    for i in range(n_dry):
        phi_dry = dry_modes[:, i]
        for j in range(n_wet):
            phi_wet = wet_modes[:, j]
            num = np.abs(np.vdot(phi_dry, phi_wet)) ** 2
            den = np.vdot(phi_dry, phi_dry) * np.vdot(phi_wet, phi_wet)
            MAC[i, j] = num / den if den != 0 else 0.0

    return MAC


if __name__ == "__main__":
    # Run simple test
    test_with_simple_system()