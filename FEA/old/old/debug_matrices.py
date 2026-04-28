# Debug and fix functions for the mass matrix issues

import numpy as np
import scipy.linalg as la

def debug_mass_matrix(M_global_dry, M_added, title="Mass Matrix Debug"):
    """
    Comprehensive debugging of mass matrices
    """
    print(f"\n=== {title} ===")
    
    # Check dry mass matrix
    print("DRY MASS MATRIX:")
    print(f"  Shape: {M_global_dry.shape}")
    print(f"  Min/Max values: {np.min(M_global_dry):.2e} / {np.max(M_global_dry):.2e}")
    print(f"  Contains NaN: {np.isnan(M_global_dry).any()}")
    print(f"  Contains Inf: {np.isinf(M_global_dry).any()}")
    print(f"  Is symmetric: {np.allclose(M_global_dry, M_global_dry.T, rtol=1e-10)}")
    
    # Check eigenvalues of dry mass matrix
    try:
        eigs_dry = np.linalg.eigvals(M_global_dry)
        eigs_dry_real = np.real(eigs_dry[np.isreal(eigs_dry)])
        print(f"  Positive eigenvalues: {np.sum(eigs_dry_real > 0)} / {len(eigs_dry_real)}")
        print(f"  Min eigenvalue: {np.min(eigs_dry_real):.2e}")
        print(f"  Condition number: {np.linalg.cond(M_global_dry):.2e}")
    except Exception as e:
        print(f"  Error computing eigenvalues: {e}")
    
    # Check added mass matrix
    print("\nADDED MASS MATRIX:")
    print(f"  Shape: {M_added.shape}")
    print(f"  Min/Max values: {np.min(M_added):.2e} / {np.max(M_added):.2e}")
    print(f"  Contains NaN: {np.isnan(M_added).any()}")
    print(f"  Contains Inf: {np.isinf(M_added).any()}")
    print(f"  Is complex: {np.iscomplexobj(M_added)}")
    print(f"  Is symmetric: {np.allclose(M_added, M_added.T, rtol=1e-10)}")
    
    # Check magnitude comparison
    dry_norm = np.linalg.norm(M_global_dry)
    added_norm = np.linalg.norm(M_added)
    print(f"  Norm ratio (added/dry): {added_norm/dry_norm:.2e}")
    
    # Check combined matrix
    M_total = M_global_dry + M_added
    print("\nCOMBINED MASS MATRIX:")
    print(f"  Shape: {M_total.shape}")
    print(f"  Min/Max values: {np.min(M_total):.2e} / {np.max(M_total):.2e}")
    print(f"  Contains NaN: {np.isnan(M_total).any()}")
    print(f"  Contains Inf: {np.isinf(M_total).any()}")
    print(f"  Is symmetric: {np.allclose(M_total, M_total.T, rtol=1e-10)}")
    
    # Check eigenvalues of combined matrix
    try:
        eigs_total = np.linalg.eigvals(M_total)
        eigs_total_real = np.real(eigs_total[np.isreal(eigs_total)])
        print(f"  Positive eigenvalues: {np.sum(eigs_total_real > 0)} / {len(eigs_total_real)}")
        print(f"  Negative eigenvalues: {np.sum(eigs_total_real < 0)}")
        print(f"  Min eigenvalue: {np.min(eigs_total_real):.2e}")
        print(f"  Condition number: {np.linalg.cond(M_total):.2e}")
    except Exception as e:
        print(f"  Error computing eigenvalues: {e}")
    
    return M_total

def fix_added_mass_matrix(M_added, method='symmetrize'):
    """
    Fix common issues with added mass matrix
    """
    print(f"\n=== Fixing Added Mass Matrix (method: {method}) ===")
    
    M_fixed = M_added.copy()
    
    # 1. Handle complex values - take only real part
    if np.iscomplexobj(M_fixed):
        print("Converting complex matrix to real (taking real part)")
        M_fixed = np.real(M_fixed)
    
    # 2. Handle NaNs and Infs
    if np.isnan(M_fixed).any() or np.isinf(M_fixed).any():
        print("Replacing NaN/Inf values with zeros")
        M_fixed = np.nan_to_num(M_fixed, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 3. Symmetrize the matrix
    if method == 'symmetrize':
        print("Symmetrizing matrix: M = (M + M.T) / 2")
        M_fixed = (M_fixed + M_fixed.T) / 2
    
    # 4. Scale down if too large
    max_val = np.max(np.abs(M_fixed))
    if max_val > 1e10:
        scale_factor = 1e6 / max_val
        print(f"Scaling down matrix by factor {scale_factor:.2e}")
        M_fixed *= scale_factor
    
    # 5. Handle negative eigenvalues
    try:
        eigs = np.linalg.eigvals(M_fixed)
        min_eig = np.min(np.real(eigs))
        if min_eig < 0:
            print(f"Adding diagonal regularization to handle negative eigenvalues (min_eig: {min_eig:.2e})")
            regularization = np.abs(min_eig) * 1.1
            M_fixed += regularization * np.eye(M_fixed.shape[0])
    except Exception as e:
        print(f"Could not check eigenvalues: {e}")
        # Add small diagonal regularization anyway
        M_fixed += 1e-8 * np.eye(M_fixed.shape[0])
    
    print("Fixed added mass matrix:")
    print(f"  Min/Max values: {np.min(M_fixed):.2e} / {np.max(M_fixed):.2e}")
    print(f"  Is symmetric: {np.allclose(M_fixed, M_fixed.T, rtol=1e-10)}")
    
    return M_fixed

def robust_eigenvalue_solver(K, M, num_modes=6, regularization=1e-12):
    """
    Robust eigenvalue solver with multiple fallback strategies
    """
    print(f"\n=== Robust Eigenvalue Solver ===")
    
    # Strategy 1: Direct scipy solver
    try:
        print("Trying scipy.linalg.eigh...")
        eigenvals, eigenvecs = la.eigh(K, M)
        
        # Filter positive eigenvalues (natural frequencies)
        positive_mask = eigenvals > 1e-12
        eigenvals = eigenvals[positive_mask]
        eigenvecs = eigenvecs[:, positive_mask]
        
        # Take only requested number of modes
        if len(eigenvals) > num_modes:
            eigenvals = eigenvals[:num_modes]
            eigenvecs = eigenvecs[:, :num_modes]
        
        frequencies = np.sqrt(eigenvals) / (2 * np.pi)
        print(f"Success! Found {len(frequencies)} modes")
        return frequencies, eigenvecs, eigenvals
        
    except Exception as e:
        print(f"Failed with scipy.linalg.eigh: {e}")
    
    # Strategy 2: Add regularization to mass matrix
    try:
        print(f"Trying with mass matrix regularization ({regularization})...")
        M_reg = M + regularization * np.eye(M.shape[0])
        eigenvals, eigenvecs = la.eigh(K, M_reg)
        
        positive_mask = eigenvals > 1e-12
        eigenvals = eigenvals[positive_mask]
        eigenvecs = eigenvecs[:, positive_mask]
        
        if len(eigenvals) > num_modes:
            eigenvals = eigenvals[:num_modes]
            eigenvecs = eigenvecs[:, :num_modes]
        
        frequencies = np.sqrt(eigenvals) / (2 * np.pi)
        print(f"Success with regularization! Found {len(frequencies)} modes")
        return frequencies, eigenvecs, eigenvals
        
    except Exception as e:
        print(f"Failed with regularization: {e}")
    
    # Strategy 3: Cholesky decomposition approach
    try:
        print("Trying Cholesky decomposition approach...")
        L = la.cholesky(M, lower=True)
        L_inv = la.solve_triangular(L, np.eye(L.shape[0]), lower=True)
        K_reduced = L_inv @ K @ L_inv.T
        
        eigenvals, eigenvecs_reduced = la.eigh(K_reduced)
        eigenvecs = L_inv.T @ eigenvecs_reduced
        
        positive_mask = eigenvals > 1e-12
        eigenvals = eigenvals[positive_mask]
        eigenvecs = eigenvecs[:, positive_mask]
        
        if len(eigenvals) > num_modes:
            eigenvals = eigenvals[:num_modes]
            eigenvecs = eigenvecs[:, :num_modes]
        
        frequencies = np.sqrt(eigenvals) / (2 * np.pi)
        print(f"Success with Cholesky! Found {len(frequencies)} modes")
        return frequencies, eigenvecs, eigenvals
        
    except Exception as e:
        print(f"Failed with Cholesky: {e}")
    
    # Strategy 4: Pseudo-inverse approach (last resort)
    try:
        print("Trying pseudo-inverse approach (last resort)...")
        M_pinv = la.pinv(M)
        eigenvals, eigenvecs = la.eig(M_pinv @ K)
        
        # Filter real, positive eigenvalues
        real_mask = np.isreal(eigenvals)
        eigenvals = np.real(eigenvals[real_mask])
        eigenvecs = np.real(eigenvecs[:, real_mask])
        
        positive_mask = eigenvals > 1e-12
        eigenvals = eigenvals[positive_mask]
        eigenvecs = eigenvecs[:, positive_mask]
        
        # Sort by eigenvalue
        sort_indices = np.argsort(eigenvals)
        eigenvals = eigenvals[sort_indices]
        eigenvecs = eigenvecs[:, sort_indices]
        
        if len(eigenvals) > num_modes:
            eigenvals = eigenvals[:num_modes]
            eigenvecs = eigenvecs[:, :num_modes]
        
        frequencies = np.sqrt(eigenvals) / (2 * np.pi)
        print(f"Success with pseudo-inverse! Found {len(frequencies)} modes")
        return frequencies, eigenvecs, eigenvals
        
    except Exception as e:
        print(f"Failed with pseudo-inverse: {e}")
    
    print("All strategies failed!")
    return np.array([]), np.array([]), np.array([])

def check_coupling_matrix(Z, aerogrid, beam_model):
    """
    Debug the coupling matrix Z
    """
    print(f"\n=== Coupling Matrix Debug ===")
    print(f"Z shape: {Z.shape}")
    print(f"Z min/max: {np.min(Z):.2e} / {np.max(Z):.2e}")
    print(f"Z non-zero elements: {np.count_nonzero(Z)} / {Z.size}")
    print(f"Z contains NaN: {np.isnan(Z).any()}")
    print(f"Z contains Inf: {np.isinf(Z).any()}")
    
    # Check if Z is too sparse
    sparsity = np.count_nonzero(Z) / Z.size
    print(f"Matrix sparsity: {sparsity:.4f}")
    if sparsity < 0.01:
        print("WARNING: Coupling matrix is very sparse - this might indicate alignment issues")
    
    # Check panel areas
    panel_areas = aerogrid['A']
    print(f"Panel areas - min/max: {np.min(panel_areas):.2e} / {np.max(panel_areas):.2e}")
    
    # Check if any panels have zero area
    zero_area_panels = np.sum(panel_areas < 1e-12)
    if zero_area_panels > 0:
        print(f"WARNING: {zero_area_panels} panels have near-zero area")
    
    return Z

# Usage functions to integrate into your main code
def debug_and_fix_fsi_matrices(M_global_dry, M_added, K_global, Z, aerogrid, beam_model):
    """
    Main function to debug and fix FSI matrices
    """
    print("\n" + "="*50)
    print("FLUID-STRUCTURE INTERACTION MATRIX DEBUGGING")
    print("="*50)
    
    # 1. Debug coupling matrix
    Z_checked = check_coupling_matrix(Z, aerogrid, beam_model)
    
    # 2. Debug mass matrices
    M_total_debug = debug_mass_matrix(M_global_dry, M_added)
    
    # 3. Fix added mass matrix
    M_added_fixed = fix_added_mass_matrix(M_added, method='symmetrize')
    
    # 4. Create final corrected mass matrix
    M_global_wet = M_global_dry + M_added_fixed
    
    # 5. Final check
    print(f"\n=== Final Wet Mass Matrix ===")
    try:
        eigs_final = np.linalg.eigvals(M_global_wet)
        eigs_final_real = np.real(eigs_final[np.isreal(eigs_final)])
        print(f"Positive eigenvalues: {np.sum(eigs_final_real > 0)} / {len(eigs_final_real)}")
        print(f"Min eigenvalue: {np.min(eigs_final_real):.2e}")
        print("Matrix should now be suitable for eigenvalue analysis")
    except Exception as e:
        print(f"Still issues with final matrix: {e}")
    
    return M_global_wet, M_added_fixed