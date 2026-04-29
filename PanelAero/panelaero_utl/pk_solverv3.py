import numpy as np
from numpy.linalg import norm, inv
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from numpy.linalg import solve as lin_solve
import warnings

# Import utilities
from .pk_method_utl import mode_matching


class PKSolverV3:


    def __init__(self, M_hat, C_hat, K_hat, omega_n, k_guess):

        self.M = M_hat
        self.C = C_hat
        self.K = K_hat
        self.omega_n = np.sqrt(omega_n)  # Ensure we have frequencies
        self.k_guess = k_guess  # Initial reduced frequency guess

        assert self.M.shape[0] == self.M.shape[1]
        self.n = self.M.shape[0]
        
        # History tracking for extrapolation
        self.results_history = []  # Store last 2 velocity steps
        
        # Diagnostic storage
        self.omega_history = []
        self.mode_switch_warnings = 0
        self.convergence_issues = 0
        self.extrapolation_failures = 0
        
        # Quality control
        self.last_good_omegas = None
        self.iteration_log = []  # Detailed iteration log
        
        # Storage for stiffness and damping contributions at each converged speed
        self.contributions_history = []  # List of dicts with V, K_struct, K_aero, C_struct, C_aero
        
        # Storage for w_modal components at each converged speed
        self.w_modal_history = []  # List of dicts with V, w_velocity_norm, w_slope_norm
        
        print("\n" + "="*70)
        print("PK-METHOD SOLVER V3 - PAPER-COMPLIANT")
        print("="*70)
        print(f"Number of modes: {len(self.omega_n)}")
        print(f"Structural natural frequencies:")
        for i in range(min(5, len(self.omega_n))):
            print(f"  Mode {i}: {self.omega_n[i]:.2f} rad/s {self.omega_n[i]/(2*np.pi):.2f} Hz")
        print("="*70 + "\n")


    def _Ac(self, V, rho, b, k, Qhh, store_contributions=False):
        
        n = self.n
        I = np.eye(n)
        Z = np.zeros((n, n))
        #L = 2 * np.abs(np.cos(20*np.pi/180))  # Effective length for added mass (flat plate approximation)

        #qinf_0 = 2.609 * rho * ((2*b * L)/np.pi)**(3/2)
        qinf = 0.5 * rho * V**2
        # MODIFICATION FOR PANELAERO: Remove 'b' factor (no semicorda dependence)
        # PanelAero uses s_bar = s/V, not s_bar = (Vs*b)/... like ETH
        # Therefore: [C] = [C_s] - (1/2)*rho*V*[A1], not with 'b'
        qinf_I = 0.5 * rho * V / k

        #M_aero = qinf_0 * np.real(Qhh)
        K_aero = qinf * np.real(Qhh)
        C_aero = qinf_I * np.imag(Qhh)

        # Paper Eq. 1: [M p² + (B - ¼ρcV Q^I/k)p + (K - ½ρV² Q^R)]{u} = 0
        RHS_M = self.M #+ M_aero
        RHS_K = self.K - K_aero
        RHS_C = self.C + C_aero

        K_struct = self.K
        K_aero = qinf * np.real(Qhh)
        C_struct = self.C
        C_aero = qinf_I * np.imag(Qhh)

        A21 = -np.linalg.solve(RHS_M, RHS_K)
        A22 = -np.linalg.solve(RHS_M, RHS_C)

        # Paper Eq. 3: Canonical form
        Ac = np.block([[Z, I],
                       [A21, A22]])
        
        if store_contributions:
            return Ac, {
                'K_struct': K_struct,
                'K_aero': K_aero,
                'C_struct': C_struct,
                'C_aero': C_aero,
                'M_struct': self.M,
                'M_effective': RHS_M,
                'K_effective': RHS_K,
                'C_effective': RHS_C

            }
        
        return Ac


    def _predict_eigenvalue_linear(self, s, results_i_minus_1, results_i_minus_2):

        if results_i_minus_1 is None or results_i_minus_2 is None:
            return None, False
        
        if s >= len(results_i_minus_1['modes']) or s >= len(results_i_minus_2['modes']):
            return None, False
        
        p_v1 = results_i_minus_1['modes'][s]['p']
        p_v2 = results_i_minus_2['modes'][s]['p']
        
        # Check if mode was lost in previous steps (p will be NaN)
        if np.isnan(p_v1) or np.isnan(p_v2):
            return None, False
        
        # Paper Eq. (Section 3.2.1, Step 1)
        p_ext = 2.0 * p_v1 - p_v2
        
        return p_ext, True


    def _hybrid_k_guess(self, s, V, b, omega_s_prev_iter, omega_s_minus_1_converged, fXK0=0.618):

        if s == 0:
            return self.k_guess  # First mode: use configured default
        
        # Check if previous mode (s-1) was lost
        if np.isnan(omega_s_minus_1_converged):
            # Previous mode lost → cannot use hybrid formula
            # Fallback to k_base only
            print(f"      Note: Previous mode lost, using k_base without hybridization")
            k_base = omega_s_prev_iter / V
            return k_base
        
        k_base = omega_s_prev_iter / V
        k_converged = omega_s_minus_1_converged / V
        
        # Paper Eq. 15
        k_hybrid = k_base + fXK0 * (k_converged - k_base)
        
        # Sanity check
        if k_hybrid < 1e-6:
            print(f"      Warning: k_hybrid too small ({k_hybrid:.2e}), using {self.k_guess:.2e}")
            k_hybrid = self.k_guess
        
        return k_hybrid


    def _relax_k(self, k_old, omega_new, V, b, fRLX=0.618):

        print (f"      Relaxing k: old={k_old:.6g}, new={omega_new/V:.6g}, fRLX={fRLX}")
        k_target = omega_new / V
        k_new = k_old + fRLX * (k_target - k_old)
        k_new = max(k_new, 1e-6)

        return k_new


    def _normalize_and_fix_sign(self, u, reference=None):

        # Mass-normalization: u^T M u = 1
        n_half = len(u) // 2
        u_h = u[:n_half]
        
        norm_factor = np.sqrt(np.abs(u_h.conj().T @ self.M @ u_h))
        if norm_factor < 1e-12:
            norm_factor = np.linalg.norm(u)
        
        u_norm = u / norm_factor
        
        # Sign fixing
        if reference is not None:
            # Match sign to reference
            dot_product = np.real(u_norm.conj().T @ reference)
            if dot_product < 0:
                u_norm = -u_norm
        else:
            # Fix sign by making first significant element positive
            significant_idx = np.argmax(np.abs(u_norm[:n_half]))
            if np.real(u_norm[significant_idx]) < 0:
                u_norm = -u_norm
        
        return u_norm


    def solve_at_velocity(self, i, V, V_list, dry_vals, dry_vecs, k0_guess, rho, b, 
                         Qg_func, modes, max_iter, tol,
                         fXK0, fRLX, freq_margin, w_freq, w_mac, results, perturb_k,
                         mac_matching=True, last_converged_mode_matching=True):

        results_for_V = []
        last_sorted_omegas = []
        converged_omegas_at_V = []
        converged_vecs_at_V = []
        
        # Extract converged frequencies from previous velocity step (V_{i-1})
        # Mode matching bounds values to store betwn iterations (Paper Section 3.2.2)
        last_converged_omegas = []
        if i > 0 and len(results) > 0 and 'modes' in results[i-1]:
            for s in modes:
                if s < len(results[i-1]['modes']):
                    last_converged_omegas.append(results[i-1]['modes'][s]['omega'])
                else:
                    # Fallback to structural frequency if mode not found
                    last_converged_omegas.append(self.omega_n[s])

        print(f"\n  Solving at V = {V:.2f} m/s (fXK0={fXK0}, fRLX={fRLX})")
        
        for s in modes:
            print(f"\n  === MODE {s} ===")
            
            # Check if this mode was already lost in previous velocity step
            if i > 0 and s < len(results[i-1]['modes']):
                prev_mode_data = results[i-1]['modes'][s]
                if prev_mode_data.get('mode_lost', False):
                    print(f"    ⊗ MODE {s} already lost at previous velocity step")
                    print(f"       → Skipping (inserting NaN)")
                    
                    # Insert NaN for this mode (already lost)
                    results_for_V.append({
                        "mode": int(s),
                        "p": np.nan,
                        "omega": np.nan,
                        "sigma": np.nan,
                        "gamma": np.nan,
                        "u": None,
                        "k": np.nan,
                        "vecs": None,
                        "converged": False,
                        "iterations": 0,
                        "mode_lost": True
                    })
                    
                    continue
            
            # STEP 1: Initial guess for k
            if s == 0:
                if i == 0:

                    k = self.k_guess  # Small default for first mode at first velocity
                    p_initial_guess = None
                    print(f"    k_initial = {k:.6g} (from structural ω_n[0]={self.omega_n[0]:.3f} rad/s)")
                else:
                    # Subsequent steps: use previous-step converged frequency
                    k = k0_guess  # already set by caller from previous result
                    p_initial_guess = None
                    print(f"    k_initial = {k:.6g} (from previous velocity step)")

            else:
                # Check if this is the first velocity step
                if i == 0 and last_sorted_omegas[s] < 8:
                    # First velocity: use structural frequency for k-guess
                    k = self.k_guess
                    p_initial_guess = None
                    print(f"    k_initial = {k:.6g} (from structural ω_n[{s}]={self.omega_n[s]:.3f} rad/s)")
                              
                else:
                    # Use hybrid k-guess (Paper Eq. 15)
                    omega_s_prev = last_sorted_omegas[s] if s < len(last_sorted_omegas) else self.omega_n[s]
                    omega_s_minus_1 = results_for_V[s-1]['omega']
                    
                    k = self._hybrid_k_guess(s, V, b, omega_s_prev, omega_s_minus_1, fXK0)
                    print(f"    k_hybrid = {k:.6g} (Paper Eq. 15)")
                    
                    # Initial guess from previous mode
                    p_initial_guess = results_for_V[s-1]['p']
                    print(f"    p_initial_guess from mode {s-1}: {p_initial_guess:.6f}")
            
            # STEP 2: Get reference for extrapolation
            p_ext = None
            omega_ext = None
            use_extrapolation = False
            
            if i >= 2:
                # Linear extrapolation from 2 previous velocities (Paper Section 3.2.1)
                p_ext, success = self._predict_eigenvalue_linear(
                    s, results[i-1], results[i-2]
                )
                
                if success:
                    omega_ext = np.abs(np.imag(p_ext))
                    use_extrapolation = True
                    print(f"    Using linear extrapolation: p_ext = {p_ext:.6f}, ω_ext = {omega_ext:.3f} rad/s")
                else:
                    self.extrapolation_failures += 1
                    print(f"    Extrapolation failed, using previous velocity only")
            
            # Fallback to single-step prediction
            if not use_extrapolation and i > 0 and s < len(results[i-1]['modes']):
                p_ext = results[i-1]['modes'][s]['p']
                omega_ext = np.abs(np.imag(p_ext))
                #print(f"    Using previous velocity: p_prev = {p_ext:.6f}")
            
            # STEP 3: PK ITERATION
            p_prev_iter = p_initial_guess
            prev_iter_vec = None   # eigenvector selected at previous iteration (intra-velocity MAC)
            converged = False
            wet_mode_identified = False  # Flag to track if we've identified a valid wet mode for this structural mode
            
            for j in range(max_iter):
                if j == 0 or j % 10 == 0:
                    print(f"    Iteration {j}")

                # Debug: print current reduced frequency k and the frequency computed from it
                try:
                    omega_from_k = k * V
                except Exception:
                    omega_from_k = float('nan')
                print(f"      [K-DEBUG] Iter {j}: k={k:.6g}, ω_from_k={omega_from_k:.6f} rad/s")
                
                # Build aeroelastic matrix
                wet_vecs = dry_vecs  # Could use converged wet modes here
                
                # Update Qhh every iteration with current p
                if p_prev_iter is not None:
                    Qhh = Qg_func(k, V, p_prev_iter, s, j)
                else:
                    Qhh = Qg_func(k, V, None, s, j)
                
                Ac_result = self._Ac(V, rho, b, k, Qhh, store_contributions=False)
                if isinstance(Ac_result, tuple):
                    Ac, contributions = Ac_result
                else:
                    Ac = Ac_result
                    contributions = None
                
                # Solve eigenvalue problem
                vals_all, vecs_all = eig(Ac)
                
                # Keep positive imaginary parts only, keep only physically meaningful modes.
                imag = np.imag(vals_all);       pos_idx = np.where(imag >= -1e-3)[0]   # allow ω → 0
                vals = vals_all[pos_idx];       vecs = vecs_all[:, pos_idx]
                
                # Sort by frequency
                omegas = np.abs(np.imag(vals));     order = np.argsort(omegas)
                vals = vals[order];       vecs = vecs[:, order]
                omegas = omegas[order]

                # ── Strip spurious near-zero eigenvalues ──────────────────────────
                # The state-space Ac matrix can produce extra near-zero roots
                # (numerical artefacts, rigid-body-like modes from the aero lag
                # states, etc.).  When they sit at ω ≈ 0 they are sorted to the
                # front and corrupt the mode selection at first velocity step.
                #
                # Heuristic: discard any eigenvalue whose frequency is less than
                # 10% of the lowest structural frequency. We only apply this trim
                # when there are *more* roots than modes to avoid discarding a
                # legitimate near-zero flutter mode.
                omega_phys_min = self.omega_n[0] * 0.1 if len(self.omega_n) > 0 else 1.0
                phys_mask = omegas >= omega_phys_min
                n_phys = int(np.sum(phys_mask))
                if n_phys >= len(self.omega_n):
                    # Safe to strip — we still have enough physical roots
                    n_stripped = len(omegas) - n_phys
                    if n_stripped > 0 and j == 0:  # Only print at first iteration to avoid spam
                        print(f"      [strip] Removed {n_stripped} near-zero root(s) "
                              f"(ω < {omega_phys_min:.2f} rad/s) from sorted set")
                    vals  = vals[phys_mask]
                    vecs  = vecs[:, phys_mask]
                    omegas = omegas[phys_mask]
                # ─────────────────────────────────────────────────────────────────

                last_sorted_omegas = omegas
                
                # Log iteration
                self.iteration_log.append({
                    'velocity': V,
                    'mode': s,
                    'iteration': j,
                    'k': k,
                    'omegas': omegas.copy(),
                    'n_modes': len(omegas)
                })
                
                # DEBUG: Print all wet frequencies available at this iteration
                print(f"      [DEBUG] Wet frequencies at iteration {j}: {omegas[:10]} rad/s (showing first 10)")
                print(f"      [DEBUG] Total number of wet modes: {len(omegas)}")
                
                # STEP 4: Mode matching
                mode_freq_margin = freq_margin
                
                # Retrieve the converged eigenvector of THIS mode (s) from the
                # previous velocity step — used as the MAC reference so that MAC
                # measures cross-speed continuity of mode s, not orthogonality
                # between adjacent modes at the same speed.
                # NOTE: This will be None at first velocity (i==0), which disables MAC.
                prev_step_vec = None
                if i > 0 and s < len(results[i-1]['modes']):
                    prev_step_vec = results[i-1]['modes'][s].get('u', None)
                
                # Special case: First velocity step, first iteration -> match to structural frequency
                if i == 0 and j == 0 and V == V_list[0]: #or wet_mode_identified == False:
                    # At V0, iteration 0: select mode closest to structural natural frequency
                    # This avoids picking up spurious near-zero eigenvalues
                    omega_struct = self.omega_n[s]
                    freq_errors = np.abs(omegas - omega_struct)
                    closest_idx = int(np.argmin(freq_errors))

                    if freq_errors[closest_idx] > 0.3 * omega_struct:
                        print(f"      ⚠ Warning: Selected mode differs by {freq_errors[closest_idx]/omega_struct*100:.1f}% from structural frequency")
                        print(f"                 RUnning the analysis again with k_relaxed")
                        # update ?!
                        p_sel = 0 + 1j * omega_struct  # Force mode matching to fail and trigger relaxed k guess in next iteration
                        u_sel = dry_vecs[:, s]
                        
                        p_sel = vals[closest_idx]
                        u_sel = vecs[:, closest_idx]
                        
                        wet_mode_identified = False
                    else: 
                        print(f"      Selected mode is within 30% of structural frequency, proceeding with this match")
                        p_sel = vals[closest_idx]
                        u_sel = vecs[:, closest_idx]
                        print(f"      Using structural frequency matching: ω_struct={omega_struct:.3f} rad/s "
                              f"→ selected ω={np.abs(np.imag(p_sel)):.3f} rad/s (index {closest_idx})")
                        
                        wet_mode_identified = True
                
                else:
                    # All other cases: conditional mode matching based on flags
                    if mac_matching:
                        # ── MAC-BASED MODE MATCHING ─────────────────────────────────
                        # Use full MAC-driven mode matching with eigenvector references
                        # At first velocity (i==0, j>0): prev_step_vec=None → MAC disabled
                        # At subsequent velocities (i>0): prev_step_vec available → MAC enabled
                        # prev_iter_vec carries the eigenvector from the previous PK iteration
                        # (same velocity step) and acts as intra-velocity MAC reference when
                        # no cross-velocity reference exists yet.
                        
                        # Predict p_target from last converged p values (for MAC < 0.7 fallback)
                        # Uses quadratic extrapolation (3 points) or linear (2 points) as fallback
                        p_target = mode_matching.predict_p_target_from_history(results, s, i)
                        if p_target is not None:
                            print(f"      [P_TARGET] Predicted from history: p={p_target:.6f}, ω={np.abs(np.imag(p_target)):.3f} rad/s")
                        else:
                            print(f"      [P_TARGET] Not available (need 2+ velocity steps)")
                        
                        _msw = [self.mode_switch_warnings]
                        p_sel, u_sel, j_sel = mode_matching.mode_matching_paper_compliant(
                            vals, vecs, s, j,
                            last_converged_omegas,
                            converged_omegas_at_V,
                            converged_vecs_at_V,
                            mode_freq_margin,
                            w_freq, w_mac,
                            self.omega_n, _msw,
                            prev_step_vec=prev_step_vec,
                            prev_iter_vec=prev_iter_vec,
                            p_target=p_target
                        )
                        self.mode_switch_warnings = _msw[0]
                    else:
                        # ── SIMPLE FREQUENCY-BASED MODE MATCHING ────────────────────
                        # Use simple frequency-based mode selection without MAC
                        print(f"      [MAC disabled] Using frequency-based mode selection")
                        
                        if last_converged_mode_matching and len(last_converged_omegas) > 0 and s < len(last_converged_omegas):
                            # Use last converged frequency with margin
                            omega_ref = last_converged_omegas[s]
                            lower_bound = omega_ref * (1 - mode_freq_margin)
                            upper_bound = omega_ref * (1 + mode_freq_margin)
                            print(f"      Using last converged frequency: ω_ref={omega_ref:.3f} rad/s, bounds=[{lower_bound:.3f}, {upper_bound:.3f}]")
                        else:
                            # Use structural natural frequency with margin
                            omega_ref = self.omega_n[s]
                            lower_bound = omega_ref * (1 - 0.2)  # ±20% margin
                            upper_bound = omega_ref * (1 + 0.2)
                            print(f"      Using structural frequency: ω_struct={omega_ref:.3f} rad/s, bounds=[{lower_bound:.3f}, {upper_bound:.3f}]")
                        
                        # Find closest mode within bounds
                        bound_mask = (omegas >= lower_bound) & (omegas <= upper_bound)
                        if np.any(bound_mask):
                            valid_indices = np.where(bound_mask)[0]
                            errors = np.abs(omegas[valid_indices] - omega_ref)
                            best_local = int(np.argmin(errors))
                            j_sel = valid_indices[best_local]
                            p_sel = vals[j_sel]
                            u_sel = vecs[:, j_sel]
                            print(f"      Selected: ω={np.abs(np.imag(p_sel)):.3f} rad/s (closest in bounds)")
                        else:
                            # No mode in bounds, find closest overall
                            errors = np.abs(omegas - omega_ref)
                            j_sel = int(np.argmin(errors))
                            p_sel = vals[j_sel]
                            u_sel = vecs[:, j_sel]
                            print(f"      ⚠ No mode in bounds, selected closest: ω={np.abs(np.imag(p_sel)):.3f} rad/s")
                
                # Check if mode was lost (no modes in bounds)
                if p_sel is None:
                    print(f"    ✗ MODE {s} NOT FOUND (no eigenvalues in frequency bounds)")
                    print(f"       → Mode has disappeared, inserting NaN")
                    
                    # Insert NaN result to signal mode loss
                    results_for_V.append({
                        "mode": int(s),
                        "p": np.nan,
                        "omega": np.nan,
                        "sigma": np.nan,
                        "gamma": np.nan,
                        "u": None,
                        "k": np.nan,
                        "vecs": None,
                        "converged": False,
                        "iterations": j + 1,
                        "mode_lost": True
                    })
                    
                    # Don't add to converged_omegas (mode is gone)
                    # Break out of iteration loop for this mode
                    converged = True
                    break
                
                # Normalize and fix sign
                #print(f"    Before normalization: ||u_sel||={norm(u_sel):.3e}, first element={u_sel[0]:.3e}")
                #u_sel = self._normalize_and_fix_sign(
                #    u_sel, 
                #    reference=converged_vecs_at_V[-1] if len(converged_vecs_at_V) > 0 else None
                #)
                #print(f"    After normalization: ||u_sel||={norm(u_sel):.3e}, first element={u_sel[0]:.3e}")
                
                # Extract eigenvalue components
                omega_sel = np.abs(np.imag(p_sel))
                sigma_sel = np.real(p_sel)
                
                if abs(omega_sel) > 1e-6:
                    gamma_sel = (sigma_sel / omega_sel)
                else:
                    gamma_sel = 0.0
                
                # STEP 5: Under-relaxation (Paper Eq. 17)
                k_new = self._relax_k(k, omega_sel, V, b, fRLX)
                print (f"      Updated k: {k_new:.6g} (from {k:.6g})")
                k_new_rounded = np.round(k_new, decimals=3)
                k_rounded = np.round(k, decimals=3)  # Round to avoid tiny oscillations causing infinite loops
                
                #if k_new_rounded == k_rounded:
                #    print(f"      k did not change after relaxation, breaking to avoid infinite loop")
                #    k_new = k * (1+perturb_k)  # Perturb k slightly to allow iteration to proceed
                
                # STEP 6: Convergence check
                k_change = abs(k_new - k)
                
                if k_change < tol:#and wet_mode_identified:
                    # Recompute Qhh with store_w_modal=True to get final w_modal components
                    Qg_result = Qg_func(k_new, V, p_sel, s, j, store_w_modal=True)
                    if isinstance(Qg_result, tuple):
                        Qhh_final, w_modal_data = Qg_result
                    else:
                        Qhh_final = Qg_result
                        w_modal_data = None
                    
                    # Recompute Ac with store_contributions=True to get final values
                    Ac_result = self._Ac(V, rho, b, k_new, Qhh_final, store_contributions=True)
                    if isinstance(Ac_result, tuple):
                        _, contributions = Ac_result
                    else:
                        contributions = None
                    
                    # Note: a wet frequency of zero is physically meaningful (divergence /
                    # flutter coalescence) and is NOT treated as mode loss here.
                    # The mode is kept and recorded normally so plots show the full curve.
                    
                    print(f"    ✓ CONVERGED at iteration {j}")
                    print(f"      wet_eigenvalues: {vals[:5]}... (first 5)")
                    print(f"      k = {k_new:.6g}, ω = {omega_sel:.4f} rad/s, γ = {gamma_sel:.6f}, p = {p_sel:.6f}")
                    
                    results_for_V.append({
                        "mode": int(s),
                        "p": complex(p_sel),
                        "omega": float(omega_sel),
                        "sigma": float(sigma_sel),
                        "gamma": float(gamma_sel),
                        "u": u_sel,
                        "k": float(k_new),
                        "vecs": vecs,  # Store all vecs for van Zyl later
                        "converged": True,
                        "iterations": j + 1,
                        "contributions": contributions,  # Store K and C contributions
                        "w_modal_data": w_modal_data  # Store w_modal components
                    })
                    
                    converged_omegas_at_V.append(omega_sel)
                    converged_vecs_at_V.append(u_sel)
                    converged = True
                    break
                
                # Update for next iteration
                k = k_new
                p_prev_iter = p_sel
                prev_iter_vec = u_sel   # carry eigenvector forward for intra-velocity MAC
                
                if j % 10 == 0 and j > 0:
                    print(f"      k={k:.6g}, ω={omega_sel:.4f} rad/s, Δk={k_change:.2e}")

            
            # Handle non-convergence
            if not converged:
                print(f"    ✗ NO CONVERGENCE after {max_iter} iterations")
                print(f"      Final: k={k:.6g}, ω={omega_sel:.4f} rad/s, γ={gamma_sel:.6f}")
                
                results_for_V.append({
                    "mode": int(s),
                    "p": complex(p_sel),
                    "omega": float(omega_sel),
                    "sigma": float(sigma_sel),
                    "gamma": float(gamma_sel),
                    "u": u_sel,
                    "k": float(k_new),
                    "vecs": vecs,
                    "converged": False,
                    "iterations": max_iter,
                    "warn": "no_convergence"
                })
                
                converged_omegas_at_V.append(omega_sel)
                self.convergence_issues += 1
        
        # Aggregate contributions for this velocity step
        # Store average contributions across all converged modes
        M_struct = []
        K_struct = []
        K_aero = []
        K_effective = []
        C_struct = []
        C_aero = []
        C_effective = []
        w_velocity = []
        w_slope = []
        
        for mode_data in results_for_V:
            if mode_data.get('converged', False):
                if 'contributions' in mode_data:
                    contrib = mode_data['contributions']
                    if contrib is not None:
                        M_struct.append(contrib['M_struct'])
                        K_struct.append(contrib['K_struct'])
                        K_aero.append(contrib['K_aero'])
                        K_effective.append(contrib['K_effective'])
                        C_struct.append(contrib['C_struct'])
                        C_aero.append(contrib['C_aero'])
                        C_effective.append(contrib['C_effective'])
                
                if 'w_modal_data' in mode_data:
                    w_data = mode_data['w_modal_data']
                    if w_data is not None:
                        w_velocity.append(w_data['w_velocity_norm'])
                        w_slope.append(w_data['w_slope_norm'])
        
        if len(K_struct) > 0:
            
            self.contributions_history.append({
                'V': float(V),
                'M_struct': M_struct,
                'K_struct': K_struct,
                'K_aero': K_aero,
                'K_effective': K_effective,
                'C_struct': C_struct,
                'C_aero': C_aero,
                'C_effective': C_effective
            })
        
        if len(w_velocity) > 0:
            self.w_modal_history.append({
                'V': float(V),
                'w_velocity_norm': float(np.mean(w_velocity)),
                'w_slope_norm': float(np.mean(w_slope))
            })
        
        return results_for_V


    def _check_frequency_continuity(self, results, max_jump=0.30):

        problems = []
        
        for i in range(1, len(results)):
            V_curr = results[i]['V']
            modes_prev = results[i-1]['modes']
            modes_curr = results[i]['modes']
            
            for s in range(min(len(modes_prev), len(modes_curr))):
                omega_prev = modes_prev[s]['omega']
                omega_curr = modes_curr[s]['omega']
                
                if omega_prev < 1e-6:
                    continue
                
                rel_change = abs(omega_curr - omega_prev) / omega_prev
                
                if rel_change > max_jump:
                    problems.append({
                        'velocity': V_curr,
                        'mode': s,
                        'omega_prev': omega_prev,
                        'omega_curr': omega_curr,
                        'rel_change': rel_change
                    })
        
        return problems


    def sweep(self, V_list, rho, b, Qg_func, modes, dry_vals, dry_vecs, 
              tol, fXK0, fRLX, freq_margin, perturb_k, w_freq, w_mac, max_iter, 
              mac_matching, last_converged_mode_matching, **kwargs):
        
        results = []
        k0_guess = self.k_guess  # Use configured k_guess instead of hardcoded 1e-3
        
        print("\n" + "="*70)
        print("STARTING PK-METHOD FLUTTER SWEEP (V3 - PAPER-COMPLIANT)")
        print("="*70)
        print(f"Number of velocities: {len(V_list)}")
        print(f"Number of modes: {len(modes)}")
        print(f"Parameters: fXK0={kwargs.get('fXK0', 0.618)}, fRLX={kwargs.get('fRLX', 0.618)}")
        print("="*70)
        
        for i, V in enumerate(V_list):
            print(f"\n{'='*70}")
            print(f"VELOCITY STEP {i+1}/{len(V_list)}: V = {V:.2f} m/s")
            print(f"{'='*70}")
            
            res_V = self.solve_at_velocity(
                i, V, V_list, dry_vals, dry_vecs, k0_guess, rho, b, 
                         Qg_func, modes, max_iter, tol,
                         fXK0, fRLX, freq_margin, w_freq, w_mac, results, perturb_k,
                         mac_matching, 
                         last_converged_mode_matching)
            
            results.append({"V": float(V), "modes": res_V})
            
            # Update k0_guess for the next velocity step using the converged k of mode 0.
            # This gives a physically meaningful starting reduced frequency for mode 0
            # at each new velocity, instead of the constant 1e-3 sentinel.
            if len(res_V) > 0 and res_V[0].get('converged', False) and not np.isnan(res_V[0].get('k', np.nan)):
                k0_guess = res_V[0]['k']
        
        print("\n" + "="*70)
        print("PK ITERATIONS COMPLETED")
        print("="*70)
        
        # Check continuity
        problems = self._check_frequency_continuity(results)
        if problems:
            print(f"\n⚠ WARNING: {len(problems)} frequency discontinuities detected:")
            for p in problems[:5]:  # Show first 5
                print(f"  V={p['velocity']:.2f} m/s, Mode {p['mode']}: "
                      f"{p['omega_prev']:.3f} → {p['omega_curr']:.3f} rad/s "
                      f"(Δ={100*p['rel_change']:.1f}%)")
        
        print("\n" + "="*70)
        print("FLUTTER SWEEP SUMMARY")
        print("="*70)
        print(f"Mode switch warnings: {self.mode_switch_warnings}")
        print(f"Convergence issues: {self.convergence_issues}")
        print(f"Extrapolation failures: {self.extrapolation_failures}")
        print("="*70 + "\n")
        
        return results


    def check_finite(self, array, name):
        """Check for NaN or Inf in arrays"""
        if np.any(np.isnan(array)):
            raise ValueError(f"NaN detected in {name}")
        if np.any(np.isinf(array)):
            raise ValueError(f"Inf detected in {name}")


    def make_Qg_func(self, config, FSI_path, Z, Apan, Z_qs, Phi, values, b, c_sound, 
                     out_dir_klist, out_dir_vjj, alpha_r, Z_force=None, panel_areas=None):

        
        from Qjj.precompute_qjj import open_qjj_index_old, interp_qjj_from_disk_old
        # VLM functions not available - commented out
        from Qjj.precompute_qjj_vlm import open_qjj_index_vlm, interp_qjj_vlm_from_disk
        

        k_list, Ma_list, shape, dtype = open_qjj_index_old(out_dir_klist)
        
        from panelaero_utl.pk_method_utl.roger_fit import build_Qroger

        def Qg_func(k_red, V_dlm, p_conv, s, j, store_w_modal=False, still_fluid_eigs=False):
            
            omega_dlm = k_red * V_dlm

            if k_red <= 30:
                k_dlm = k_red
                Ma_dlm = V_dlm / c_sound
            
                Qjj_dlm = interp_qjj_from_disk_old(out_dir_klist, k_dlm, Ma_dlm)

                
            else:
                raise ValueError(f"k_red={k_red:.6g} > 30, insufficient aerodynamic data")
            
            self.check_finite(Qjj_dlm, "Qjj_dlm")

            if j == 0 and s == 0:
                omega = k_red * V_dlm   # coerente con k
                p_over_V = 1j * omega / V_dlm # new
                p = 1j * omega
                #print(f"using p = {p:.6g} (coerente con k)")
            elif still_fluid_eigs:
                p_over_V = 1j * struct_freqs[s] / V_dlm
                p = 1j * struct_freqs[s]
                print(f"using p = {p:.6g} (from still fluid eigenvalues)")
                
            else:
                #omega = k_red * V_dlm   # coerente con k
                #p_over_V = 1j * omega / V_dlm # new
                #p = 1j * omega
                if p_conv is not None:
                    p_over_V = 1j * np.imag(p_conv) / V_dlm
                    p = 1j * np.imag(p_conv)
                    print(f"using p = {p:.6g} (from wet frequency convergence)")
                else:
                    struct_freqs = np.sqrt(values)
                    p_over_V = 1j * struct_freqs[s] / V_dlm
                    p = 1j * struct_freqs[s]
                    print(f"using p = {p:.6g} (from structural frequency)")


            jk_over_b = 1j * k_red  # jk (b dependency removed)
            w_velocity_modal = p_over_V * Z  @ Phi
            
            # Term 2: Slope contribution G_x
            w_slope_modal = Z_qs @ Phi

            # Apply aerodynamic operators to downwash components
            w_aero_velocity = (Qjj_dlm) @ w_velocity_modal
            w_aero_slope = (Qjj_dlm) @ w_slope_modal
            
            # Total downwash
            w_modal = w_aero_velocity + w_aero_slope

            Z_force_modal = Z_force @ Phi  # Project force influence matrix to modal coordinates
            
            # Build diagonal matrix A from panel areas
            A_diag = np.diag(panel_areas)  # (n_panels, n_panels)

            Q_modal = Z_force_modal.T @ A_diag @ w_modal

            self.check_finite(Q_modal, "Q_modal")

            if store_w_modal:
                # Return Q_modal and the two w_modal components
                return Q_modal, {
                    'w_velocity_norm': norm(w_aero_velocity),
                    'w_slope_norm': norm(w_aero_slope),
                    'Z_force_modal_norm': norm(Z_force_modal)
                }
            
            return Q_modal
        
        return Qg_func


    def print_diagnostics(self):
        """Print detailed diagnostics"""
        print("\n" + "="*70)
        print("DETAILED DIAGNOSTICS")
        print("="*70)
        print(f"Total iterations logged: {len(self.iteration_log)}")
        print(f"Mode switch warnings: {self.mode_switch_warnings}")
        print(f"Convergence issues: {self.convergence_issues}")
        print(f"Extrapolation failures: {self.extrapolation_failures}")
        
        if len(self.iteration_log) > 0:
            # Analyze iteration counts per mode
            from collections import Counter
            
            mode_iters = Counter()
            for entry in self.iteration_log:
                mode_iters[entry['mode']] += 1
            
            print(f"\nIterations per mode:")
            for mode in sorted(mode_iters.keys()):
                print(f"  Mode {mode}: {mode_iters[mode]} iterations")
            
            # Find problematic velocities
            vel_iters = Counter(entry['velocity'] for entry in self.iteration_log)
            problematic = [(v, count) for v, count in vel_iters.items() if count > 100]
            
            if problematic:
                print(f"\nProblematic velocities (>100 iterations):")
                for v, count in sorted(problematic)[:10]:
                    print(f"  V={v:.2f} m/s: {count} iterations")
        
        print("="*70 + "\n")