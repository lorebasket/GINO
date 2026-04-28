    #f = np.array([30.0, 40.0, 60.0])

    #f0 = [0.05]
    #f1 = np.arange(0.1, 1.1, 0.1)
    #f2 = np.arange(1.5, 10.1, 0.5)
    #f3 = np.arange(11, 24, 1)
    #f4 = np.arange(25, 200, 5)
    #f = np.concatenate((f0, f1, f2, f3, f4))
    #f = np.arange(0.5, 0.7, 0.1)

        #V1 = np.arange(3, 49, 1)
    #V2 = np.arange(50, 99, 10)
    #V3 = np.arange(100, 200, 50)
    #V = np.concatenate((V1, V2, V3))
    #V = [0.1, 10, 20, 50, 80] # inflow velocity [m/s]

# === After you build K_global, M_global, and BEFORE modal_analysis ===
import numpy as np
dof_per_node = 6
total_dof = K_global.shape[0]
# same "cantilever" BC the analysis uses: fix first node's 6 DOFs
constrained_dofs = list(range(dof_per_node))
# build the "free DOF" mask like analysis.modal_analysis does
all_dofs = np.arange(total_dof, dtype=int)
mask = np.ones(total_dof, dtype=bool)
mask[np.array(constrained_dofs, dtype=int)] = False
free = all_dofs[mask]
# mapping of component index -> name, matching your element order
dof_names = ['ux', 'uy', 'uz', 'rx', 'ry', 'rz']
def idx_of(comp):
    comp_id = dof_names.index(comp)
    return [i for i in free if (i % dof_per_node) == comp_id]
Uy = idx_of('uy')
Rz = idx_of('rz')
Uz = idx_of('uz')
Ry = idx_of('ry')
print("[BC check] counts:",
      "ux", len(idx_of('ux')),
      "uy", len(Uy),
      "uz", len(Uz),
      "rx", len(idx_of('rx')),
      "ry", len(Ry),
      "rz", len(Rz))
# reduce K the same way modal_analysis does (sym + free/free)
K_ff = 0.5 * (K_global + K_global.T)
K_ff = K_ff[np.ix_(free, free)]
# extract the assembled plane blocks
Uy_ff = [i for i,_i in enumerate(free) if (free[i] % dof_per_node) == dof_names.index('uy')]
Rz_ff = [i for i,_i in enumerate(free) if (free[i] % dof_per_node) == dof_names.index('rz')]
Uz_ff = [i for i,_i in enumerate(free) if (free[i] % dof_per_node) == dof_names.index('uz')]
Ry_ff = [i for i,_i in enumerate(free) if (free[i] % dof_per_node) == dof_names.index('ry')]
n_uyrz = np.linalg.norm(K_ff[np.ix_(Uy_ff, Rz_ff)])  # strong plane
n_uzry = np.linalg.norm(K_ff[np.ix_(Uz_ff, Ry_ff)])  # weak plane
print(f"||K_ff[uy,rz]|| = {n_uyrz:.3e}   ||K_ff[uz,ry]|| = {n_uzry:.3e}")

    #print("\nPlotting undeformed shape...")
    #visualization.plot_undeformed_shape(nodes)
    
    ## Plot deformed shape
    #print("\nPlotting deformed shape...")
    #visualization.plot_deformed_shape(nodes, displacements, section_breaks)
    
    ## Calculate theoretical tip displacement (Euler-Bernoulli)
    #tip_force = 100  # Magnitude of force [N]
    #L = beam_length
    #I = Iyy
    #tip_theoretical_EB = tip_force * L**3 / (3 * E * I)

    ## Calculate theoretical tip displacement (Timoshenko)
    #G = E / (2 * (1 + 0.3))  # Shear modulus
    #A = b * h - ((b - 2 * s) * (h - 2 * s))  # Area
    #tip_theoretical_T = tip_theoretical_EB * (1 + (E * I) / (G * A * L**2))

    ## Print comparison
    #print(f"Theoretical tip displacement (Euler-Bernoulli): {tip_theoretical_EB:.4f} mm")
    #print(f"Theoretical tip displacement (Timoshenko): {tip_theoretical_T:.4f} mm")
    #print(f"FEA tip displacement: {abs(tip_disp):.6f} mm")


        ## ==== MODULE FOR BLADE ==== ##
        #sections_props = ps.parse_section_props_csv(sections_props_csv)
        #sc_x = 0.5 - sections_props['SC_X'][0]
        #sc_y = 0 #sections_props['SC_Y'][0]
        #
        #contour_coords = contourn.main() # contour section coords
        #translated_contour_coords = contourn.translate_sections(contour_coords, sc_x, sc_y)
        #rotated_contour_coords = contourn.rotate_sections(translated_contour_coords, attack_angle[0])
        #
        #results = blade_fluid_main.main(
        #    blade_name, csv_filename, output_dir_ac,
        #    beam_model, chord_tip, chord_root, nspan, nchord,
        #    attack_angle, V, f, rotated_contour_coords)

    # Plot mode shapes
    #print("\nPlotting mode shapes...")
    #
    #nodes_arr = np.array([n["position"] for n in nodes], dtype=float)
    #nodes_for_plot = nodes_arr[:, [1, 0, 2]]
    #
    #visualization.plot_mode_shapes(nodes_for_plot, dry_mode_shapes, frequencies, dof_per_node=6,
    #                           which_modes=range(1, min(10, dry_mode_shapes.shape[1])+1),
    #                           components=('uy','uz'),  # pick what you care about
    #                           amplify=None,            # auto scale
    #                           plot_3d=False)

def eig_to_vg_vf(lam):
    sig = np.real(lam)
    omg = np.abs(np.imag(lam))
    # avoid divide-by-zero at near-rigid roots
    eps = 1e-12
    g   = -sig / np.maximum(omg, eps)         # V-g style
    f   = omg / (2*np.pi)                     # Hz
    
    return g, f


def modal_analysis(K_global, M_global, total_dof, num_modes=6, C_global=None,
                  dof_per_node=6, constrained_dofs=None, verbose=True):
    
    # Cantilever beam by default
    if constrained_dofs is None:
        constrained_dofs = list(range(dof_per_node))

    all_dofs = np.arange(total_dof, dtype=int)
    mask = np.ones(total_dof, dtype=bool)
    mask[np.array(constrained_dofs, dtype=int)] = False
    free = all_dofs[mask]

    # Reduce matrices
    K_ff = 0.5 * (K_global + K_global.T)
    K_ff = K_ff[np.ix_(free, free)]
    
    # Regularize mass matrix
    M_ff = M_global[np.ix_(free, free)]
    M_ff = 0.5 * (M_ff + M_ff.T)
    
    # For numerical stability, scale the problem (is it usefull??)
    scale = 1.0 / np.max(np.abs(K_ff))
    K_ff_scaled = K_ff * scale
    
    # Solve the eigenvalue problem
    try:
        # Try with subset_by_index first (faster)
        vals, vecs_ff = eigh(K_ff_scaled, M_ff, 
                            subset_by_index=(0, min(num_modes, len(free))-1))
    except Exception:
        # Fall back to standard eigh if subset fails
        if verbose:
            print("Falling back to standard eigh (slower)")
        vals, vecs_ff = eigh(K_ff_scaled, M_ff)
        # Sort by eigenvalue magnitude
        idx = np.argsort(np.abs(vals))[:num_modes]
        vals = vals[idx]
        vecs_ff = vecs_ff[:, idx]
    
    # Undo scaling
    vals = vals / scale
    
    # Sort by frequency (not eigenvalue magnitude)
    idx = np.argsort(np.abs(vals))
    vals = vals[idx]
    vecs_ff = vecs_ff[:, idx]
    
    # Calculate frequencies in Hz
    omegas = np.sqrt(np.abs(vals))
    freqs_hz = omegas / (2 * np.pi)
    
    # Lift to full DOF
    mode_shapes = np.zeros((total_dof, vecs_ff.shape[1]), dtype=float)
    mode_shapes[free, :] = vecs_ff
    
    # Normalize mode shapes to unit modal mass
    for i in range(vecs_ff.shape[1]):
        m_i = vecs_ff[:, i].T @ M_ff @ vecs_ff[:, i]
        if m_i > 0:
            vecs_ff[:, i] /= np.sqrt(m_i)
    
    return freqs_hz, mode_shapes, omegas, vecs_ff

def frf_modal(Mr, Cr, Kr, MW_r, CW_r, omega, Qr):
    """
    H(ω) = [-ω^2(Mr+MWr) + iω(Cr+CWr) + (Kr)]^{-1}
    x̂_r(ω) = H(ω) @ Qr(ω)
    """
    A = - (omega**2) * (Mr + MW_r) + 1j*omega*(Cr + CW_r) + Kr
    return np.linalg.solve(A, Qr)

            #if s > 0 and (ksm1_sm1 is not None):
#
            #    k_s1_s1 = float(ksm1_sm1)
#
            #    # 2) Lock aerodynamics at previous k_{s-1,s-1}
            #    Qhh_lock = Qg_func(k_s1_s1, b, V)
            #    Ac_lock  = self._Ac(V, rho, b, k_s1_s1, Qhh_lock)
            #    
            #    #
            #    omega_hint_s = omegasm1_sm1  #(k_init[s] * V) / b  # if k_init[s] > 0 else None
            #    p_lock, _, order_lock = _eig_solve_and_pick(Ac_lock, omega_hint=omega_hint_s)
            #    
            #    # pick the best candidate for mode s under locked aero:
            #    p_s_s1 = p_lock[order_lock[0]]
            #    omega_s_s1 = abs(np.imag(p_s_s1))
            #    k_s_s1 = (omega_s_s1 * b) / V
#
            #    # 3) Hybrid starting value:
            #    ks = k_s_s1 + fxk0 * (k_s1_s1 - k_s_s1)
            #    if verbose:
            #        print(f"[Hybrid] mode {s}: ks-1,s-1={k_s1_s1:.6g}, ks,s-1={k_s_s1:.6g} -> ks={ks:.6g}")


#
    #def _lock_eigenroot_in_pk_iter(self, eigvals, eigvecs, s, omega_struct, prev_mode_p=None,
    #                               freq_margin=0.05):
    #    #(lock eigenroots)
    #    # wn can either increase or decrease
#
    #    # sort by frequency
    #    ω = np.abs(np.imag(eigvals))
    #    order = np.argsort(ω)
    #    eigvals, eigvecs, ω = eigvals[order], eigvecs[:, order], ω[order]
#
    #    # Step 1: neighbor-based bounds (±5%) around structural neighbors
    #    lb = 0.0 if s == 0 else (1 - freq_margin) * omega_struct[s-1] # lower boundary
    #    ub = np.inf if s >= len(omega_struct) - 1 else (1 + freq_margin) * omega_struct[s+1] # upper boundary
#
    #    # Step 2: candidates
    #    candidates = np.where((ω >= lb) & (ω <= ub))[0] # candidates
#
    #    if candidates.size and prev_mode_p is not None:
    #        # among bounded candidates, pick the one closest to previous-mode converged p
    #        j = candidates[np.argmin(np.abs(eigvals[candidates] - prev_mode_p))]
    #        return eigvals[j], eigvecs[:, j], j
#
    #    if candidates.size:
    #        # still bounded; choose the closest by frequency to ω_struct[s]
    #        target = omega_struct[s]
    #        j = candidates[np.argmin(np.abs(ω[candidates] - target))]
    #        return eigvals[j], eigvecs[:, j], j
#
    #    # Step 3: fallback: pick by increasing frequency order
    #    j = min(s, len(ω) - 1)
#
    #    return eigvals[j], eigvecs[:, j], j

#    
#    
#    def solve_at_velocity(self, V, k0, rho, b, Qg_func, modes, max_iter=100, tol=1e-6, relax=0.618, fxk0=0.65, verbose=True):
#
#        def _eig_solve_and_pick(Ac, omega_hint=None):
#
#            vals, vecs = eig(Ac)
#
#            # p are the eigenvalues (σ + i ω)
#            p = vals
#            omega = np.abs(np.imag(p))
#
#            if omega_hint is None:
#                order = np.argsort(omega)
#            else:
#                order = np.argsort(np.abs(omega - omega_hint)) # variation ordering
#
#            return p, vecs, order
#
#        results = []
#        
#        k_init = np.array(k0, dtype=float)              # copy to avoid mutating caller's array
#
#        mode_indices = modes #list(range(modes))
#
#        ksm1_sm1 = None
#        omegasm1_sm1 = None
#
#        for s in mode_indices:
#            
#            ks = float(k_init[s])
#
#            converged = False
#            p_star = None
#            it_done = 0
#
#            if verbose:
#                print(f"Mode: {s}, k0_in: {ks:.6g}, Speed: {V}")
#
#            # --- PK iterations for mode s at speed V ---
#            for it in range(1, max_iter + 1):
#                if verbose:
#                    print(f"  Iteration {it}")
#
#                # Evaluate aero at current k_s
#                Qhh = Qg_func(ks, b, V)  # pass non-dimensional k
#                Ac  = self._Ac(V, rho, b, ks, Qhh)
#
#                # Solve and pick eigenvalue closest to previous omega
#                omega_hint = (ks * V) / b           # omega coming from the iteraion before
#                p_all, _, order = _eig_solve_and_pick(Ac, omega_hint=omega_hint)
#                p_sel = p_all[order[0]]
#
#                sigma = np.real(p_sel)
#                omega = abs(np.imag(p_sel))
#                k = omega * b / V
#
#                # Deferred-correction / under-relaxed update for k
#                k_new = ks + relax * (k - ks)       # (1.0 - relax) * ks + relax * k
#
#                # Relative convergence on k
#                if abs(k_new - ks) <= tol * max(1.0, abs(ks)):
#                    converged = True
#                    p_star = p_sel
#                    it_done = it
#                    ks = k_new
#                    if verbose:
#                        print(f"    CONVERGED at it{it}: k={ks:.6g}, ω={omega:.6g}, σ={sigma:.6g}")
#                    break
#
#                # continue iterations
#                ks = k_new
#                p_star = p_sel
#                it_done = it
#
#                print(f"k={ks:.6g}, ω={omega:.6g}, σ={sigma:.6g}")
#
#            # store results for this mode
#            if p_star is None:
#                # fall back to last computed p in case of no iterations
#                p_star = p_sel
#
#            sigma = np.real(p_star)
#            omega = abs(np.imag(p_star))
#            results.append({
#                "mode": s,
#                "k": float(ks),
#                "omega": float(omega),
#                "sigma": float(sigma),
#                "p": complex(p_star),
#                "converged": bool(converged),
#                "it": int(it_done),
#            })
#
#            # cache for hybrid seeding of the next mode
#            ksm1_sm1 = float(ks)
#            omegasm1_sm1 = float(omega)
#
#        return results

comparison = True 

if comparison == True:
    print("Z shape:", Z.shape)
    print("Mean/max Z:", np.mean(np.abs(Z)), np.max(np.abs(Z)))

    print("K_hat norm:", np.linalg.norm(K_hat))
    print("M_hat norm:", np.linalg.norm(M_hat))

    # --- Print key model parameters and compare to reference ---
    print("\n=== Goland Model Parameters and Modal Comparison ===")
    print(f"Span: {beam_length:.3f} m (Reference: 6.096 m)")
    print(f"Chord: {chord:.3f} m (Reference: 1.8288 m)")
    print(f"Mass per unit length: {mu:.3f} kg/m (Reference: 35.71 kg/m)")
    print(f"EI (Bending): {EIyy:.2e} N*m^2 (Reference: 9.77e6 N*m^2)")
    print(f"GJ (Torsion): {GJ:.2e} N*m^2 (Reference: 0.99e6 N*m^2)")
    print(f"Ixx (Torsion): {i11:.3f} kg*m (Reference: 8.64 kg*m)")
    print(f"xEA/chord: {xea/chord:.2f} (Reference: 0.33)")
    print(f"xCM/chord: {xcm/chord:.2f} (Reference: 0.43)")

    print(f"Number of panels: {nspan * nchord}")
    print(f"Number of panels along span: {nspan}")
    print(f"Number of panels along chord: {nchord}")

    print("\nFirst 5 natural frequencies (Hz):")
    for i, freq in enumerate(freqs[:5]):
        print(f"  Mode {i+1}: {freq:.4f} Hz (Reference: see BYU example)")

    print("\nRayleigh damping coefficients:")
    print(f"  alpha: {alpha:.3e}")
    print(f"  beta:  {beta:.3e}")

    print("\nNorms of key matrices (should be in same ballpark as reference):")
    print(f"K_hat shape: {K_hat.shape}")
    print(f"M_hat shape: {M_hat.shape}")

    print(f"  M_ff norm: {np.linalg.norm(Mff)}")
    print(f"  K_ff norm: {np.linalg.norm(Kff)}")

    print(f"dry_vectors shape: {dry_vectors.shape}")
    print(f"dry_vectors: {dry_vectors}")

    print(f"dry_values shape: {dry_values.shape}")
    print(f"dry_values: {dry_values}")

    print(f"  M_hat norm: {np.linalg.norm(M_hat)}")
    print(f"  K_hat norm: {np.linalg.norm(K_hat)}")

    # For a typical speed and reduced frequency:
    V_test = 100.0  # m/s
    c_air = 343.0
    b = chord / 2
    k_test = 0.2
    Ma_test = V_test / c_air
    Qjj_test = DLM.calc_Qjj(aerogrid, Ma_test, k_test)
    Qpp_test = Z.T @ Qjj_test @ Z
    Q_modal_test = dry_vectors.T @ Qpp_test @ dry_vectors
    print(f"\nQ_modal norm at V={V_test} m/s, k={k_test}: {np.linalg.norm(Q_modal_test)}")
    
    span = beam_length
    chord = chord
    num_panels = nspan * nchord
    total_area = span * chord
    panel_area = total_area / num_panels
    print(f"Total wing area: {total_area:.3f} m^2")
    print(f"Panel area: {panel_area:.5f} m^2 (should be physical)")

#def precompute_qjj_grid(FSI_path, aerogrid, k_list, Ma_list, out_file=None, verbose=True):
#    import numpy as np
#    import sys
#
#    sys.path.append(FSI_path + '/PanelAero')
#    
#    from panelaero_utl import DLM
#
#    k_list = np.asarray(k_list)
#    Ma_list = np.asarray(Ma_list)
#    Qjj_shape = None
#
#    # Arrays to hold real and imaginary parts
#    Qjj_real = None
#    Qjj_imag = None
#
#    for i, k in enumerate(k_list):
#        for j, Ma in enumerate(Ma_list):
#            if verbose:
#                print(f"Computing Qjj for k={k:.4g}, Ma={Ma:.4g} ({i+1}/{len(k_list)}, {j+1}/{len(Ma_list)})")
#            Qjj = DLM.calc_Qjj(aerogrid, Ma, k)
#
#            if Qjj_real is None:
#                Qjj_shape = Qjj.shape
#                Qjj_real = np.zeros((len(k_list), len(Ma_list), *Qjj_shape))
#                Qjj_imag = np.zeros((len(k_list), len(Ma_list), *Qjj_shape))
#            
#            Qjj_real[i, j, :, :] = np.real(Qjj)
#            Qjj_imag[i, j, :, :] = np.imag(Qjj)
#
#    if out_file:
#        np.savez_compressed(
#            out_file, k_list=k_list, Ma_list=Ma_list,
#            Qjj_real=Qjj_real, Qjj_imag=Qjj_imag
#        )
#        if verbose:
#            print(f"Saved Qjj real/imag grid to {out_file}")
#
#    return k_list, Ma_list, Qjj_real, Qjj_imag
#
#def load_qjj_grid_for_interp(filename):
#    from scipy.interpolate import RegularGridInterpolator
#    import numpy as np
#    from pathlib import Path
#    
#    data = np.load(filename)
#    k_list = data['k_list']
#    Ma_list = data['Ma_list']
#    Qjj_real = data['Qjj_real']
#    Qjj_imag = data['Qjj_imag']
#
#    # Create interpolation objects, one for each panel pair (i, j)
#    n_i, n_j = Qjj_real.shape[2], Qjj_real.shape[3]
#    interp_real = [[RegularGridInterpolator((k_list, Ma_list), Qjj_real[..., i, j], bounds_error=False, fill_value=None)
#                    for j in range(n_j)] for i in range(n_i)]
#    interp_imag = [[RegularGridInterpolator((k_list, Ma_list), Qjj_imag[..., i, j], bounds_error=False, fill_value=None)
#                    for j in range(n_j)] for i in range(n_i)]
#    return k_list, Ma_list, Qjj_real, Qjj_imag, interp_real, interp_imag
#
#def interpolate_qjj(interp_real, interp_imag, k, Ma):
#    import numpy as np
#
#    n_i, n_j = len(interp_real), len(interp_real[0])
#    Qjj = np.zeros((n_i, n_j), dtype=np.complex128)
#    for i in range(n_i):
#        for j in range(n_j):
#            Qjj[i, j] = interp_real[i][j]([[k, Ma]]) + 1j * interp_imag[i][j]([[k, Ma]])
#    return Qjj
#
#def _tile_path(base_dir, alpha_r, n_span, n_chord, k, Ma):
#    base_dir = Path(base_dir)
#    base_dir.mkdir(parents=True, exist_ok=True)
#    # nomi compatti e ordinabili
#    return base_dir / f"Qjj_a{alpha_r:.4f}_ns{n_span}_nc{n_chord}_k{float(k):.6f}_Ma{float(Ma):.6f}.npy"
#
#def precompute_qjj_tiles(FSI_path, aerogrid, k_list, Ma_list, out_dir,
#                         alpha_r=0.0, n_span=None, n_chord=None,
#                         dtype=np.complex64, verbose=True):
#    """
#    Calcola Qjj per ogni (k,Ma) e salva UN file .npy per tile.
#    Usa complex64 per dimezzare lo spazio su disco.
#    """
#    import sys
#    sys.path.append(FSI_path + '/PanelAero')
#    from panelaero_utl import DLM
#
#    k_list = np.asarray(k_list, float)
#    Ma_list = np.asarray(Ma_list, float)
#
#    for i, k in enumerate(k_list):
#        for j, Ma in enumerate(Ma_list):
#            if verbose:
#                print(f"[{i+1}/{len(k_list)}] k={k:.4g}, Ma={Ma:.4g} -> computing")
#            Qjj = DLM.calc_Qjj(aerogrid, Ma, k).astype(dtype, copy=False)
#            fpath = _tile_path(out_dir, alpha_r, n_span, n_chord, k, Ma)
#            np.save(fpath, Qjj)
#            if verbose:
#                print(f"  saved: {fpath}")
#
#    # Salviamo anche la “meta” piccola
#    meta = dict(k_list=k_list, Ma_list=Ma_list,
#                alpha=float(alpha_r), n_span=int(n_span or -1), n_chord=int(n_chord or -1),
#                dtype=str(np.dtype(dtype)))
#    np.save(Path(out_dir)/"grid_meta.npy", meta, allow_pickle=True)
#    return meta






#def _find_bracketing(vals, x):
#    """Restituisce (i0,i1,t) tali che x è tra vals[i0] e vals[i1] e t in [0,1]."""
#    vals = np.asarray(vals, float)
#    if x <= vals[0]:  # clamp
#        return 0, 0, 0.0
#    if x >= vals[-1]:
#        return len(vals)-1, len(vals)-1, 0.0
#    i1 = np.searchsorted(vals, x)
#    i0 = i1 - 1
#    v0, v1 = vals[i0], vals[i1]
#    t = (x - v0) / (v1 - v0)
#    return i0, i1, float(t)
#
#def _load_tile(base_dir, alpha_r, n_span, n_chord, k, Ma):
#    path = _tile_path(base_dir, alpha_r, n_span, n_chord, k, Ma)
#    return np.load(path)
#
#def interpolate_qjj_tiled(base_dir, alpha_r, n_span, n_chord,
#                          k_list, Ma_list, k, Ma):
#    """
#    Interpolazione bilineare caricando solo 4 file (k0,k1) x (Ma0,Ma1).
#    Se k o Ma sono fuori griglia, fa clamp al bordo (t=0).
#    """
#    i0, i1, tk = _find_bracketing(k_list, k)
#    j0, j1, tm = _find_bracketing(Ma_list, Ma)
#
#    k0, k1 = k_list[i0], k_list[i1]
#    Ma0, Ma1 = Ma_list[j0], Ma_list[j1]
#
#    Q00 = _load_tile(base_dir, alpha_r, n_span, n_chord, k0, Ma0)
#    if i0 == i1 and j0 == j1:
#        return Q00  # esatto su nodo di griglia
#
#    # Carica solo ciò che serve
#    Q10 = Q00 if i0 == i1 else _load_tile(base_dir, alpha_r, n_span, n_chord, k1, Ma0)
#    Q01 = Q00 if j0 == j1 else _load_tile(base_dir, alpha_r, n_span, n_chord, k0, Ma1)
#    if i0 == i1 and j0 == j1:
#        Q11 = Q00
#    elif i0 == i1:
#        Q11 = Q01
#    elif j0 == j1:
#        Q11 = Q10
#    else:
#        Q11 = _load_tile(base_dir, alpha_r, n_span, n_chord, k1, Ma1)
#
#    # Interpolazione bilineare su matrici
#    # (1-tk)*(1-tm)*Q00 + tk*(1-tm)*Q10 + (1-tk)*tm*Q01 + tk*tm*Q11
#    return ((1-tk)*(1-tm)*Q00 +
#            tk*(1-tm)*Q10 +
#            (1-tk)*tm*Q01 +
#            tk*tm*Q11)









## === from pk solver === ##
p_sel, u_sel, j_sel = self._lock_eigenroot_in_pk_iter(
    vals, vecs, s, converged_omegas_at_V, prev_mode_p, prev_iter_p_s, freq_margin
)

    def _lock_eigenroot_in_pk_iter(self, eigvals, eigvecs, s, converged_omegas_at_V,
                                   prev_mode_p=None, prev_iter_p_s=None,
                                   freq_margin=None, omega_abs_floor=1e-3):
        import numpy as np

        # Sort by frequency ω = |Im(p)|
        omega = np.abs(np.imag(eigvals))
        order = np.argsort(omega)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        omega = omega[order]
        n = omega.size

        def finite(z): 
            return (z is not None) and np.isfinite(np.real(z)) and np.isfinite(np.imag(z))

        # Target ω for this mode (structural frequency)
        if hasattr(self, "omega_n") and s < len(self.omega_n):
            target_struct = float(self.omega_n[s])
        else:
            target_struct = float(omega[min(s, n-1)])

        # Use converged value if available
        if 0 <= s < len(converged_omegas_at_V) and np.isfinite(converged_omegas_at_V[s]):
            target = float(converged_omegas_at_V[s])
        else:
            target = target_struct

        # ---------- BRACKETING ----------
        # Lower bound: from previous mode
        if s > 0:
            if s-1 < len(converged_omegas_at_V) and np.isfinite(converged_omegas_at_V[s-1]):
                lb = (1.0 - freq_margin) * float(converged_omegas_at_V[s-1])
            elif finite(prev_mode_p):
                lb = (1.0 - freq_margin) * abs(np.imag(prev_mode_p))
            elif hasattr(self, "omega_n") and s-1 < len(self.omega_n):
                lb = (1.0 - freq_margin) * float(self.omega_n[s-1])
            else:
                lb = 0.0
        else:
            lb = 0.0

        # Upper bound: from next mode OR structural spacing
        if s+1 < len(converged_omegas_at_V) and np.isfinite(converged_omegas_at_V[s+1]):
            ub = (1.0 + freq_margin) * float(converged_omegas_at_V[s+1])
        elif hasattr(self, "omega_n") and s+1 < len(self.omega_n):
            ub = (1.0 + freq_margin) * float(self.omega_n[s+1])
        else:
            # No next mode info: use structural spacing as guide
            # Stay within reasonable distance from structural frequency
            ub = (1.0 + 3*freq_margin) * target_struct

        # Safety checks
        if not np.isfinite(lb): lb = 0.0
        if not np.isfinite(ub) or ub <= lb:
            ub = max(2.0 * target_struct, (1.0 + 3*freq_margin) * target_struct)

        # Apply absolute floor
        lb = max(lb, omega_abs_floor)

        # ---------- CANDIDATE SELECTION ----------
        mask = (omega >= lb) & (omega <= ub)
        candidates = np.where(mask)[0]

        # If no candidates, expand search gradually
        if candidates.size == 0:
            for expansion in [1.5, 2.0, 3.0]:
                lb_exp = max(omega_abs_floor, lb / expansion)
                ub_exp = ub * expansion
                mask = (omega >= lb_exp) & (omega <= ub_exp)
                candidates = np.where(mask)[0]
                if candidates.size > 0:
                    break

        # Last resort: pick s-th by frequency
        if candidates.size == 0:
            j = min(s, n-1)
            return eigvals[j], eigvecs[:, j], j

        # ---------- SELECTION PRIORITY ----------
        # 1. Continuity within iteration (prev_iter_p_s)
        if finite(prev_iter_p_s):
            j = candidates[np.argmin(np.abs(eigvals[candidates] - prev_iter_p_s))]
            return eigvals[j], eigvecs[:, j], j

        # 2. Closest to target frequency (converged or structural)
        j = candidates[np.argmin(np.abs(omega[candidates] - target))]

        # 3. Sanity check: don't jump too far from structural frequency
        selected_omega = omega[j]
        max_deviation = 2.0  # Allow 2x deviation from structural frequency

        if abs(selected_omega - target_struct) > max_deviation * target_struct:
            # Selected frequency is too far - pick closest to structural instead
            j_struct = candidates[np.argmin(np.abs(omega[candidates] - target_struct))]
            print(f"    WARNING: Selected omega {selected_omega:.2f} too far from structural {target_struct:.2f}, using {omega[j_struct]:.2f} instead")
            j = j_struct

        return eigvals[j], eigvecs[:, j], j



##### ===== Qjj precomputation and interpolation ===== #####

def precompute_qjj_slices(FSI_path, aerogrid, omega_list, V_list, out_dir, c_sound,
                          dtype=np.float32, verbose=True):

    import sys
    sys.path.append(FSI_path + '/PanelAero')
    from panelaero_utl import DLM   # usa la tua DLM

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    omega_list = np.asarray(omega_list, dtype=float)
    V_list = np.asarray(V_list, dtype=float)

    Q_shape = None

    for i_omega, omega in enumerate(omega_list):
        for i_V, V in enumerate(V_list):
            if verbose:
                print(f"[Qjj] Computing omega[{i_omega}]={omega:.6g}, V[{i_V}]={V:.6g}")

            k_red = omega / V

            Ma = V / c_sound
            # === calcolo Qjj complessa per questo (k,Ma) ===
            Qjj = DLM.calc_Qjj(aerogrid, k_red, Ma)

            if Q_shape is None:
                Q_shape = Qjj.shape

            # Salvataggio separato Re/Im (tipicamente basta float32)
            fname_base = out_dir / f"omega_{i_omega:03d}__V_{i_V:04d}"
            np.save(f"{fname_base}_real.npy",
                    Qjj.real.astype(dtype), allow_pickle=False)
            np.save(f"{fname_base}_imag.npy",
                    Qjj.imag.astype(dtype), allow_pickle=False)

    # indice per trovare rapidamente file e info
    index = {
        "omega_list": omega_list.tolist(),
        "V_list": V_list.tolist(),
        "shape": list(Q_shape),
        "dtype": np.dtype(dtype).name,
        "n_omega": len(omega_list),
        "n_V": len(V_list),
        "pattern": "ma_{i_ma:03d}__k_{i_k:04d}_(real|imag).npy"
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    if verbose:
        print(f"[Qjj] Wrote index to {out_dir/'index.json'}")

def _load_index(root):
    root = Path(root)
    idx = json.loads((root / "index.json").read_text())
    return {
        "root": root,
        "omega_list": np.array(idx["omega_list"], dtype=float),
        "V_list": np.array(idx["V_list"], dtype=float),
        "shape": tuple(idx["shape"]),
        "dtype": np.float32 if idx["dtype"] == "float32" else np.float64,
        "n_omega": int(idx["n_omega"]),
        "n_V": int(idx["n_V"]),
    }

def _bracket(arr, x):
    """
    Restituisce (i0, i1, t) tali che arr[i0] <= x <= arr[i1] e
    t è il peso lineare in [0,1] (t=0 -> i0, t=1 -> i1).
    Se x è fuori range, clamp ai bordi e t=0.
    """
    if x <= arr[0]: return 0, 0, 0.0
    if x >= arr[-1]: return len(arr)-1, len(arr)-1, 0.0
    i1 = int(np.searchsorted(arr, x, side="right"))
    i0 = i1 - 1
    t = (x - arr[i0]) / (arr[i1] - arr[i0])
    return i0, i1, float(t)

def _load_slice(root, i_omega, i_V, part, mmap=True, dtype=np.float32):
    """
    part: 'real' oppure 'imag'
    """
    root = Path(root)
    suffix = "_real.npy" if part == "real" else "_imag.npy"
    path = root / f"omega_{i_omega:03d}__V_{i_V:04d}{suffix}"
    # mmap_mode='r' evita di caricare in RAM tutto subito
    return np.load(path, mmap_mode="r" if mmap else None).astype(dtype, copy=False)

def interp_qjj_from_disk(root, omega_query, V_query, mmap=True):
    """
    Interpola bilinearmente Qjj in (omega, V) caricando solo
    i 2x2 vicini (fino a 4 matrici) per Re e Im, poi ricompone la parte complessa.
    """
    meta = _load_index(root)
    omega_list, V_list = meta["omega_list"], meta["V_list"]
    dtype = meta["dtype"]

    # FIX: These lines were still using k_list and Ma_list
    i0, i1, t_omega = _bracket(omega_list, omega_query)
    j0, j1, t_V = _bracket(V_list, V_query)

    # Carica solo i vicini necessari
    R00 = _load_slice(meta["root"], i0, j0, "real", mmap, dtype)
    R10 = _load_slice(meta["root"], i1, j0, "real", mmap, dtype) if i1 != i0 else R00
    R01 = _load_slice(meta["root"], i0, j1, "real", mmap, dtype) if j1 != j0 else R00
    R11 = _load_slice(meta["root"], i1, j1, "real", mmap, dtype) if (i1 != i0 or j1 != j0) else R00

    I00 = _load_slice(meta["root"], i0, j0, "imag", mmap, dtype)
    I10 = _load_slice(meta["root"], i1, j0, "imag", mmap, dtype) if i1 != i0 else I00
    I01 = _load_slice(meta["root"], i0, j1, "imag", mmap, dtype) if j1 != j0 else I00
    I11 = _load_slice(meta["root"], i1, j1, "imag", mmap, dtype) if (i1 != i0 or j1 != j0) else I00

    # Interpolazione bilineare separata su Re e Im:
    # (1-t_omega)*( (1-t_V)*00 + t_V*01 ) + t_omega*( (1-t_V)*10 + t_V*11 )
    def bilinear(A00, A10, A01, A11):
        return (1-t_omega)*((1-t_V)*A00 + t_V*A01) + t_omega*((1-t_V)*A10 + t_V*A11)

    Qr = bilinear(R00, R10, R01, R11)
    Qi = bilinear(I00, I10, I01, I11)

    # Ricomponi complesso
    return Qr.astype(dtype, copy=False) + 1j * Qi.astype(dtype, copy=False)

def open_qjj_index(root):
    """Comodo se vuoi solo leggere k_list e Ma_list per sapere cosa c'è su disco."""
    meta = _load_index(root)
    return meta["omega_list"], meta["V_list"], meta["shape"], meta["dtype"]



# Also print the sum of forces for verification
print("\nSum of forces (should balance the applied load):")
print(f"  Fx: {reaction_forces[0]:.2f} N (should be ~0 for pure vertical load)")
print(f"  Fy: {reaction_forces[1]:.2f} N (axial force, should be ~0 for pure vertical load)")
print(f"  Fz: {reaction_forces[2]:.2f} N (should be {abs(load_magnitude_max):.2f} N to balance the applied load)")

# Print moments about the root
print("\nMoments about the root (right-hand rule):")
print(f"  Mx: {reaction_forces[3]:.2f} Nm (bending moment about x-axis - should be non-zero)")
print(f"  My: {reaction_forces[4]:.2f} Nm (torsion moment about y-axis - should be ~0 for pure vertical load)")
print(f"  Mz: {reaction_forces[5]:.2f} Nm (bending moment about z-axis - should be ~0 for pure vertical load)")

# Expected bending moment for a cantilever beam with tip load
expected_bending_moment = load_magnitude_max * beam_length  # Negative because load is downward
print(f"\nExpected bending moment at root (Mx): {expected_bending_moment:.2f} Nm")