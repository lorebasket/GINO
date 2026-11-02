import numpy as np
from numpy.linalg import norm, inv
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from numpy.linalg import solve as lin_solve
import warnings

# Import utilities
from . import mode_matching
from .hydro_system import _interp_modal_matrix, _build_augmented_matrix_with_hydro
from . import empirical_fluid_damping


def solve_augmented_pk_with_capytaine(
    *,
    V,
    modes,
    omega_n,
    M_base,
    C_base,
    K_eff,
    Blag,
    blag,
    omega_h,
    added_mass_h,
    added_damping_h,
    rho,
    prev_freqs=None,
    prev_vecs=None,
    prev_eigs=None,
    k_guess=1e-3,
    k0_prev=None,
    fXK0=0.618,
    freq_margin=0.15,
    w_freq=0.8,
    w_mac=0.6,
    max_iter=80,
    tol=1e-3,
    relax=0.6,
    dimensionless_vgvf_results=False,
    resting_fluid_analysis=False,
    skip_inter_mode_on_perfect_mac=False,
    config=None,
):
    """
    Solve frequency-dependent augmented aeroelastic system with PK fixed-point iteration.
    """
    Nm = M_base.shape[0]
    N = len(Blag)
    q_dyn = 0.5 * rho * V ** 2

    g_out = np.full(len(modes), np.nan, dtype=float)
    f_out = np.full(len(modes), np.nan, dtype=float)
    vecs_out = np.zeros(((2 + N) * Nm, len(modes)), dtype=complex)
    eigs_out = np.full(len(modes), np.nan + 0j, dtype=complex)
    k_out = np.full(len(modes), np.nan, dtype=float)
    converged_out = np.zeros(len(modes), dtype=bool)

    last_converged_omegas = []
    if prev_freqs is not None:
        for i_mode, mode in enumerate(modes):
            if i_mode < len(prev_freqs) and np.isfinite(prev_freqs[i_mode]):
                last_converged_omegas.append(float(prev_freqs[i_mode]))
            else:
                last_converged_omegas.append(float(omega_n[mode]))
    else:
        last_converged_omegas = [float(omega_n[mode]) for mode in modes]

    converged_omegas_at_V = []
    converged_vecs_at_V = []

    for i_mode, mode in enumerate(modes):
        print(f"Solving mode {mode}...")
        if i_mode == 0:
            if k0_prev is not None and np.isfinite(k0_prev):
                k = float(max(k0_prev, 1e-6))
            elif prev_freqs is not None and i_mode < len(prev_freqs) and np.isfinite(prev_freqs[i_mode]):
                k = float(max(prev_freqs[i_mode] / V, 1e-6))
            else:
                k = float(max(k_guess, 1e-6))
        else:
            if prev_freqs is not None and i_mode < len(prev_freqs) and np.isfinite(prev_freqs[i_mode]):
                omega_s_prev = float(prev_freqs[i_mode])
            else:
                omega_s_prev = float(omega_n[mode])
            k_base = max(omega_s_prev / V, 1e-6)
            if len(converged_omegas_at_V) > 0 and np.isfinite(converged_omegas_at_V[i_mode - 1]):
                k_prev_mode = max(float(converged_omegas_at_V[i_mode - 1]) / V, 1e-6)
                k = max(k_base + fXK0 * (k_prev_mode - k_base), 1e-6)
            else:
                k = k_base

        p_sel = None
        u_sel = None
        prev_iter_vec = None
        converged = False
        mode_switch_warnings_ref = [0]

        for j in range(max_iter):
            omega_iter = float(max(k * V, 1e-6))
            A_add = _interp_modal_matrix(omega_h, added_mass_h, omega_iter)
            B_add = _interp_modal_matrix(omega_h, added_damping_h, omega_iter)

            
            M_eff = M_base + A_add
            C_eff = C_base + B_add
            if config is not None:
                C_emp = empirical_fluid_damping.modal_empirical_C(
                    config, V, M_eff, omega_ref=omega_iter
                )
                if np.any(C_emp != 0.0):
                    C_eff = C_eff + C_emp

            A_aug = _build_augmented_matrix_with_hydro(
                M_eff=M_eff,
                C_eff=C_eff,
                K_eff=K_eff,
                q_dyn=q_dyn,
                Blag=Blag,
                blag=blag,
                V=V,
                Nm=Nm,
                N=N,
            )

            vals, vecs = eig(A_aug)
            pos = np.where(np.imag(vals) > 1e-8)[0]
            if len(pos) == 0:
                break
            vals = vals[pos]
            vecs = vecs[:, pos]
            omegas = np.abs(np.imag(vals))
            order = np.argsort(omegas)
            vals = vals[order]
            vecs = vecs[:, order]
            omegas = omegas[order]

            prev_step_vec = None
            if prev_vecs is not None and prev_vecs.shape[1] > i_mode:
                prev_step_vec = prev_vecs[:, i_mode]
            p_target = None
            if prev_eigs is not None and i_mode < len(prev_eigs) and np.isfinite(np.abs(prev_eigs[i_mode])):
                p_target = prev_eigs[i_mode]

            p_sel, u_sel, _ = mode_matching.mode_matching_paper_compliant(
                vals=vals,
                vecs=vecs,
                s=mode,
                j=j,
                last_converged_omegas=last_converged_omegas,
                converged_omegas_at_V=converged_omegas_at_V,
                converged_vecs_at_V=converged_vecs_at_V,
                freq_margin=freq_margin,
                w_freq=w_freq,
                w_mac=w_mac,
                omega_n=omega_n,
                mode_switch_warnings_ref=mode_switch_warnings_ref,
                prev_step_vec=prev_step_vec,
                prev_iter_vec=prev_iter_vec,
                p_target=p_target,
                skip_inter_mode_on_perfect_mac=skip_inter_mode_on_perfect_mac,
            )
            if p_sel is None:
                break

            prev_iter_vec = u_sel
            omega_new = float(np.abs(np.imag(p_sel)))
            k_target = max(omega_new / V, 1e-6)
            k_new = max(k + relax * (k_target - k), 1e-6)
            if abs(k_new - k) < tol:
                k = k_new
                converged = True
                break
            k = k_new

        print(f"Mode {mode} converged in {j} iterations with k = {k:.4f}")

        
        if dimensionless_vgvf_results:
            if p_sel is not None:
                eigs_out[i_mode] = p_sel
                vecs_out[:, i_mode] = u_sel
                f_out[i_mode] = float(np.abs(np.imag(p_sel)))
                g_out[i_mode] = float(np.real(p_sel)/f_out[i_mode])
                k_out[i_mode] = float(k)
                converged_out[i_mode] = converged
                converged_omegas_at_V.append(float(np.abs(np.imag(p_sel))))
                converged_vecs_at_V.append(u_sel)
                
                print(f"        f_out = {f_out[i_mode]}")
                print(f"        g_out = {g_out[i_mode]}")

            else:
                converged_omegas_at_V.append(np.nan)
                converged_vecs_at_V.append(None)
        
        else:
            if p_sel is not None:
                eigs_out[i_mode] = p_sel
                vecs_out[:, i_mode] = u_sel
                f_out[i_mode] = float(np.abs(np.imag(p_sel)))
                #if resting_fluid_analysis:

                g_out[i_mode] = float(np.real(p_sel))
                k_out[i_mode] = float(k)
                converged_out[i_mode] = converged
                converged_omegas_at_V.append(float(np.abs(np.imag(p_sel))))
                converged_vecs_at_V.append(u_sel)

                print(f"        f_out = {f_out[i_mode]}")
                print(f"        g_out = {g_out[i_mode]}")

            else:
                converged_omegas_at_V.append(np.nan)
                converged_vecs_at_V.append(None)

    return g_out, f_out, vecs_out, eigs_out, k_out, converged_out