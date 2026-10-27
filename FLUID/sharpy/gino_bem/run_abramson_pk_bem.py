#!/usr/bin/env python3
"""
Abramson 1965 — hybrid SHARPy UVLM + Capytaine BEM flutter with per-mode p-k.

Removes the UVLM apparent-mass terms (``--added-mass-removal``, GINO patch in
``sharpy/linear/src/linuvlm.py``) and injects frequency-dependent Capytaine
modal added mass A(omega) / radiation damping B(omega) into the linear beam
(``capy_bem_bridge``), iterating each tracked structural branch to frequency
consistency (p-k), mirroring COUPLING/hydroelastic_utl/pk_method.py.

  conda activate sharpy
  cd sharpy
  python gino_bem/run_abramson_pk_bem.py --coarse --dry-check
  python gino_bem/run_abramson_pk_bem.py --coarse --added-mass-removal gamma_dot
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.linalg as sclalg

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
_FSI_ROOT = _SHARPY_ROOT.parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))

from gino_bem.abramson1965_wing import Abramson1965  # noqa: E402
from gino_bem import run_abramson_water as R  # noqa: E402
from gino_bem.abramson_tracking_config import ABRAMSON_V_MIN_M_S  # noqa: E402
from gino_bem.rayleigh_damping import (  # noqa: E402
    ABRAMSON_RAYLEIGH_MODE_IDS,
    ABRAMSON_RAYLEIGH_ZETAS,
    install_rayleigh_structural_damping,
)
from gino_bem.capy_bem_bridge import (  # noqa: E402
    apply_bem_from_capy,
    capy_at_omega,
    check_capy_basis,
    default_capytaine_npz,
    load_capytaine_npz,
    velocity_sweep_with_bem,
)
from gino_bem.mode_alignment import (  # noqa: E402
    align_capy_to_sharpy,
    sharpy_modal_displacements,
    strip_added_mass_modal,
)
from gino_bem.track_structural_modes import beam_state_indices  # noqa: E402
from gino_bem.mac_track import mac  # noqa: E402
from gino_bem.contribution_extraction import extract_contributions_at_velocity  # noqa: E402
import sharpy.sharpy_main  # noqa: E402

KN = 0.514444
V_EXP_KN, F_EXP_HZ = 48.1, 17.5
V_GINO_KN = 48.75
GRIDS = {
    "coarse": (6, 12, 4),
    "default": (8, 16, 6),
    "fine": (12, 24, 8),
    # Medium grid: nspan=20 (span), nchord=10 (chord) → M chord, N span
    "n20c10": (10, 20, 6),
    # Rebalanced production grid (discretization study): 12 span × 16 chord
    "n12c16": (16, 12, 6),
}
BRANCH_NAMES = {0: "bending", 1: "torsion", 2: "mode3", 3: "mode4"}


# --------------------------------------------------------------------------- case build
def build_case(nm: int, *, grid: tuple[int, int, int], added_mass_removal: str,
               rayleigh: bool, zetas, v_max_kn: float, case_tag: str):
    M, N, Ms = grid
    route = _SHARPY_ROOT / "gino_bem"
    (route / "cases").mkdir(parents=True, exist_ok=True)
    out_root = route / "output"
    out_root.mkdir(exist_ok=True)
    ws = Abramson1965(
        M=M, N=N, Mstar_fact=Ms, u_inf=1.0, alpha=0.0, rho=997.0,
        route=str(route / "cases"), case_name=case_tag,
    )
    ws.clean_test_files()
    ws.update_derived_params()
    ws.set_default_config_dict()
    ws.generate_aero_file()
    ws.generate_fem_file()
    vel = R._abramson_velocity_list_m_s(v_max_m_s=R._kn_to_m_s(v_max_kn))
    R._configure_sharpy_case(
        ws, num_modes=nm, velocity_list=vel,
        v_min=float(vel.min()), v_max=float(vel.max()), n_vel=len(vel),
        out_root=out_root, rom_settings=None, u_inf=1.0, k_points=[0.0],
    )
    # our own p-k sweep replaces the built-in AsymptoticStability velocity loop
    ws.config["SHARPy"]["flow"] = [
        "BeamLoader", "AerogridLoader", "StaticCoupled", "Modal", "LinearAssembler",
    ]
    aero = ws.config["LinearAssembler"]["linear_system_settings"]["aero_settings"]
    aero["added_mass_removal"] = added_mass_removal
    ws.config.write()
    if rayleigh:
        install_rayleigh_structural_damping(
            mode_ids=ABRAMSON_RAYLEIGH_MODE_IDS, target_zetas=zetas,
        )
        print(f">>> Rayleigh damping: zeta={zetas} on modes {list(ABRAMSON_RAYLEIGH_MODE_IDS)}", flush=True)
    print(f">>> UVLM added_mass_removal = '{added_mass_removal}'  grid M,N,Ms = {grid}", flush=True)
    data = sharpy.sharpy_main.main(["", ws.route + ws.case_name + ".sharpy"])
    return data, vel


# --------------------------------------------------------------------------- p-k core
def _eig_at(ls, V: float, *, capy=None, T=None, omega: float | None = None, n_modes: int = 4,
            bem: bool = True, A_offset=None):
    """One assembly + eigendecomposition at velocity V with A(omega),B(omega) injected."""
    t_ref = float(ls.uvlm.sys.ScalingFacts["length"]) / float(V)
    if bem and capy is not None and omega is not None:
        apply_bem_from_capy(ls, capy, float(omega), n_modes=n_modes, T=T, t_ref=t_ref,
                            A_offset=A_offset)
    ss = ls.update(float(V))
    w, Vec = sclalg.eig(ss.A, right=True)
    lam = np.log(w) / (t_ref * ss.dt)
    return lam, Vec, ss


def _candidates(lam, Vec, beam_idx, *, part_floor=0.01, freq_window_hz=(1.0, 40.0)):
    freq = lam.imag / (2 * np.pi)
    vnorm = np.linalg.norm(Vec, axis=0)
    part = np.linalg.norm(Vec[beam_idx, :], axis=0) / np.maximum(vnorm, 1e-30)
    keep = [k for k in range(len(lam))
            if lam[k].imag > 1e-6
            and freq_window_hz[0] <= freq[k] <= freq_window_hz[1]
            and part[k] >= part_floor]
    return keep, part


def _select(lam, Vec, cand, beam_idx, mode_j, omega_target, ref_vec, nm, *,
            w_mac=0.7, w_freq=0.3, jump_frac=0.30):
    """Pick the eigenvalue matched to branch ``mode_j``. Returns (k, mac) or (None, 0)."""
    if not cand:
        return None, 0.0
    scored = []
    for k in cand:
        om = float(lam[k].imag)
        prox = float(np.exp(-abs(om - omega_target) / max(omega_target, 1e-30)))
        if ref_vec is not None:
            m = mac(ref_vec, Vec[beam_idx, k])
            # continuity guard: big frequency jump only allowed on near-perfect MAC
            if abs(om - omega_target) > jump_frac * omega_target and m < 0.9:
                continue
            scored.append((w_mac * m + w_freq * prox, m, k))
        else:
            # first velocity: prefer dominant displacement DOF (q block = first nm);
            # coupled wet modes may not have a dominant-DOF match (handled below)
            disp = np.abs(Vec[beam_idx[:nm], k])
            if int(np.argmax(disp)) != mode_j:
                continue
            scored.append((prox, prox, k))
    if not scored and ref_vec is None:
        # fallback: frequency-nearest candidate to the wet seed (strongly coupled
        # branches, e.g. torsion carrying dominant bending displacement content)
        k_sel = min(cand, key=lambda k: abs(float(lam[k].imag) - omega_target))
        prox = float(np.exp(-abs(float(lam[k_sel].imag) - omega_target) / max(omega_target, 1e-30)))
        return k_sel, prox
    if not scored:
        return None, 0.0
    scored.sort(reverse=True, key=lambda t: t[0])
    _, m_sel, k_sel = scored[0]
    return k_sel, float(m_sel)


def pk_solve_mode_at_velocity(ls, capy, T, V, mode_j, omega_seed, ref_vec, beam_idx, nm, *,
                              bem=True, A_offset=None, relax=0.618, tol_rel=1e-3, max_iter=25,
                              min_iter=2, jump_frac=0.30, part_floor=0.01,
                              freq_window_hz=(1.0, 40.0)):
    """Frequency-matched (p-k) solve of one structural branch at one velocity."""
    omega_j = float(omega_seed)
    n_iter_max = max_iter if bem else 1     # without BEM nothing depends on omega
    result = dict(V=float(V), mode=mode_j, converged=False, n_iter=0,
                  g=np.nan, omega=np.nan, freq_Hz=np.nan, mac=0.0, part=np.nan, vec=None,
                  lam_full=None, vec_beam_full=None, part_full=None)
    for it in range(n_iter_max):
        lam, Vec, ss = _eig_at(ls, V, capy=capy, T=T, omega=omega_j, n_modes=nm, bem=bem,
                               A_offset=A_offset)
        cand, part = _candidates(lam, Vec, beam_idx, part_floor=part_floor,
                                 freq_window_hz=freq_window_hz)
        k, m_sel = _select(lam, Vec, cand, beam_idx, mode_j, omega_j, ref_vec, nm,
                           jump_frac=jump_frac)
        result.update(n_iter=it + 1, lam_full=lam, vec_beam_full=Vec[beam_idx, :], part_full=part)
        if k is None:
            return result                    # branch lost at this velocity
        omega_new = float(lam[k].imag)
        d_rel = abs(omega_new - omega_j) / max(omega_j, 1e-30)
        omega_j = (1.0 - relax) * omega_j + relax * omega_new
        gamma = float(lam[k].real) / max(abs(omega_new), 1e-30)
        result.update(g=gamma, omega=omega_new, freq_Hz=omega_new / (2 * np.pi),
                      mac=m_sel, part=float(part[k]), vec=Vec[beam_idx, k].copy())
        if (d_rel < tol_rel and it + 1 >= min_iter) or not bem:
            result["converged"] = True
            return result
    warnings.warn(f"p-k: mode {mode_j} at V={V:.2f} m/s not converged after {n_iter_max} iters "
                  f"(last d_omega/omega={d_rel:.2e})", RuntimeWarning)
    return result


def pk_sweep_bem(ls, capy, T, v_list, *, nm, track_modes=(0, 1), bem=True, A_offset=None,
                 relax=0.618, tol_rel=1e-3, max_iter=25, part_floor=0.01,
                 freq_window_hz=(1.0, 40.0), step_report=None):
    """Outer velocity sweep, per-mode p-k. Returns (rows, spectrum-collectors)."""
    beam_idx = beam_state_indices(ls.ss, nm)

    # physical dry omegas from the NPZ source eigenvalues, mapped to SHARPy order via T
    lam_g = np.asarray(capy.get("source_eigenvalues", []), dtype=float).ravel()
    omega_dry = np.full(nm, np.nan)
    for j in range(min(nm, T.shape[1])):
        i = int(np.argmax(np.abs(T[:, j])))
        if i < len(lam_g):
            omega_dry[j] = np.sqrt(abs(lam_g[i]))
    if np.isnan(omega_dry[list(track_modes)]).any():
        raise RuntimeError("Cannot seed p-k: missing dry frequencies (NPZ lacks source_eigenvalues).")

    # wet seed: omega_dry / sqrt(1 + A_jj(omega_dry))  (avoids dry-seed lock-on)
    seeds = {}
    for j in track_modes:
        if bem:
            A0, _ = capy_at_omega(capy, omega_dry[j], mode_slice=slice(0, nm), warn_out_of_grid=False)
            A0 = T[:nm, :nm].T @ A0 @ T[:nm, :nm]
            seeds[j] = omega_dry[j] / np.sqrt(1.0 + max(A0[j, j], 0.0))
        else:
            seeds[j] = omega_dry[j]
        print(f">>> branch {BRANCH_NAMES.get(j, j)}: dry {omega_dry[j]/2/np.pi:.2f} Hz -> "
              f"seed {seeds[j]/2/np.pi:.2f} Hz", flush=True)

    branch_state = {j: dict(omega=seeds[j], vec=None, lost=0) for j in track_modes}
    rows = []
    contributions_history = []
    spec_v, spec_lam, spec_vec, spec_part = [], [], [], []
    for iv, V in enumerate(np.asarray(v_list, dtype=float).ravel()):
        if V <= 0:
            continue
        if iv % 10 == 0:
            print(f"    p-k sweep {iv + 1}/{len(v_list)}  V={V:.2f} m/s ({V/KN:.1f} kn)", flush=True)
        last_res = None
        branch_results = {}
        for j in track_modes:
            st = branch_state[j]
            if st["lost"] >= 3:
                continue
            res = pk_solve_mode_at_velocity(
                ls, capy, T, V, j, st["omega"], st["vec"], beam_idx, nm,
                bem=bem, A_offset=A_offset, relax=relax, tol_rel=tol_rel,
                max_iter=max_iter, part_floor=part_floor,
                freq_window_hz=freq_window_hz,
            )
            last_res = res
            if res["vec"] is not None:
                st["omega"], st["vec"], st["lost"] = res["omega"], res["vec"], 0
                branch_results[j] = res
                rows.append(dict(branch=BRANCH_NAMES.get(j, str(j)), V_m_s=res["V"],
                                 V_kn=res["V"] / KN, g=res["g"], freq_Hz=res["freq_Hz"],
                                 mac=res["mac"], part=res["part"], n_iter=res["n_iter"],
                                 converged=res["converged"]))
            else:
                st["lost"] += 1
                if st["lost"] == 3:
                    print(f"    !! branch {BRANCH_NAMES.get(j, j)} lost at V={V/KN:.1f} kn "
                          f"(3 consecutive failures) — terminated", flush=True)
        contrib = extract_contributions_at_velocity(
            ls, capy, T, V, branch_results, nm=nm, track_modes=track_modes,
            bem=bem, A_offset=A_offset,
        )
        if contrib is not None:
            contributions_history.append(contrib)
        if last_res is not None and last_res["lam_full"] is not None:
            spec_v.append(float(V))
            spec_lam.append(last_res["lam_full"])
            spec_vec.append(last_res["vec_beam_full"])
            spec_part.append(last_res["part_full"])
        if step_report is not None and last_res is not None and last_res["lam_full"] is not None:
            step_report(iv, float(V), branch_results, last_res)
    return rows, (spec_v, spec_lam, spec_vec, spec_part, beam_idx), contributions_history


# --------------------------------------------------------------------------- const-A eigensweep
def eig_sweep_const_A(ls, capy, T, v_list, *, nm, omega_star, reporter=None,
                      track_modes=(0, 1), seed_part_floor=5e-4,
                      freq_window_hz=(1.0, 40.0), A_offset=None, bem=True):
    """Plain eigenvalue sweep with A(omega*), B(omega*) frozen — no p-k iteration.

    The UVLM is a state-space LTI model (no frequency dependence); the only
    omega-dependent input is the Capytaine A(omega)/B(omega). Freezing them at
    omega* makes each velocity ONE eigensolve giving ALL roots simultaneously;
    branches are recovered afterwards by MAC-chaining over the full spectrum
    (no participation floor, no seeds, no branch loss). Validated on Model2t
    (identical V_f at omega* 3.2x apart); run two omega* to confirm flatness
    on a new case.
    """
    beam_idx = beam_state_indices(ls.ss, nm)
    rows = []
    spec_v, spec_lam, spec_vec, spec_part = [], [], [], []
    chains: list[dict] = []          # [{name, ref (beam vec), j}]
    for iv, V in enumerate(np.asarray(v_list, dtype=float).ravel()):
        lam, Vec, _ = _eig_at(ls, float(V), capy=capy, T=T, omega=float(omega_star),
                              n_modes=nm, bem=bem, A_offset=A_offset)
        vec_beam = Vec[beam_idx, :]
        vnorm = np.maximum(np.linalg.norm(Vec, axis=0), 1e-300)
        part = np.linalg.norm(vec_beam, axis=0) / vnorm
        freq = lam.imag / (2 * np.pi)

        if iv == 0:
            cand = np.flatnonzero((lam.imag > 1e-6) & (part > seed_part_floor)
                                  & (freq >= freq_window_hz[0])
                                  & (freq <= freq_window_hz[1]))
            cand = sorted(cand, key=lambda k: lam[k].imag)
            for k in cand[: nm]:
                q = np.abs(vec_beam[:nm, k])
                j = int(np.argmax(q))
                name = BRANCH_NAMES.get(j, f"mode{j}")
                if any(c["name"] == name for c in chains):
                    name = f"{name}_b"
                chains.append(dict(name=name, ref=vec_beam[:, k].copy(), j=j))
            print(f">>> const-A eigensweep: A,B frozen at omega* = {omega_star:g} rad/s; "
                  f"chains seeded: {[c['name'] for c in chains]}", flush=True)

        for c in chains:
            macs = (np.abs(np.conj(c["ref"]) @ vec_beam) ** 2
                    / ((np.conj(c["ref"]) @ c["ref"]).real
                       * np.maximum(np.einsum("ij,ij->j", np.conj(vec_beam),
                                              vec_beam).real, 1e-300)))
            macs = np.where(lam.imag >= -1e-9, macs.real, -1.0)
            k = int(np.argmax(macs))
            m_sel = float(macs[k])
            g = float(lam[k].real / max(abs(lam[k].imag), 1e-30))
            rows.append(dict(branch=c["name"], V_m_s=float(V), V_kn=float(V) / KN,
                             g=g, freq_Hz=float(freq[k]), mac=m_sel,
                             part=float(part[k]), n_iter=1, converged=m_sel > 0.4))
            if m_sel > 0.4:
                c["ref"] = vec_beam[:, k].copy()

        spec_v.append(float(V))
        spec_lam.append(lam)
        spec_vec.append(vec_beam)
        spec_part.append(part)
        if iv % 10 == 0:
            print(f"    const-A sweep {iv + 1}/{len(v_list)}  V={V:.2f} m/s "
                  f"({V/KN:.1f} kn)", flush=True)
        if reporter is not None:
            reporter(iv, float(V), {}, dict(lam_full=lam, vec_beam_full=vec_beam,
                                            part_full=part))
    return rows, (spec_v, spec_lam, spec_vec, spec_part, beam_idx)


# --------------------------------------------------------------------------- outputs
def flutter_crossing(rows, branch):
    pts = sorted(((r["V_m_s"], r["g"], r["freq_Hz"]) for r in rows if r["branch"] == branch))
    for (v0, g0, f0), (v1, g1, f1) in zip(pts, pts[1:]):
        if g0 <= 0.0 < g1:
            a = -g0 / (g1 - g0)
            return v0 + a * (v1 - v0), f0 + a * (f1 - f0)
    return None, None


def write_outputs(rows, spectra, outdir: Path, nm: int, tag: str, track_modes,
                  contributions_history=None):
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"abramson_pk_bem_tracked_nm{nm}.csv"
    with csv_path.open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=["branch", "V_m_s", "V_kn", "g", "freq_Hz",
                                            "mac", "part", "n_iter", "converged"])
        wri.writeheader()
        for r in rows:
            wri.writerow(r)

    spec_v, spec_lam, spec_vec, spec_part, beam_idx = spectra
    if spec_v:
        np.savez(
            outdir / f"abramson_sharpy_eigvecs_nm{nm}.npz",
            velocities=np.array(spec_v), lam=np.array(spec_lam),
            struct_vecs=np.array(spec_vec), participation=np.array(spec_part),
            beam_idx=np.asarray(beam_idx), kn=KN,
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axg, axf) = plt.subplots(1, 2, figsize=(13, 5))
    col = {"bending": "#1f77b4", "torsion": "#ff7f0e", "mode3": "#2ca02c", "mode4": "#d62728"}
    summary = {}
    # branch names from the rows themselves: p-k rows use BRANCH_NAMES, const-A
    # chains are named by dominant |q| content and may carry a "_b" suffix
    names = list(dict.fromkeys(r["branch"] for r in rows))
    for name in names:
        pts = sorted(((r["V_kn"], r["g"], r["freq_Hz"]) for r in rows if r["branch"] == name))
        if not pts:
            continue
        v, g, fz = map(np.array, zip(*pts))
        c = col.get(name, col.get(name.removesuffix("_b"), None))
        axg.plot(v, g, ".-" if not name.endswith("_b") else "x--", ms=3, color=c, label=name)
        axf.plot(v, fz, ".-" if not name.endswith("_b") else "x--", ms=3, color=c, label=name)
        vf, ff = flutter_crossing(rows, name)
        summary[name] = (vf, ff)
        if vf is not None:
            axg.axvline(vf / KN, color=c or "k", ls=":", lw=1)
    for ax in (axg, axf):
        ax.axvline(V_EXP_KN, color="k", ls="--", lw=1, label=f"exp {V_EXP_KN} kn")
        ax.axvline(V_GINO_KN, color="gray", ls="-.", lw=1, label=f"GINO {V_GINO_KN} kn")
        ax.grid(alpha=0.3)
        ax.set_xlabel("V [kn]")
    axf.axhline(F_EXP_HZ, color="k", ls=":", lw=0.8)
    axg.axhline(0.0, color="k", lw=0.8)
    axg.set_ylabel(r"$g = \sigma/\omega$ [–]")
    axf.set_ylabel("frequency [Hz]")
    axg.legend(fontsize=8)
    fig.suptitle(f"Abramson 1965 — SHARPy p-k + Capytaine BEM ({tag})")
    fig.tight_layout()
    fig.savefig(outdir / f"abramson_pk_bem_vgvf_nm{nm}.png", dpi=150)
    plt.close(fig)

    if contributions_history:
        from gino_bem.contributions_plot import plot_stiffness_damping_contributions

        plot_stiffness_damping_contributions(
            contributions_history, outdir, v_knots=False, show_plot=False,
        )

    return csv_path, summary


# --------------------------------------------------------------------------- dry check
def dry_check(data, ls, capy, nm):
    """Print added-mass block norms + basis/alignment report, then return T."""
    sys_uvlm = ls.uvlm.sys
    K, K_star, Kzeta = sys_uvlm.K, sys_uvlm.K_star, sys_uvlm.Kzeta
    C = sys_uvlm.SS.C
    css_block = np.linalg.norm(np.asarray(C[:, K + K_star:2 * K + K_star]))
    D = sys_uvlm.D_predictor if sys_uvlm.D_predictor is not None else sys_uvlm.SS.D
    dss_block = np.linalg.norm(np.asarray(D[:, 3 * Kzeta:6 * Kzeta]))
    print("\n=== dry check ===")
    print(f"added_mass_removal = '{getattr(sys_uvlm, 'added_mass_removal', '?')}'")
    print(f"|Css gamma_dot block|_F = {css_block:.6e}   (0 for gamma_dot/full)")
    print(f"|Dss zeta_dot  block|_F = {dss_block:.6e}   (0 for full)")
    T, report = align_capy_to_sharpy(capy, data, nm)
    return T


def _dry_omegas_phys(ls, nm: int) -> np.ndarray:
    """Physical dry natural frequencies [rad/s] from the unscaled K snapshot."""
    bs = ls.beam.sys
    K_ref = bs.scaled_reference_matrices.get("C") is not None and \
        bs.scaled_reference_matrices.get("K")
    if K_ref is None or K_ref is False:
        K_ref = bs.Kstr
    U = np.asarray(bs.U, dtype=float)[:, :nm]
    return np.sqrt(np.abs(np.einsum("ij,jk,ki->i", U.T, np.asarray(K_ref, dtype=float), U)))


def _probe_top_roots(ls, beam_idx, V: float, n_top: int = 8, *,
                     part_floor: float = 0.005, part_ceiling: float | None = None):
    """Top-participation oscillatory roots at V (no BEM), wide frequency window.

    ``part_ceiling`` (optional) discards near-pure beam roots — the spurious
    fixed-scaled-frequency Newmark root carries part ~1 on Model2t, far above
    any physical wet root (part <~ 0.01)."""
    lam, Vec, _ = _eig_at(ls, V, bem=False)
    cand, part = _candidates(lam, Vec, beam_idx, part_floor=part_floor,
                             freq_window_hz=(1.0, 100.0))
    if part_ceiling is not None:
        cand = [k for k in cand if part[k] <= part_ceiling]
    top = sorted(cand, key=lambda k: -part[k])[:n_top]
    return [(float(lam[k].imag), Vec[beam_idx, k].copy()) for k in top]


def _probe_uvlm_added_mass_diag(ls_none, ls_gd, beam_idx, beam_idx_gd, nm: int,
                                V: float, *, part_floor: float = 0.005,
                                part_ceiling: float | None = None,
                                require_all: bool = True,
                                return_macs: bool = False) -> np.ndarray:
    """Measured UVLM apparent-mass diagonal by MAC-paired frequency ratios.

    Roots of the intact ('none') and apparent-mass-free ('gamma_dot') systems
    are paired by mode-shape MAC (same physical branch), then assigned to
    structural modes by nearest dry frequency of the gamma_dot root. Aero
    stiffness cancels in the ratio (omega_gd/omega_none)^2.
    """
    roots_n = _probe_top_roots(ls_none, beam_idx, V,
                               part_floor=part_floor, part_ceiling=part_ceiling)
    roots_g = _probe_top_roots(ls_gd, beam_idx_gd, V,
                               part_floor=part_floor, part_ceiling=part_ceiling)
    om_dry = _dry_omegas_phys(ls_gd, nm)
    pairs = []   # (mac, i_none, j_gd)
    for i, (_, v_n) in enumerate(roots_n):
        for j, (_, v_g) in enumerate(roots_g):
            pairs.append((mac(v_n, v_g), i, j))
    pairs.sort(reverse=True, key=lambda t: t[0])
    used_n, used_g, matched = set(), set(), []
    for m, i, j in pairs:
        if i in used_n or j in used_g or m < 0.5:
            continue
        # removing apparent mass can only RAISE a root's frequency; a pair
        # with om_gd <= om_none is a cross-mode mismatch, not a measurement
        if roots_g[j][0] <= roots_n[i][0] * 1.001:
            continue
        used_n.add(i)
        used_g.add(j)
        matched.append((roots_n[i][0], roots_g[j][0], m))
    diag = np.full(nm, np.nan)
    diag_mac = np.full(nm, np.nan)
    used_modes = set()
    for om_n, om_g, m in sorted(matched, key=lambda t: -t[2]):
        jm = int(np.argmin([np.inf if k in used_modes else abs(om_g - om_dry[k])
                            for k in range(nm)]))
        if abs(om_g - om_dry[jm]) > 0.4 * om_dry[jm]:
            continue
        used_modes.add(jm)
        diag[jm] = (om_g / om_n) ** 2 - 1.0
        diag_mac[jm] = m
        print(f"[delta]   mode {jm}: f_none={om_n/2/np.pi:6.2f} Hz  f_gd={om_g/2/np.pi:6.2f} Hz "
              f"(dry {om_dry[jm]/2/np.pi:6.2f} Hz, MAC={m:.3f}) -> A_jj={diag[jm]:.3f}")
    # Second pass: assign leftover MAC pairs to unassigned modes (no dry-freq gate).
    assigned_pairs = {(om_n, om_g) for j in range(nm) if np.isfinite(diag[j])
                      for om_n, om_g, _ in matched
                      if abs((om_g / om_n) ** 2 - 1.0 - diag[j]) < 1e-6}
    for om_n, om_g, m in sorted(matched, key=lambda t: -t[2]):
        if (om_n, om_g) in assigned_pairs:
            continue
        free = [j for j in range(nm) if j not in used_modes]
        if not free:
            break
        jm = int(min(free, key=lambda j: abs(om_g - om_dry[j])))
        used_modes.add(jm)
        diag[jm] = (om_g / om_n) ** 2 - 1.0
        diag_mac[jm] = m
        assigned_pairs.add((om_n, om_g))
        print(f"[delta]   mode {jm} (2nd pass): f_none={om_n/2/np.pi:6.2f} Hz  "
              f"f_gd={om_g/2/np.pi:6.2f} Hz (MAC={m:.3f}) -> A_jj={diag[jm]:.3f}")
    if np.isnan(diag).any():
        if require_all:
            raise RuntimeError(f"UVLM apparent-mass probe incomplete: diag={diag} "
                               f"(matched {len(matched)} root pairs)")
        print(f"[delta] WARNING: probe incomplete, diag={np.array2string(diag, precision=3)} "
              "— missing modes fall back to ratio-scaled strip values")
    diag = np.maximum(diag, 1e-3)   # NaN propagates (handled by caller)
    if return_macs:
        return diag, diag_mac
    return diag


def build_delta_offset(data, capy, T, nm: int, ls, *, grid, rayleigh, zetas, v_max_kn,
                       a_ea: float = -0.5, v_probe: float = 4.5,
                       build_case_fn=None, probe_tag: str = "abramson_pk_probe_gd",
                       strip_b: float | None = None, rho: float = 997.0,
                       wetted_mask: np.ndarray | None = None,
                       probe_part_floor: float = 0.005,
                       probe_part_ceiling: float | None = None,
                       require_all_modes: bool = True) -> np.ndarray:
    """UVLM apparent mass in SHARPy modal coords, for delta injection.

    The 2D analytic flat-plate strip matrix overestimates the 3D UVLM (finite-
    span relief at AR~2.5), so the diagonal is MEASURED: a probe case with
    ``gamma_dot`` removal is built and the wet-frequency shift at the same
    velocity gives A_uvlm_jj = (omega_gd/omega_none)^2 - 1 (aero stiffness
    cancels in the ratio). The strip matrix supplies the off-diagonal coupling
    structure, rescaled to the measured diagonal.

    Reusable by other geometries (e.g. Model2t): ``build_case_fn`` builds the
    gamma_dot probe case (same keyword signature as the local ``build_case``),
    ``strip_b`` is the semichord [m], ``a_ea`` the EA in semichords aft of
    midchord, and ``wetted_mask`` (over the free-node span stations) restricts
    the strip projection to the submerged span for surface-piercing struts.
    ``require_all_modes=False`` fills modes the probe cannot pair (e.g. m4
    hidden behind spurious roots) with ratio-scaled strip diagonals.
    """
    if build_case_fn is None:
        build_case_fn = build_case
    y_s, W_s, TH_s, _ = sharpy_modal_displacements(data, nm)
    if wetted_mask is not None:
        wetted_mask = np.asarray(wetted_mask, dtype=bool)
        print(f"[delta] strip projection restricted to wetted span: "
              f"{int(wetted_mask.sum())}/{wetted_mask.size} stations")
        y_s, W_s, TH_s = y_s[wetted_mask], W_s[:, wetted_mask], TH_s[:, wetted_mask]
    strip_kw = dict(a=a_ea, rho=rho)
    if strip_b is not None:
        strip_kw["b"] = float(strip_b)
    A_strip = strip_added_mass_modal(y_s, W_s, TH_s, **strip_kw)
    A_hf = np.asarray(capy["added_mass"], dtype=float)[-1][:nm, :nm]
    A_hf = T[:nm, :nm].T @ A_hf @ T[:nm, :nm]
    if A_strip[0, 1] * A_hf[0, 1] < 0:
        print("[delta] flipping strip theta convention to match Capytaine coupling sign")
        A_strip = strip_added_mass_modal(y_s, W_s, -TH_s, **strip_kw)

    print("[delta] building gamma_dot probe case to measure UVLM apparent mass ...", flush=True)
    data_gd, _ = build_case_fn(nm, grid=grid, added_mass_removal="gamma_dot", rayleigh=rayleigh,
                               zetas=zetas, v_max_kn=v_max_kn, case_tag=probe_tag)
    ls_gd = data_gd.linear.linear_system
    beam_idx = beam_state_indices(ls.ss, nm)
    beam_idx_gd = beam_state_indices(ls_gd.ss, nm)
    # Merge probes over one or more velocities: per mode keep the best-MAC
    # pairing, and reject measurements above the 2D strip value — 3D relief can
    # only REDUCE apparent mass, so diag > strip flags a contaminated pairing
    # (circulatory stiffening / cross-mode coupling at that probe speed).
    strip_diag = np.diag(A_strip)
    v_probes = np.atleast_1d(np.asarray(v_probe, dtype=float)).ravel()
    diag_meas = np.full(nm, np.nan)
    mac_best = np.full(nm, -1.0)
    for vp in v_probes:
        print(f"[delta] probe V={vp:g} m/s: MAC-pairing none vs gamma_dot roots ...", flush=True)
        d, dm = _probe_uvlm_added_mass_diag(ls, ls_gd, beam_idx, beam_idx_gd, nm, float(vp),
                                            part_floor=probe_part_floor,
                                            part_ceiling=probe_part_ceiling,
                                            require_all=False, return_macs=True)
        bad = np.isfinite(d) & (d > 1.05 * np.maximum(strip_diag, 1e-12))
        if bad.any():
            print(f"[delta]   rejecting modes {np.flatnonzero(bad)} at V={vp:g}: "
                  f"A_jj above the 2D strip bound (contaminated pairing)")
            d[bad] = np.nan
        upd = np.isfinite(d) & (np.nan_to_num(dm, nan=-1.0) > mac_best)
        diag_meas[upd] = d[upd]
        mac_best[upd] = dm[upd]
    if require_all_modes and np.isnan(diag_meas).any():
        raise RuntimeError(f"UVLM apparent-mass probe incomplete after "
                           f"{len(v_probes)} probe speed(s): diag={diag_meas}")
    print(f"[delta] measured UVLM A_jj = {np.array2string(diag_meas, precision=3)}  "
          f"(2D strip diag = {np.array2string(strip_diag, precision=3)})")
    missing = np.isnan(diag_meas)
    if missing.any():
        found = ~missing
        if not found.any():
            raise RuntimeError("UVLM apparent-mass probe paired no modes at all")
        ratio = float(np.mean(diag_meas[found] / np.maximum(np.diag(A_strip)[found], 1e-12)))
        diag_meas[missing] = ratio * np.diag(A_strip)[missing]
        print(f"[delta] filled modes {np.flatnonzero(missing)} with strip diag x "
              f"mean 3D-relief ratio {ratio:.3f} -> "
              f"A_jj = {np.array2string(diag_meas, precision=3)}")

    D = np.diag(np.sqrt(diag_meas / np.maximum(np.diag(A_strip), 1e-12)))
    A_uvlm = D @ A_strip @ D
    print("[delta] A_uvlm (measured-diag, strip coupling structure):")
    print(np.array2string(A_uvlm, precision=3, suppress_small=True))
    print("[delta] A_capy(omega_max) in SHARPy basis:")
    print(np.array2string(A_hf, precision=3, suppress_small=True))
    ev = np.linalg.eigvalsh(np.eye(nm) + A_hf - A_uvlm)
    print(f"[delta] eig(I + A_capy - A_uvlm) = {np.array2string(ev, precision=3)}")
    if ev.min() < 0.05:
        raise RuntimeError("delta injection makes the modal mass matrix near-singular; "
                           "check basis alignment / strip parameters")
    return A_uvlm


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--added-mass-removal", choices=["none", "gamma_dot", "full"], default="gamma_dot")
    ap.add_argument("--bem-mode", choices=["replace", "delta"], default="replace",
                    help="replace: remove UVLM apparent mass, inject A_capy. "
                         "delta: keep UVLM intact (removal forced to 'none'), inject "
                         "A_capy - A_flat_plate so the UVLM keeps its forward-speed "
                         "non-circulatory damping")
    ap.add_argument("--strip-a", type=float, default=-0.5,
                    help="elastic-axis position in semichords aft of midchord (EA at c/4 = -0.5)")
    ap.add_argument("--delta-v-probe", type=str, default="3.0,4.5,6.0",
                    help="comma-separated probe speeds [m/s] for the delta apparent-mass "
                         "measurement (best-MAC merge per mode). Keep V high enough that "
                         "the UVLM Nyquist pi*M*V/c clears the mode band: at V<~2 m/s the "
                         "10-20 Hz Abramson modes alias (near-zero A_jj with high MAC)")
    ap.add_argument("--no-bem", action="store_true", help="skip Capytaine injection")
    ap.add_argument("--single-omega", action="store_true",
                    help="debug: single global-omega fixed point (prototype behavior)")
    ap.add_argument("--num-modes", type=int, default=4)
    ap.add_argument("--track-modes", type=str, default="0,1")
    ap.add_argument("--v-min-kn", type=float, default=8.0)
    ap.add_argument("--v-max-kn", type=float, default=60.0)
    ap.add_argument("--v-step-kn", type=float, default=None,
                    help="uniform velocity sweep step [kn]; e.g. 1.0 → 8,9,...,60 kn")
    ap.add_argument("--stride", type=int, default=1, help="subsample the dense GINO velocity list")
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--fine", action="store_true")
    ap.add_argument("--grid", type=str, default=None,
                    choices=sorted(GRIDS.keys()),
                    help="named aerogrid preset (e.g. n20c10 = 20 span × 10 chord panels)")
    ap.add_argument("--grid-span", type=int, default=None, help="spanwise panel count N (overrides preset grid)")
    ap.add_argument("--grid-chord", type=int, default=None, help="chordwise panel count M (overrides preset grid)")
    ap.add_argument("--mstar-fact", type=int, default=None, help="wake chord panel factor Ms (default 6)")
    ap.add_argument("--outdir", type=Path, default=None, help="output directory for CSV/PNG/NPZ")
    ap.add_argument("--capy-npz", type=Path, default=None)
    ap.add_argument("--no-rayleigh", action="store_true")
    ap.add_argument("--zeta-bend", type=float, default=ABRAMSON_RAYLEIGH_ZETAS[0])
    ap.add_argument("--zeta-tors", type=float, default=ABRAMSON_RAYLEIGH_ZETAS[1])
    ap.add_argument("--relax", type=float, default=0.618)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--const-a-omega", type=float, default=None, metavar="OMEGA",
                    help="Freeze Capytaine A,B at this omega [rad/s] and do a plain "
                    "eigenvalue sweep (no p-k iteration): one eigensolve per "
                    "velocity, all roots at once, branches MAC-chained over the "
                    "full spectrum. Justified when A(omega) is nearly flat "
                    "(proven on Model2t); run two omega* to confirm flatness here.")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--dry-check", action="store_true",
                    help="stop after LinearAssembler; print patch + alignment report")
    args = ap.parse_args()

    nm = args.num_modes
    if args.grid_span is not None or args.grid_chord is not None:
        if args.grid_span is None or args.grid_chord is None:
            raise SystemExit("--grid-span and --grid-chord must be set together")
        Ms = args.mstar_fact if args.mstar_fact is not None else 6
        grid = (args.grid_chord, args.grid_span, Ms)
        grid_name = f"n{args.grid_span}c{args.grid_chord}"
    elif args.grid is not None:
        grid_name = args.grid
        grid = GRIDS[grid_name]
    else:
        grid_name = "coarse" if args.coarse else ("fine" if args.fine else "default")
        grid = GRIDS[grid_name]
    bem = not args.no_bem
    removal = args.added_mass_removal
    if args.bem_mode == "delta":
        if not bem:
            raise SystemExit("--bem-mode delta requires BEM injection (drop --no-bem)")
        if removal != "none":
            print(f"[delta] forcing added_mass_removal 'none' (was '{removal}'): "
                  "the UVLM must stay intact for delta injection")
            removal = "none"
    track_modes = tuple(int(x) for x in args.track_modes.split(",") if x.strip() != "")
    mode_tag = "delta" if args.bem_mode == "delta" else removal
    tag = args.tag or f"{mode_tag}_{'bem' if bem else 'nobem'}_nm{nm}_{grid_name}"

    capy = check_capy_basis(load_capytaine_npz(args.capy_npz or default_capytaine_npz()))

    data, vel = build_case(
        nm, grid=grid, added_mass_removal=removal,
        rayleigh=not args.no_rayleigh, zetas=(args.zeta_bend, args.zeta_tors),
        v_max_kn=args.v_max_kn, case_tag=f"abramson_pk_{tag}",
    )
    ls = data.linear.linear_system

    T = dry_check(data, ls, capy, nm) if args.dry_check else align_capy_to_sharpy(capy, data, nm)[0]
    A_offset = None
    if args.bem_mode == "delta":
        A_offset = build_delta_offset(
            data, capy, T, nm, ls, grid=grid, rayleigh=not args.no_rayleigh,
            zetas=(args.zeta_bend, args.zeta_tors), v_max_kn=args.v_max_kn,
            a_ea=args.strip_a,
            v_probe=[float(x) for x in args.delta_v_probe.split(",") if x.strip()],
            probe_part_floor=1e-4, probe_part_ceiling=0.5,
            require_all_modes=False,
        )
    if args.dry_check:
        print("=== dry check done (no sweep) ===")
        return 0

    v_min = max(R._kn_to_m_s(args.v_min_kn), ABRAMSON_V_MIN_M_S)
    if args.v_step_kn is not None:
        kn = np.arange(float(args.v_min_kn), float(args.v_max_kn) + 0.5 * float(args.v_step_kn),
                       float(args.v_step_kn))
        v_list = kn * KN
        print(f">>> velocity sweep: {len(v_list)} points, "
              f"{args.v_min_kn:.0f}–{args.v_max_kn:.0f} kn every {args.v_step_kn:g} kn", flush=True)
    else:
        v_list = vel[vel >= v_min][:: max(1, args.stride)]
        print(f">>> velocity sweep: {len(v_list)} points (dense GINO list"
              f"{f', stride={args.stride}' if args.stride > 1 else ''})", flush=True)

    if args.single_omega:
        res = velocity_sweep_with_bem(ls, capy, v_list, n_modes=nm, T=T)
        for r in res[:: max(1, len(res) // 20)]:
            print(f"  V={r['V']/KN:6.1f} kn  g={r['g']:+.3e}  f={r['omega_rad_s']/2/np.pi:6.2f} Hz")
        return 0

    if args.const_a_omega is not None:
        rows, spectra = eig_sweep_const_A(
            ls, capy, T, v_list, nm=nm, omega_star=args.const_a_omega,
            A_offset=A_offset, bem=bem,
        )
        contributions_history = None
        track_modes = tuple(range(nm))     # plot every MAC-chained branch
        tag += f"_constA_w{args.const_a_omega:g}"
    else:
        rows, spectra, contributions_history = pk_sweep_bem(
            ls, capy, T, v_list, nm=nm, track_modes=track_modes, bem=bem, A_offset=A_offset,
            relax=args.relax, tol_rel=args.tol,
        )
    outdir = args.outdir or (
        _FSI_ROOT / "output_data" / "ABRAMSON1965" / "07_sharpy_pk_bem" / tag
    )
    csv_path, summary = write_outputs(
        rows, spectra, outdir, nm, tag, track_modes,
        contributions_history=contributions_history,
    )
    print(f"\n=== {tag} ===")
    for name, (vf, ff) in summary.items():
        if vf is None:
            print(f"  {name:>8s}: no flutter crossing in sweep")
        else:
            print(f"  {name:>8s}: V_f = {vf/KN:.2f} kn ({vf:.2f} m/s), f_f = {ff:.2f} Hz  "
                  f"[exp {V_EXP_KN} kn / GINO {V_GINO_KN} kn]")
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
