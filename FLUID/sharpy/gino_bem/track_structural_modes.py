#!/usr/bin/env python3
"""
Clean V-g / V-f for Abramson 1965 (stock SHARPy UVLM, water) with structural
mode tracking by EIGENVECTOR PARTICIPATION.

The full-order linear UVLM has ~280 eigenvalues per velocity, dominated by
heavily/lightly damped *wake* modes that pollute the 4-20 Hz band and make
frequency-only tracking jump between branches. Here we keep only the modes
whose eigenvector energy lives in the structural (beam modal) states, then
track the bending and torsion branches by continuity.

  conda activate sharpy
  cd sharpy && python gino_bem/track_structural_modes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sclalg

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))

from gino_bem.abramson1965_wing import Abramson1965  # noqa: E402
from gino_bem import run_abramson_water as R  # noqa: E402
from gino_bem.abramson_tracking_config import (  # noqa: E402
    ABRAMSON_V_MIN_KN,
    ABRAMSON_V_MIN_M_S,
    at_or_above_m_s,
)
from gino_bem.rayleigh_damping import (  # noqa: E402
    ABRAMSON_RAYLEIGH_MODE_IDS,
    ABRAMSON_RAYLEIGH_ZETAS,
    install_rayleigh_structural_damping,
)
import sharpy.sharpy_main  # noqa: E402

KN = 0.514444


def build_and_run(nm=4, *, rayleigh: bool = True, zetas=ABRAMSON_RAYLEIGH_ZETAS, v_max_kn: float = 32.0):
    M, N, Ms = 6, 12, 4
    route = _SHARPY_ROOT / "gino_bem"
    (route / "cases").mkdir(parents=True, exist_ok=True)
    out_root = route / "output"
    out_root.mkdir(exist_ok=True)
    case_tag = f"abramson_track_nm{nm}" + ("_rayleigh" if rayleigh else "")
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
    # Skip the redundant built-in AsymptoticStability velocity sweep: we only need
    # the assembled linear_system object; our own eigenvector loop does the sweep.
    ws.config["SHARPy"]["flow"] = [
        "BeamLoader", "AerogridLoader", "StaticCoupled", "Modal", "LinearAssembler",
    ]
    ws.config.write()
    if rayleigh:
        install_rayleigh_structural_damping(
            mode_ids=ABRAMSON_RAYLEIGH_MODE_IDS,
            target_zetas=zetas,
        )
        print(
            f">>> Rayleigh structural damping: "
            f"ζ=({zetas[0]:.4g}, {zetas[1]:.4g}) on modes {list(ABRAMSON_RAYLEIGH_MODE_IDS)}",
            flush=True,
        )
    print(
        f">>> Velocity sweep: {len(vel)} points, "
        f"V=[{vel.min():.2f}, {vel.max():.2f}] m/s "
        f"({vel.min()/KN:.1f}–{vel.max()/KN:.1f} kn)",
        flush=True,
    )
    print(">>> Running SHARPy flow up to LinearAssembler (no built-in sweep)...", flush=True)
    data = sharpy.sharpy_main.main(["", ws.route + ws.case_name + ".sharpy"])
    print(">>> LinearAssembler done; starting eigenvector participation loop.", flush=True)
    return data, vel, nm


def beam_state_indices(ss, num_modes):
    """Return indices of the beam (modal) states inside the coupled state vector."""
    names = []
    try:
        for v in ss.state_variables:
            names.append((v.name, int(v.first_position), int(v.size)))
    except Exception:
        names = []
    print("\nCoupled state-variable blocks (name, first, size):")
    for n, f, s in names:
        print(f"   {n:>20s}  start={f:<6d} size={s}")
    beam_idx = []
    for n, f, s in names:
        if any(k in n.lower() for k in ("q", "beam", "eta", "modal", "flex")):
            beam_idx.extend(range(f, f + s))
    if not beam_idx:
        # fallback: beam states are the trailing 2*num_modes states
        ntot = ss.A.shape[0]
        beam_idx = list(range(ntot - 2 * num_modes, ntot))
        print(f"   (name match failed -> fallback trailing {2*num_modes} states)")
    print(f"   beam state count = {len(beam_idx)} / {ss.A.shape[0]} total\n")
    return np.array(beam_idx, dtype=int)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-modes", type=int, default=4,
                    help="structural modes kept in the linear beam projection (2 = bending+torsion only)")
    ap.add_argument("--no-rayleigh", action="store_true",
                    help="disable GINO-matched Rayleigh structural damping")
    ap.add_argument("--zeta-bend", type=float, default=ABRAMSON_RAYLEIGH_ZETAS[0])
    ap.add_argument("--zeta-tors", type=float, default=ABRAMSON_RAYLEIGH_ZETAS[1])
    ap.add_argument("--output-tag", type=str, default=None,
                    help="subfolder under sharpy_uvlm_water (default: _baseline_nm{N}[_rayleigh])")
    ap.add_argument("--v-max-kn", type=float, default=32.0,
                    help="maximum velocity in the eigen sweep [kn] (default 32; use 150 for extended run)")
    args = ap.parse_args()
    nm = args.num_modes
    rayleigh = not args.no_rayleigh
    zetas = (float(args.zeta_bend), float(args.zeta_tors))
    v_max_kn = float(args.v_max_kn)
    data, vel, _ = build_and_run(nm, rayleigh=rayleigh, zetas=zetas, v_max_kn=v_max_kn)
    ls = data.linear.linear_system
    beam_idx = beam_state_indices(ls.ss, nm)

    # Per-velocity structural eigenvalues (participation-filtered)
    struct = []  # list of (V, [(lam_real_1/s, freq_Hz, participation), ...])
    # Collectors for MAC tracking: full continuous eigenvalues + structural eigenvector
    # blocks (the q, q_dot components) at every velocity, in a fixed state basis.
    mac_vels, mac_lam, mac_vecs, mac_part = [], [], [], []
    for iu, u in enumerate(vel):
        u = float(u)
        if u <= 0:
            continue
        if iu % 20 == 0:
            print(f"    eig loop {iu+1}/{len(vel)}  V={u:.2f} m/s ({u/KN:.1f} kn)", flush=True)
        ss = ls.update(u)
        w, V = sclalg.eig(ss.A, right=True)
        dt_dim = ls.uvlm.sys.ScalingFacts["length"] / u * ss.dt
        lam = np.log(w) / dt_dim
        vnorm = np.linalg.norm(V, axis=0)
        pnorm = np.linalg.norm(V[beam_idx, :], axis=0)
        part = pnorm / np.maximum(vnorm, 1e-30)
        mac_vels.append(u)
        mac_lam.append(lam)                     # (Nstates,) continuous-time
        mac_vecs.append(V[beam_idx, :])         # (n_beam_states, Nstates)
        mac_part.append(part)                   # (Nstates,)
        modes = []
        for k in range(len(lam)):
            if lam[k].imag <= 1e-6:
                continue  # upper half plane only
            f_hz = lam[k].imag / (2 * np.pi)
            if f_hz < 1.0 or f_hz > 40.0:
                continue
            modes.append((float(lam[k].real), float(f_hz), float(part[k])))
        struct.append((u, modes))

    # Identify structural modes: highest-participation ones per velocity
    # Track 2 branches (bending, torsion) by continuity, seeded at the highest-V
    # end where the spectrum is clean.
    def pick_top(modes, n=4, pmin=0.2):
        cand = [m for m in modes if m[2] >= pmin]
        cand.sort(key=lambda m: -m[2])
        return cand[:n]

    # Build branch series by frequency continuity over participation-filtered modes
    vels = [u for u, _ in struct]
    # Seed frequencies from the participation-top modes at the first reliable speed
    # (>= ABRAMSON_V_MIN_M_S, where bending/torsion frequencies have stabilized).
    branches = {"bending": [], "torsion": []}
    seed_struct = next((item for item in struct if at_or_above_m_s(item[0])), struct[0])
    seed_modes = pick_top(seed_struct[1], n=6, pmin=0.05)
    seed_modes.sort(key=lambda m: m[1])  # by freq
    if len(seed_modes) >= 2:
        t_bend = seed_modes[0][1]
        t_tors = seed_modes[-1][1]
    else:
        t_bend, t_tors = 9.0, 17.0

    for u, modes in struct:
        if not at_or_above_m_s(u):
            continue
        cand = [m for m in modes if m[2] >= 0.10]
        if not cand:
            cand = modes
        # bending: nearest in freq to running bending target
        if cand:
            mb = min(cand, key=lambda m: abs(m[1] - t_bend))
            if abs(mb[1] - t_bend) < 8.0:
                branches["bending"].append((u, mb[0], mb[1], mb[2]))
                t_bend = mb[1]
            mt = min(cand, key=lambda m: abs(m[1] - t_tors))
            if abs(mt[1] - t_tors) < 8.0:
                branches["torsion"].append((u, mt[0], mt[1], mt[2]))
                t_tors = mt[1]

    # Save tracked data + plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_tag = args.output_tag
    if out_tag is None:
        out_tag = f"_baseline_nm{nm}" + ("_rayleigh" if rayleigh else "")
        if v_max_kn > 32.0 + 1e-6:
            out_tag += f"_v{int(round(v_max_kn))}"
    outdir = (
        _SHARPY_ROOT.parents[1] / "output_data" / "ABRAMSON1965" / "sharpy_uvlm_water"
        / out_tag
    )
    outdir.mkdir(parents=True, exist_ok=True)

    # Eigenvectors for MAC tracking (fixed state basis across velocities).
    np.savez(
        outdir / f"abramson_sharpy_eigvecs_nm{nm}.npz",
        velocities=np.array(mac_vels),            # (nV,)
        lam=np.array(mac_lam),                    # (nV, Nstates) continuous-time
        struct_vecs=np.array(mac_vecs),           # (nV, n_beam_states, Nstates)
        participation=np.array(mac_part),         # (nV, Nstates)
        beam_idx=np.asarray(beam_idx),
        kn=KN,
    )
    print(f">>> saved eigenvectors npz for MAC tracking", flush=True)

    import csv
    # Raw dump: EVERY mode (freq 1-40 Hz) with its structural participation, so the
    # branch tracking can be re-derived offline without re-running the eig loop.
    with (outdir / f"abramson_sharpy_raw_modes_nm{nm}.csv").open("w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["V_m_s", "V_kn", "real_1_s", "freq_Hz", "participation"])
        for u, modes in struct:
            for re, fz, pa in modes:
                wri.writerow([u, u / KN, re, fz, pa])

    tracked_csv = outdir / f"abramson_sharpy_tracked_nm{nm}.csv"
    with tracked_csv.open("w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["branch", "V_m_s", "V_kn", "real_1_s", "freq_Hz", "participation"])
        for name, pts in branches.items():
            for u, re, fz, pa in pts:
                wri.writerow([name, u, u / KN, re, fz, pa])

    fig, (axg, axf) = plt.subplots(1, 2, figsize=(13, 5))
    col = {"bending": "#1f77b4", "torsion": "#ff7f0e"}
    vfl = None
    for name, pts in branches.items():
        if not pts:
            continue
        u = np.array([p[0] for p in pts])
        re = np.array([p[1] for p in pts])
        fz = np.array([p[2] for p in pts])
        axg.plot(u / KN, re, "-", color=col[name], lw=2, label=f"{name}")
        axf.plot(u / KN, fz, "-", color=col[name], lw=2, label=f"{name}")
        if name == "torsion":
            sgn = np.sign(re)
            cx = np.where(np.diff(sgn) > 0)[0]
            if len(cx):
                i = cx[0]
                a = re[i] / (re[i] - re[i + 1])
                vfl = (u[i] + a * (u[i + 1] - u[i])) / KN
                ffl = fz[i] + a * (fz[i + 1] - fz[i])

    axg.axhline(0, color="k", lw=0.8)
    axg.axvline(48.1, color="green", ls=":", lw=1.6, label="exp 48.1 kn")
    if vfl:
        axg.axvline(vfl, color="red", ls="--", lw=1.3, label=f"UVLM flutter {vfl:.1f} kn")
    axg.set_xlabel("V [kn]"); axg.set_ylabel("damping: Re(λ) [1/s]")
    axg.set_title("V-g (structural branches, participation-tracked)")
    axg.set_xlim(ABRAMSON_V_MIN_KN, v_max_kn); axg.set_ylim(-12, 8); axg.grid(alpha=0.3); axg.legend(fontsize=8)

    axf.plot(48.1, 17.5, "*", color="green", ms=15, label="exp 17.5 Hz")
    axf.axhline(20.14, color="gray", ls=":", lw=1, label="dry torsion 20.1 Hz")
    axf.axhline(9.93, color="lightgray", ls=":", lw=1, label="dry bending 9.9 Hz")
    axf.set_xlabel("V [kn]"); axf.set_ylabel("frequency [Hz]")
    axf.set_title("V-f")
    axf.set_xlim(ABRAMSON_V_MIN_KN, v_max_kn); axf.set_ylim(0, 25); axf.grid(alpha=0.3); axf.legend(fontsize=8)
    damp_note = (
        f"Rayleigh ζ=({zetas[0]:.3g}, {zetas[1]:.3g})"
        if rayleigh
        else "undamped structural beam"
    )
    fig.suptitle(
        f"Abramson SHARPy UVLM water (num_modes={nm}, {damp_note}, V≤{v_max_kn:.0f} kn): "
        f"bending vs torsion (V ≥ {ABRAMSON_V_MIN_M_S:g} m/s ≈ {ABRAMSON_V_MIN_KN:.1f} kn)"
    )
    fig.tight_layout()
    out = outdir / f"abramson_sharpy_tracked_nm{nm}.png"
    fig.savefig(out, dpi=140)
    print("saved plot:", out)
    print("saved CSV: ", tracked_csv)
    if vfl:
        print(f"flutter (torsion branch crossing): {vfl:.2f} kn @ {ffl:.2f} Hz")

    # Modal-DOF eigenvector tracking (q0=bending, q1=torsion)
    from gino_bem.modal_dof_track import plot_tracked_branches, track_branches_by_modal_dof, write_csv

    npz_path = outdir / f"abramson_sharpy_eigvecs_nm{nm}.npz"
    dof_tracked = track_branches_by_modal_dof(
        mac_vels,
        np.array(mac_lam),
        np.array(mac_vecs),
        np.array(mac_part),
        v_min_m_s=ABRAMSON_V_MIN_M_S,
    )
    dof_png = outdir / f"abramson_sharpy_modal_dof_tracked_nm{nm}.png"
    plot_tracked_branches(
        dof_tracked,
        out_png=dof_png,
        title=(
            f"Abramson SHARPy UVLM water (num_modes={nm}, {damp_note}, V≤{v_max_kn:.0f} kn) "
            "— modal DOF q₀/q₁"
        ),
        v_min_kn=ABRAMSON_V_MIN_KN,
        v_max_kn=v_max_kn,
        kn=KN,
    )
    write_csv(dof_tracked, dof_png.with_suffix(".csv"), KN)
    print("saved modal-DOF plot:", dof_png)
    print("saved modal-DOF CSV:", dof_png.with_suffix(".csv"))


if __name__ == "__main__":
    main()
