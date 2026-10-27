#!/usr/bin/env python3
"""
Model 2T (Besch & Liu config G) — hybrid SHARPy UVLM + Capytaine BEM flutter
with per-mode p-k. Model2t twin of ``run_abramson_pk_bem.py``.

Removes the UVLM apparent-mass terms (``--added-mass-removal gamma_dot``, GINO
patch in ``sharpy/linear/src/linuvlm.py``) and injects the frequency-dependent
Capytaine modal added mass A(omega) / radiation damping B(omega) — the summed
strut+pod superposition matrices — into the linear beam (``capy_bem_bridge``),
iterating each tracked branch to frequency consistency (p-k).

Stiffness is NOMINAL (GJ, EI factors = 1), matching the GINO
``_baseline_pk_capytaine_aerob2removal`` baseline (V_f = 28.39 kn @ 6.83 Hz)
and the ``_baseline_sharpy`` stock-UVLM line (20.25 kn @ 6.93 Hz).

``--bem-mode delta`` instead keeps the UVLM fully intact (its forward-speed
non-circulatory damping included) and injects only the *difference*
A_capy(omega) - A_uvlm, where the UVLM apparent mass is measured by a
gamma_dot probe case and given strip-theory coupling structure over the
wetted span only (Model2t strut pierces the surface; EA at 69% chord,
a = +0.38 semichords).

  conda activate sharpy
  cd sharpy
  python gino_bem/run_model2t_pk_bem.py --dry-check
  python gino_bem/run_model2t_pk_bem.py --added-mass-removal gamma_dot \
      --outdir <...>/_baseline_sharpy_removegammadot_plusbem
  python gino_bem/run_model2t_pk_bem.py --bem-mode delta \
      --outdir <...>/_baseline_sharpy_deltainjection_plusbem
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
_FSI_ROOT = _SHARPY_ROOT.parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))

from gino_bem.model2t_strut import CHORD, WETTED_FROM_ROOT, XEA, Model2tStrut  # noqa: E402
from gino_bem import run_model2t_water as R  # noqa: E402
from gino_bem.run_model2t_water import (  # noqa: E402
    MODEL2T_RAYLEIGH_MODE_IDS,
    MODEL2T_RAYLEIGH_ZETAS,
)
from gino_bem.rayleigh_damping import install_rayleigh_structural_damping  # noqa: E402
from gino_bem.run_abramson_pk_bem import (  # noqa: E402
    BRANCH_NAMES,
    _eig_at,
    build_delta_offset,
    dry_check,
    flutter_crossing,
    pk_sweep_bem,
)
from gino_bem.capy_bem_bridge import check_capy_basis, load_capytaine_npz  # noqa: E402
from gino_bem.mode_alignment import align_capy_to_sharpy  # noqa: E402
import sharpy.sharpy_main  # noqa: E402

KN = 0.514444
V_EXP_KN, F_EXP_HZ = 18.1, 6.4          # Besch & Liu Fig. 11b, config G
V_GINO_KN = 28.39                        # _baseline_pk_capytaine_aerob2removal (nominal)
FREQ_WINDOW_HZ = (0.5, 30.0)             # Model2t wet band (1B ~1.9 Hz .. 2B ~25 Hz)
# (M chord, N span, Ms wake); N multiple of 10 (waterline node + even elements).
GRIDS = {
    "coarse": (6, 10, 4),
    "default": (8, 20, 6),
    # converged production grid (01_discretization_convergence_analysis)
    "n30c12": (12, 30, 6),
}


def default_model2t_capy_npz() -> Path:
    """Summed strut+pod superposition matrices (run_pod_superposition.py)."""
    capy_dir = _FSI_ROOT / "FLUID" / "capytaine" / "Model2t" / "results_modal_radiation"
    for name in ("modal_radiation_AB_depth_0.npz", "modal_radiation_AB.npz"):
        p = capy_dir / name
        if p.is_file():
            print(f"[model2t_pk_bem] using Capytaine NPZ: {p.name}")
            return p
    raise FileNotFoundError(f"no modal_radiation_AB*.npz in {capy_dir}")


# --------------------------------------------------------------------------- case build
def build_case(nm: int, *, grid: tuple[int, int, int], added_mass_removal: str,
               rayleigh: bool, zetas, v_max_kn: float, case_tag: str):
    M, N, Ms = grid
    if N % 10:
        raise ValueError(f"N must be a multiple of 10 (waterline node), got {N}")
    route = _SHARPY_ROOT / "gino_bem"
    (route / "cases").mkdir(parents=True, exist_ok=True)
    out_root = route / "output"
    out_root.mkdir(exist_ok=True)
    ws = Model2tStrut(
        M=M, N=N, Mstar_fact=Ms, u_inf=1.0, alpha=0.0, rho=997.0,
        gj_factor=1.0, ei_factor=1.0,
        route=str(route / "cases"), case_name=case_tag,
    )
    ws.clean_test_files()
    ws.update_derived_params()
    ws.set_default_config_dict()
    ws.generate_aero_file()
    ws.generate_fem_file()
    vel = R._model2t_velocity_list_m_s(v_max_m_s=v_max_kn * KN)
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
            mode_ids=MODEL2T_RAYLEIGH_MODE_IDS, target_zetas=zetas,
        )
        print(f">>> Rayleigh damping: zeta={zetas} on modes "
              f"{list(MODEL2T_RAYLEIGH_MODE_IDS)}", flush=True)
    print(f">>> UVLM added_mass_removal = '{added_mass_removal}'  grid M,N,Ms = {grid}  "
          "(nominal stiffness)", flush=True)
    data = sharpy.sharpy_main.main(["", ws.route + ws.case_name + ".sharpy"])
    return data, vel


def wetted_span_mask(data) -> np.ndarray:
    """Boolean mask over the free-node span stations (sharpy_modal_displacements
    order): True where the strut is submerged. The dry root span is the
    WETTED_FROM_ROOT arc length from the clamped node; the nearest-node
    waterline convention matches Model2tStrut.generate_aero_file (node at
    s=0.2524 is wet for N=30)."""
    pos = np.asarray(data.linear.tsstruct0.pos, dtype=float)
    bcs = np.asarray(data.structure.boundary_conditions).ravel()
    free = bcs != 1
    span_axis = int(np.argmax(pos.max(axis=0) - pos.min(axis=0)))
    s_root = float(pos[~free, span_axis][0])
    s = np.abs(pos[free, span_axis] - s_root)
    return s >= (WETTED_FROM_ROOT - 0.01)


# --------------------------------------------------------------------------- step reporter
MODE_LABELS = ("1B", "1T", "2B", "m4")


def _mac_c(a: np.ndarray, b: np.ndarray) -> float:
    num = abs(np.vdot(a, b)) ** 2
    den = np.vdot(a, a).real * np.vdot(b, b).real
    return float(num / max(den, 1e-300))


class RootPoolReporter:
    """STEP E/F-style per-velocity report: top-N root pool, MAC continuity,
    new-root detection. Unmatched ("new") roots are kept for post-processing."""

    def __init__(self, nm: int, n_v: int, *, n_pool: int = 10, part_floor: float = 1e-4,
                 freq_window_hz=FREQ_WINDOW_HZ, mac_parent_floor: float = 0.5):
        self.nm = nm
        self.n_v = n_v
        self.n_pool = n_pool
        self.part_floor = part_floor
        self.window = freq_window_hz
        self.mac_parent_floor = mac_parent_floor
        self.prev_pool: list[dict] = []
        self.pool_rows: list[dict] = []
        self.new_rows: list[dict] = []

    def __call__(self, iv: int, V: float, branch_results: dict, last_res: dict):
        lam = np.asarray(last_res["lam_full"])
        vecs = np.asarray(last_res["vec_beam_full"])   # (n_beam, n_eig)
        part = np.asarray(last_res["part_full"])
        freq = lam.imag / (2 * np.pi)
        cand = np.flatnonzero((lam.imag > 1e-6)
                              & (freq >= self.window[0]) & (freq <= self.window[1])
                              & (part >= self.part_floor))
        pool_idx = cand[np.argsort(-part[cand])][: self.n_pool]
        pool_idx = pool_idx[np.argsort(lam[pool_idx].imag)]

        print(f"\n--- VELOCITY STEP {iv + 1}/{self.n_v}: V = {V:.2f} m/s "
              f"({V / KN:.1f} kn) ---", flush=True)
        print(f"  STEP E: root pool ({len(pool_idx)} roots, top-{self.n_pool} by "
              f"participation, part >= {self.part_floor:g}, "
              f"{self.window[0]}-{self.window[1]} Hz)")
        pool = []
        for k in pool_idx:
            q = np.abs(vecs[: self.nm, k])
            q = q / max(q.max(), 1e-300)
            g = lam[k].real / max(abs(lam[k].imag), 1e-300)
            best_mac, parent = 0.0, None
            for p in self.prev_pool:
                m = _mac_c(p["vec"], vecs[:, k])
                if m > best_mac:
                    best_mac, parent = m, p
            e = dict(V=V, f_hz=float(freq[k]), g=float(g), part=float(part[k]),
                     q=q, vec=vecs[:, k].copy(), mac_parent=best_mac,
                     parent_f=None if parent is None else parent["f_hz"])
            pool.append(e)
            qs = " ".join(f"{MODE_LABELS[m]}={q[m]:.2f}" for m in range(self.nm))
            print(f"    f={e['f_hz']:8.4f} Hz  g={e['g']:+9.5f}  "
                  f"part={e['part']:.5f}  [{qs}]")
            self.pool_rows.append(dict(
                V_m_s=float(V), V_kn=float(V) / KN, f_Hz=e["f_hz"], g=e["g"],
                part=e["part"],
                **{f"q_{MODE_LABELS[m]}": float(q[m]) for m in range(self.nm)},
                dominant=MODE_LABELS[int(np.argmax(q))],
                mac_parent=best_mac,
                parent_f_Hz=e["parent_f"] if e["parent_f"] is not None else float("nan"),
            ))

        print("  STEP F: MAC continuity vs previous velocity pool")
        for e in pool:
            if iv == 0 or not self.prev_pool:
                print(f"    f={e['f_hz']:8.4f} Hz : first velocity (no parent)")
            elif e["mac_parent"] >= self.mac_parent_floor:
                df = (100 * (e["f_hz"] - e["parent_f"]) / max(e["parent_f"], 1e-9)
                      if e["parent_f"] else 0.0)
                print(f"    ✓ f={e['f_hz']:8.4f} Hz : MAC={e['mac_parent']:.3f}, "
                      f"f_prev={e['parent_f']:.4f}→{e['f_hz']:.4f} Hz ({df:+.1f}%)")
            else:
                print(f"    [INFO] NEW root: f={e['f_hz']:8.4f} Hz, "
                      f"best_MAC={e['mac_parent']:.3f} — saved for post-processing")
                self.new_rows.append(self.pool_rows[len(self.pool_rows) - len(pool)
                                                    + pool.index(e)])
        for j, res in sorted(branch_results.items()):
            print(f"  tracked {BRANCH_NAMES.get(j, j):>8s}: f={res['freq_Hz']:8.4f} Hz  "
                  f"g={res['g']:+9.5f}  MAC={res['mac']:.3f}  part={res['part']:.5f}")
        self.prev_pool = pool

    def save(self, outdir: Path):
        for fn, rows in (("root_pool_all.csv", self.pool_rows),
                         ("new_roots.csv", self.new_rows)):
            p = outdir / fn
            with p.open("w", newline="") as f:
                if rows:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
            print(f"  {fn}: {len(rows)} rows -> {p}")


# --------------------------------------------------------------------------- const-A eigensweep
def eig_sweep_const_A(ls, capy, T, v_list, *, nm, omega_star, reporter,
                      track_modes=(0, 1), seed_part_floor=5e-4,
                      freq_window_hz=FREQ_WINDOW_HZ, A_offset=None):
    """Plain eigenvalue sweep with A(omega*) frozen — no p-k iteration.

    The UVLM is a state-space LTI model (no frequency dependence); the only
    omega-dependent input is the Capytaine A(omega), which is nearly flat for
    Model2t. Freezing it at omega* makes each velocity ONE eigensolve giving
    ALL roots simultaneously; branches are recovered afterwards by MAC-chaining
    over the full spectrum (no participation floor, no seeds, no branch loss).
    """
    from gino_bem.track_structural_modes import beam_state_indices

    beam_idx = beam_state_indices(ls.ss, nm)
    rows = []
    spec_v, spec_lam, spec_vec, spec_part = [], [], [], []
    chains: list[dict] = []          # [{name, ref (beam vec), rows}]
    for iv, V in enumerate(np.asarray(v_list, dtype=float).ravel()):
        lam, Vec, _ = _eig_at(ls, float(V), capy=capy, T=T, omega=float(omega_star),
                              n_modes=nm, bem=True, A_offset=A_offset)
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
            print(f">>> const-A eigensweep: A frozen at omega* = {omega_star:g} rad/s; "
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
        if reporter is not None:
            reporter(iv, float(V), {}, dict(lam_full=lam, vec_beam_full=vec_beam,
                                            part_full=part))
    return rows, (spec_v, spec_lam, spec_vec, spec_part, beam_idx)


# --------------------------------------------------------------------------- outputs
def write_outputs(rows, spectra, outdir: Path, nm: int, tag: str, track_modes,
                  contributions_history=None):
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"model2t_pk_bem_tracked_nm{nm}.csv"
    with csv_path.open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=["branch", "V_m_s", "V_kn", "g", "freq_Hz",
                                            "mac", "part", "n_iter", "converged"])
        wri.writeheader()
        for r in rows:
            wri.writerow(r)

    spec_v, spec_lam, spec_vec, spec_part, beam_idx = spectra
    if spec_v:
        np.savez(
            outdir / f"model2t_sharpy_eigvecs_nm{nm}.npz",
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
    for j in track_modes:
        name = BRANCH_NAMES.get(j, str(j))
        pts = sorted(((r["V_kn"], r["g"], r["freq_Hz"]) for r in rows if r["branch"] == name))
        if not pts:
            continue
        v, g, fz = map(np.array, zip(*pts))
        axg.plot(v, g, ".-", ms=3, color=col.get(name, None), label=name)
        axf.plot(v, fz, ".-", ms=3, color=col.get(name, None), label=name)
        vf, ff = flutter_crossing(rows, name)
        summary[name] = (vf, ff)
        if vf is not None:
            axg.axvline(vf / KN, color=col.get(name, "k"), ls=":", lw=1)
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
    fig.suptitle(f"Model 2T (config G) — SHARPy p-k + Capytaine BEM ({tag})")
    fig.tight_layout()
    fig.savefig(outdir / f"model2t_pk_bem_vgvf_nm{nm}.png", dpi=150)
    plt.close(fig)

    if contributions_history:
        from gino_bem.contributions_plot import plot_stiffness_damping_contributions

        plot_stiffness_damping_contributions(
            contributions_history, outdir, v_knots=False, show_plot=False,
        )

    return csv_path, summary


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--added-mass-removal", choices=["none", "gamma_dot", "full"],
                    default="gamma_dot")
    ap.add_argument("--bem-mode", choices=["replace", "delta"], default="replace",
                    help="replace: remove UVLM apparent mass, inject A_capy. "
                         "delta: keep UVLM intact (removal forced to 'none'), inject "
                         "A_capy - A_uvlm so the UVLM keeps its forward-speed "
                         "non-circulatory damping")
    ap.add_argument("--strip-a", type=float, default=2.0 * (XEA - 0.5),
                    help="EA position in semichords aft of midchord for the delta "
                         "strip coupling structure (Model2t: 69%% chord -> +0.38)")
    ap.add_argument("--delta-v-probe", type=str, default="0.5,1.0,1.5",
                    help="comma-separated probe velocities [m/s] for the gamma_dot "
                         "apparent-mass measurement (delta mode only): per mode the "
                         "best-MAC pairing wins, values above the 2D strip bound are "
                         "rejected as contaminated. Keep speeds low — circulatory "
                         "stiffening corrupts the pairing (4.5 m/s fails on Model2t)")
    ap.add_argument("--no-bem", action="store_true", help="skip Capytaine injection")
    ap.add_argument("--num-modes", type=int, default=4)
    ap.add_argument("--track-modes", type=str, default="0,1")
    ap.add_argument("--v-min-kn", type=float, default=2.0)
    ap.add_argument("--v-max-kn", type=float, default=40.0)
    ap.add_argument("--v-step-kn", type=float, default=0.5)
    ap.add_argument("--grid", type=str, default="n30c12", choices=sorted(GRIDS.keys()))
    ap.add_argument("--outdir", type=Path, default=None, help="output directory for CSV/PNG/NPZ")
    ap.add_argument("--capy-npz", type=Path, default=None)
    ap.add_argument("--no-rayleigh", action="store_true")
    ap.add_argument("--zeta-1", type=float, default=MODEL2T_RAYLEIGH_ZETAS[0])
    ap.add_argument("--zeta-2", type=float, default=MODEL2T_RAYLEIGH_ZETAS[1])
    ap.add_argument("--relax", type=float, default=0.618)
    ap.add_argument("--tol", type=float, default=1e-3)
    # Model2t: with the water added mass injected, physical wet roots carry only
    # ~0.2–1% beam-state participation (aero gamma states dominate), while the
    # spurious fixed-scaled-frequency Newmark beam root has part ~1. The Abramson
    # default floor (0.01) filters out the PHYSICAL roots here.
    ap.add_argument("--part-floor", type=float, default=1e-3)
    ap.add_argument("--n-pool", type=int, default=10,
                    help="roots reported per velocity step (STEP E/F root pool)")
    ap.add_argument("--const-a-omega", type=float, default=None, metavar="OMEGA",
                    help="Freeze Capytaine A at this omega [rad/s] and do a plain "
                    "eigenvalue sweep (no p-k iteration): one eigensolve per "
                    "velocity, all roots at once, branches MAC-chained over the "
                    "full spectrum. Justified because the UVLM is LTI and A(omega) "
                    "is nearly flat for Model2t.")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--dry-check", action="store_true",
                    help="stop after LinearAssembler; print patch + alignment report")
    args = ap.parse_args()

    nm = args.num_modes
    grid = GRIDS[args.grid]
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
    tag = args.tag or f"{mode_tag}_{'bem' if bem else 'nobem'}_nm{nm}_{args.grid}"

    capy = check_capy_basis(load_capytaine_npz(args.capy_npz or default_model2t_capy_npz()))

    data, _ = build_case(
        nm, grid=grid, added_mass_removal=removal,
        rayleigh=not args.no_rayleigh, zetas=(args.zeta_1, args.zeta_2),
        v_max_kn=args.v_max_kn, case_tag=f"model2t_pk_{tag}",
    )
    ls = data.linear.linear_system

    T = dry_check(data, ls, capy, nm) if args.dry_check else align_capy_to_sharpy(capy, data, nm)[0]
    A_offset = None
    if args.bem_mode == "delta":
        # Model2t physical wet roots carry only ~0.05-1% beam participation and
        # the spurious Newmark beam root sits near part ~1 — hence the low probe
        # floor and the ceiling. m4 may be unpairable behind spurious roots, so
        # missing diagonals fall back to ratio-scaled strip values.
        A_offset = build_delta_offset(
            data, capy, T, nm, ls, grid=grid, rayleigh=not args.no_rayleigh,
            zetas=(args.zeta_1, args.zeta_2), v_max_kn=args.v_max_kn,
            a_ea=args.strip_a,
            v_probe=[float(x) for x in args.delta_v_probe.split(",") if x.strip()],
            build_case_fn=build_case, probe_tag="model2t_pk_probe_gd",
            strip_b=CHORD / 2.0, wetted_mask=wetted_span_mask(data),
            probe_part_floor=1e-4, probe_part_ceiling=0.5,
            require_all_modes=False,
        )
    if args.dry_check:
        print("=== dry check done (no sweep) ===")
        return 0

    kn = np.arange(float(args.v_min_kn), float(args.v_max_kn) + 0.5 * float(args.v_step_kn),
                   float(args.v_step_kn))
    v_list = kn * KN
    print(f">>> velocity sweep: {len(v_list)} points, "
          f"{args.v_min_kn:g}–{args.v_max_kn:g} kn every {args.v_step_kn:g} kn", flush=True)

    reporter = RootPoolReporter(nm, len(v_list), n_pool=args.n_pool,
                                part_floor=min(args.part_floor, 1e-4))
    if args.const_a_omega is not None:
        rows, spectra = eig_sweep_const_A(
            ls, capy, T, v_list, nm=nm, omega_star=args.const_a_omega,
            reporter=reporter, A_offset=A_offset,
        )
        contributions_history = None
        track_modes = tuple(range(nm))     # plot every MAC-chained branch
        tag += f"_constA_w{args.const_a_omega:g}"
    else:
        rows, spectra, contributions_history = pk_sweep_bem(
            ls, capy, T, v_list, nm=nm, track_modes=track_modes, bem=bem,
            A_offset=A_offset,
            relax=args.relax, tol_rel=args.tol, freq_window_hz=FREQ_WINDOW_HZ,
            part_floor=args.part_floor, step_report=reporter,
        )
    outdir = args.outdir or (
        _FSI_ROOT / "output_data" / "Model2t" / "00_baseline_sensitivity"
        / (("_baseline_sharpy_deltainjection_plusbem" if args.bem_mode == "delta"
            else "_baseline_sharpy_removegammadot_plusbem")
           + (f"_constA_w{args.const_a_omega:g}" if args.const_a_omega is not None else ""))
    )
    outdir.mkdir(parents=True, exist_ok=True)
    reporter.save(outdir)
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
