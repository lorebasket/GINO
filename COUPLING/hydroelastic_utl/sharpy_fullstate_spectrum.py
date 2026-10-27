"""
Full-state eigen sweep bridge: SHARPy LinearAssembler + GINO post-processing.

GINO's PK path uses a 2×nm modal companion ``Ac``; SHARPy exposes the full
UVLM+beam state ``ss.A``. This module runs (or loads) the SHARPy const-ω sweep
and MAC-chains coalescence partners with explicit frequency seeds for plots
compatible with ``pk_physical_spectrum.write_physical_root_artifacts``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from COUPLING.hydroelastic_utl import pk_physical_spectrum as pps

KN = 0.514444
REPO = Path(__file__).resolve().parents[2]
SHARPY_ROOT = REPO / "FLUID" / "sharpy"
DEFAULT_SHARPY_PYTHON = os.environ.get(
    "SHARPY_PYTHON",
    "/home/lorenzo/anaconda3/envs/sharpy/bin/python",
)

CASE_SPECS = {
    "model2t": {
        "script": "gino_bem/run_model2t_pk_bem.py",
        "grid_default": "n30c12",
        "npz_glob": "model2t_sharpy_eigvecs_nm*.npz",
        "tracked_glob": "model2t_pk_bem_tracked_nm*.csv",
        "const_omega_hz": 7.45,
        "seed_freq_hz": (1.86, 7.45),
        "freq_window_hz": (0.3, 15.0),
        "v_min_m_s_default": 2.0,
        "v_max_kn_default": 40.0,
    },
    "abramson": {
        "script": "gino_bem/run_abramson_pk_bem.py",
        "grid_default": "n12c16",
        "npz_glob": "abramson_sharpy_eigvecs_nm*.npz",
        "tracked_glob": "abramson_pk_bem_tracked_nm*.csv",
        "const_omega_hz": 17.5,
        "seed_freq_hz": (10.0, 20.5),
        "freq_window_hz": (5.0, 25.0),
        "v_min_m_s_default": 3.5,  # ~6.8 kn, above ABRAMSON_V_MIN_M_S
        "v_max_kn_default": 80.0,
    },
}


def _beam_participation(vec: np.ndarray, nm: int, mode_indices: tuple[int, ...]) -> float:
    qe = sum(float(abs(vec[i]) ** 2) for i in range(nm))
    if qe < 1e-30:
        return 0.0
    acc = 0.0
    for i in mode_indices:
        if 0 <= int(i) < nm:
            acc += abs(vec[int(i)]) ** 2
    return float(acc / qe)


def _pick_beam_by_frequency(
    lam: np.ndarray,
    vecs: np.ndarray,
    osc: np.ndarray,
    omega_target: float,
    *,
    exclude: set[int] | None = None,
) -> tuple[complex | None, np.ndarray | None, int | None]:
    ex = exclude or set()
    cands = [k for k in np.flatnonzero(osc) if int(k) not in ex]
    if not cands:
        return None, None, None
    om = np.imag(lam[cands])
    j = int(cands[int(np.argmin(np.abs(om - float(omega_target))))])
    return complex(lam[j]), vecs[:, j].copy(), j


def sweep_sharpy_npz_seeded_loci(
    npz_path: Path,
    *,
    seed_freq_hz: tuple[float, ...] = pps.DEFAULT_SEED_FREQ_HZ,
    freq_window_hz: tuple[float, float] = (0.3, 15.0),
    modes12_indices: tuple[int, ...] = (0, 1),
    mac_update_floor: float = 0.4,
) -> dict:
    """
    MAC-chain explicit frequency seeds on a SHARPy ``model2t_sharpy_eigvecs_nm*.npz``.

    Uses beam-state rows (``struct_vecs``) and SHARPy ``participation`` =
    ‖beam‖ / ‖full state‖ per root.
    """
    d = np.load(Path(npz_path))
    V_list = np.asarray(d["velocities"], float).ravel()
    lam_all = np.asarray(d["lam"], dtype=object)
    vecs_all = np.asarray(d["struct_vecs"], dtype=object)
    part_all = np.asarray(d["participation"], dtype=object)
    nm = int(vecs_all.shape[1] // 2)

    chains: list[dict] = []
    seeded = False

    for iv, V in enumerate(V_list):
        lam = np.asarray(lam_all[iv], dtype=complex).ravel()
        vecs = np.asarray(vecs_all[iv], dtype=complex)
        part_v = np.asarray(part_all[iv], dtype=float).ravel()
        freq = np.imag(lam) / (2.0 * np.pi)
        osc = (
            (np.imag(lam) > 1e-6)
            & (freq >= freq_window_hz[0])
            & (freq <= freq_window_hz[1])
        )

        if not seeded:
            used: set[int] = set()
            for bi, f_hz in enumerate(seed_freq_hz):
                om_tgt = float(f_hz) * 2.0 * np.pi
                lam0, vec0, j0 = _pick_beam_by_frequency(lam, vecs, osc, om_tgt, exclude=used)
                if lam0 is None or vec0 is None or j0 is None:
                    chains.append(
                        dict(
                            branch_id=int(bi + 1),
                            name=f"Branch{bi + 1}",
                            seed_f_hz=float(f_hz),
                            seed_omega=om_tgt,
                            omega_ref=om_tgt,
                            ref=None,
                            rows=[],
                            lost=True,
                        )
                    )
                    continue
                used.add(int(j0))
                f_pick = float(np.imag(lam0)) / (2.0 * np.pi)
                if abs(f_pick - f_hz) > 0.05:
                    print(
                        f"[fullstate] branch {bi + 1}: target {f_hz:.2f} Hz → "
                        f"picked {f_pick:.2f} Hz"
                    )
                chains.append(
                    dict(
                        branch_id=int(bi + 1),
                        name=f"Branch{bi + 1}",
                        seed_f_hz=float(f_hz),
                        seed_f_pick_hz=f_pick,
                        seed_omega=om_tgt,
                        omega_ref=float(np.imag(lam0)),
                        ref=vec0.copy(),
                        rows=[],
                        lost=False,
                    )
                )
            seeded = True
            ok = [c for c in chains if c.get("ref") is not None]
            print(
                f"[fullstate] seeded at V={V / KN:.2f} kn: "
                + ", ".join(
                    f"Branch {c['branch_id']} {c['omega_ref'] / (2 * np.pi):.2f} Hz"
                    for c in ok
                )
            )

        for c in chains:
            if c.get("lost", False):
                pps._append_lost_row(c, V, nm)
                continue

            if c["ref"] is None:
                lam_p, vec_p, _j = _pick_beam_by_frequency(lam, vecs, osc, c["seed_omega"])
                mac = 1.0
            else:
                lam_p, vec_p, mac = pps._track_root(
                    c, lam, vecs, osc, V, nm, 0.0,
                    mac_update_floor=mac_update_floor,
                    mac_terminate_floor=0.0,
                )
                if c.get("lost", False):
                    continue

            if lam_p is None or vec_p is None:
                pps._append_lost_row(c, V, nm)
                continue

            omega = float(np.imag(lam_p))
            j_part = int(np.argmin(np.abs(np.imag(lam) - omega)))
            part_sh = float(part_v[j_part]) if j_part < part_v.size else float("nan")
            c["omega_ref"] = omega
            if np.isfinite(mac) and mac >= mac_update_floor:
                c["ref"] = vec_p.copy()
            c["rows"].append(
                dict(
                    V_kn=float(V) / KN,
                    V_m_s=float(V),
                    f_Hz=omega / (2.0 * np.pi),
                    sigma=float(np.real(lam_p)),
                    g=float(np.real(lam_p) / max(abs(omega), 1e-30)),
                    part=part_sh,
                    part_bend=pps._mode_pair_participation(vec_p, nm, 0),
                    part_modes12=_beam_participation(vec_p, nm, modes12_indices),
                    mac=float(mac) if np.isfinite(mac) else np.nan,
                    k=float(omega / max(float(V), 1e-6)),
                    q=pps._modal_content(vec_p, nm),
                )
            )

    return {
        "velocities": V_list,
        "chains": chains,
        "nm": nm,
        "k_policy": "fullstate_sharpy",
        "modes12_indices": modes12_indices,
        "seed_freq_hz": seed_freq_hz,
        "npz_source": str(npz_path),
    }


def run_sharpy_const_a_sweep(
    outdir: Path,
    *,
    case: str = "model2t",
    v_min_m_s: float | None = None,
    v_max_kn: float | None = None,
    v_step_kn: float = 0.5,
    const_omega_hz: float | None = None,
    grid: str | None = None,
    sharpy_python: str | Path = DEFAULT_SHARPY_PYTHON,
    nm: int = 4,
    tag: str = "gino_fullstate",
    no_bem: bool = False,
    bem_mode: str = "replace",
    added_mass_removal: str = "gamma_dot",
) -> Path:
    """Invoke SHARPy ``--const-a-omega`` full-state sweep for *case*.

    Default hydro coupling is *replace*: strip UVLM apparent mass
    (``added_mass_removal``) and inject the full Capytaine ``A(ω)``.
    Pass ``bem_mode='delta'`` only for the legacy UVLM-intact offset injection.
    """
    if case not in CASE_SPECS:
        raise ValueError(f"unknown case {case!r}; choose from {sorted(CASE_SPECS)}")
    spec = CASE_SPECS[case]

    # Resolve to absolute: the SHARPy subprocess below runs with cwd=SHARPY_ROOT,
    # so a relative outdir would silently land under sharpy/ instead of here.
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    py = Path(sharpy_python)
    if not py.is_file():
        raise FileNotFoundError(
            f"SHARPy python not found: {py}. Set SHARPY_PYTHON or install sharpy env."
        )
    script = SHARPY_ROOT / spec["script"]
    if not script.is_file():
        raise FileNotFoundError(script)

    if v_min_m_s is None:
        v_min_m_s = float(spec["v_min_m_s_default"])
    if v_max_kn is None:
        v_max_kn = float(spec["v_max_kn_default"])
    if const_omega_hz is None:
        const_omega_hz = float(spec["const_omega_hz"])
    if grid is None:
        grid = str(spec["grid_default"])

    omega_star = float(const_omega_hz) * 2.0 * np.pi
    v_min_kn = float(v_min_m_s) / KN
    cmd = [
        str(py),
        str(script),
        "--grid", str(grid),
        "--num-modes", str(int(nm)),
        "--v-min-kn", f"{v_min_kn:.6f}",
        "--v-max-kn", f"{float(v_max_kn):.6f}",
        "--v-step-kn", f"{float(v_step_kn):.6f}",
        "--const-a-omega", f"{omega_star:.8f}",
        "--outdir", str(outdir),
        "--tag", str(tag),
    ]
    if no_bem:
        cmd.extend(["--no-bem", "--added-mass-removal", "none"])
    else:
        mode = str(bem_mode).lower()
        if mode not in ("replace", "delta"):
            raise ValueError(f"bem_mode must be 'replace' or 'delta', got {bem_mode!r}")
        cmd.extend(["--bem-mode", mode])
        if mode == "replace":
            cmd.extend(["--added-mass-removal", str(added_mass_removal)])
        # delta forces added_mass_removal=none inside run_model2t_pk_bem.py
    print(f"[fullstate:{case}] running SHARPy full-state const-A sweep:")
    print("  ", " ".join(cmd))
    subprocess.run(cmd, cwd=str(SHARPY_ROOT), check=True)

    matches = sorted(outdir.glob(spec["npz_glob"]))
    if not matches:
        raise FileNotFoundError(f"No {spec['npz_glob']} in {outdir}")
    return matches[-1]
