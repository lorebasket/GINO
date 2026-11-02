"""
Full-spectrum eigenvalue sweep for PK physical-root loci (modal participation).

Sweep policies (``k_policy``):

* ``sharpy_const_omega`` — SHARPy const-A twin: one ``eig(Ac)`` per V at
  ``k = ω* / V`` (Capytaine frozen at ω*), two branches seeded at target
  frequencies (default 1.86 + 7.45 Hz) and MAC-chained.
* ``pk_converged`` — per-branch ``eig(Ac)`` at PK-converged ``k_b(V)`` loaded
  from ``eigenvalue_data.csv`` in the flutter outdir.
* ``per_branch`` — per-branch ``k = ω_branch / V`` with MAC-updated ω.
* ``torsion_over_v`` — legacy single-k at ``ω_torsion,dry / V``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import scipy.linalg as la

KN = 0.514444
MODE_NAMES = ("1B", "1T", "2B", "m4")
# SHARPy delta hybrid coalescence partner seeds at V = 2 kn (physical_root_loci.csv)
DEFAULT_SEED_FREQ_HZ = (1.86, 7.45)
DEFAULT_CONST_OMEGA_HZ = 7.45


def _mac(a: np.ndarray, b: np.ndarray) -> float:
    num = abs(np.vdot(a, b)) ** 2
    den = float(np.vdot(a, a).real) * float(np.vdot(b, b).real)
    return num / den if den > 1e-30 else 0.0


def _modal_content(vec: np.ndarray, nm: int) -> np.ndarray:
    q = np.abs(vec[:nm])
    return q / max(float(q.max()), 1e-300)


def _mode_pair_participation(vec: np.ndarray, nm: int, mode_idx: int) -> float:
    """Displacement energy share of one modal DOF (q_m only; excludes qd_m)."""
    qe = sum(float(abs(vec[i]) ** 2) for i in range(nm))
    if qe < 1e-30:
        return 0.0
    i = int(mode_idx)
    if i < 0 or i >= nm:
        return 0.0
    return float(abs(vec[i]) ** 2 / qe)


def _modes12_participation(vec: np.ndarray, nm: int, mode_indices: tuple[int, ...]) -> float:
    """Combined 1B+1T displacement modal energy (beam-equivalent; q half only)."""
    qe = sum(float(abs(vec[i]) ** 2) for i in range(nm))
    if qe < 1e-30:
        return 0.0
    acc = 0.0
    for i in mode_indices:
        if 0 <= int(i) < nm:
            acc += abs(vec[int(i)]) ** 2
    return float(acc / qe)


def _dominant_modal_fraction(vec: np.ndarray, nm: int) -> float:
    q = _modal_content(vec, nm)
    return float(q.max())


def load_pk_branch_omega_table(
    out_dir: Path,
    *,
    n_branches: int = 2,
) -> dict[float, list[float]]:
    """Load PK converged ω(V) per branch slot from ``eigenvalue_data.csv``."""
    path = Path(out_dir) / "eigenvalue_data.csv"
    if not path.is_file():
        raise FileNotFoundError(f"PK eigenvalue_data.csv not found in {out_dir}")
    by_v: dict[float, dict[int, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            v = float(row["Velocity_m_s"])
            mode = int(row["Mode"])
            omega = float(row["Imag_Part_lambda_rad_s"])
            if mode >= n_branches:
                continue
            by_v.setdefault(v, {})[mode] = omega
    table: dict[float, list[float]] = {}
    for v, modes in sorted(by_v.items()):
        om = [float("nan")] * n_branches
        for m, w in modes.items():
            if 0 <= m < n_branches:
                om[m] = w
        table[v] = om
    return table


def _nearest_pk_omega(table: dict[float, list[float]], V: float, branch: int) -> float:
    if not table:
        return float("nan")
    keys = np.asarray(sorted(table.keys()), float)
    i = int(np.argmin(np.abs(keys - float(V))))
    om = table[float(keys[i])][branch]
    return float(om)


def _eigenspectrum(
    build_fn,
    matrix_iter_state: dict,
    k: float,
    V: float,
    nm: int,
    *,
    freq_window_hz: tuple[float, float],
    mode_s: int = 0,
):
    matrix_iter_state["s"] = int(mode_s)
    matrix_iter_state["j"] = 0
    matrix_iter_state["p"] = None
    Ac = np.asarray(build_fn(float(k), float(V)), dtype=complex)
    vals, vecs = la.eig(Ac)
    order = np.argsort(-np.imag(vals))
    vals = vals[order]
    vecs = vecs[:, order]
    freq = np.imag(vals) / (2.0 * np.pi)
    osc = (
        (np.imag(vals) > 1e-6)
        & (freq >= freq_window_hz[0])
        & (freq <= freq_window_hz[1])
    )
    return vals, vecs, osc, freq


def _pick_by_frequency(
    vals: np.ndarray,
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
    om = np.imag(vals[cands])
    j = int(cands[int(np.argmin(np.abs(om - float(omega_target))))])
    return complex(vals[j]), vecs[:, j].copy(), j


def _pick_by_mac(
    vals: np.ndarray,
    vecs: np.ndarray,
    osc: np.ndarray,
    ref: np.ndarray,
) -> tuple[complex | None, np.ndarray | None, int | None, float]:
    macs = np.array([_mac(ref, vecs[:, k]) if osc[k] else -1.0 for k in range(vecs.shape[1])])
    j = int(np.argmax(macs))
    if macs[j] < 0.0:
        return None, None, None, float("nan")
    return complex(vals[j]), vecs[:, j].copy(), j, float(macs[j])


def _append_lost_row(c: dict, V: float, nm: int, *, mac=np.nan, k=np.nan) -> None:
    c["rows"].append(
        dict(
            V_kn=float(V) / KN,
            V_m_s=float(V),
            f_Hz=np.nan,
            sigma=np.nan,
            g=np.nan,
            part=np.nan,
            part_bend=np.nan,
            mac=float(mac) if np.isfinite(mac) else np.nan,
            k=float(k) if np.isfinite(k) else np.nan,
            q=np.full(nm, np.nan),
        )
    )


def _append_chain_row(c: dict, V: float, nm: int, lam: complex, vec: np.ndarray, mac: float, k: float, modes12_indices: tuple[int, ...]) -> None:
    omega = float(np.imag(lam))
    c["rows"].append(
        dict(
            V_kn=float(V) / KN,
            V_m_s=float(V),
            f_Hz=omega / (2.0 * np.pi),
            sigma=float(np.real(lam)),
            g=float(np.real(lam) / max(abs(omega), 1e-30)),
            part=_modes12_participation(vec, nm, modes12_indices),
            part_bend=_mode_pair_participation(vec, nm, 0),
            mac=float(mac) if np.isfinite(mac) else np.nan,
            k=float(k),
            q=_modal_content(vec, nm),
        )
    )


def _init_chains_from_targets(
    seed_freq_hz: tuple[float, ...],
    vals: np.ndarray,
    vecs: np.ndarray,
    osc: np.ndarray,
    *,
    match_tol_frac: float = 0.25,
) -> list[dict]:
    """Seed MAC chains at target frequencies; fall back to two lowest roots."""
    chains: list[dict] = []
    used: set[int] = set()
    osc_idx = sorted(int(k) for k in np.flatnonzero(osc))

    for bi, f_hz in enumerate(seed_freq_hz):
        om_tgt = float(f_hz) * 2.0 * np.pi
        lam, vec, j = _pick_by_frequency(vals, vecs, osc, om_tgt, exclude=used)
        ok = False
        if lam is not None and j is not None and vec is not None:
            om_pick = float(np.imag(lam))
            if abs(om_pick - om_tgt) / max(om_tgt, 1e-6) <= match_tol_frac:
                ok = True
        if not ok:
            lam, vec, j = None, None, None

        if lam is None:
            for kk in osc_idx:
                if int(kk) in used:
                    continue
                lam, vec, j = complex(vals[kk]), vecs[:, kk].copy(), int(kk)
                break

        if lam is None or vec is None or j is None:
            print(f"[physical_sweep] WARNING: no oscillatory root for branch {bi + 1}")
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
                    seed_index=None,
                )
            )
            continue

        used.add(int(j))
        f_pick = float(np.imag(lam)) / (2.0 * np.pi)
        if abs(f_pick - f_hz) > 0.05:
            print(
                f"[physical_sweep] branch {bi + 1}: target {f_hz:.2f} Hz → "
                f"picked {f_pick:.2f} Hz (index {j})"
            )
        chains.append(
            dict(
                branch_id=int(bi + 1),
                name=f"Branch{bi + 1}",
                seed_f_hz=float(f_hz),
                seed_f_pick_hz=f_pick,
                seed_omega=om_tgt,
                omega_ref=float(np.imag(lam)),
                ref=vec.copy(),
                rows=[],
                lost=False,
                seed_index=int(j),
            )
        )
    return chains


def _mac_terminate_or_record(
    c: dict,
    V: float,
    nm: int,
    mac: float,
    k: float,
    *,
    mac_terminate_floor: float,
) -> bool:
    """Return True if branch is lost this step."""
    if np.isfinite(mac) and mac < mac_terminate_floor and c["ref"] is not None:
        print(
            f"[physical_sweep] Branch {c['branch_id']} lost at "
            f"V={V / KN:.2f} kn (MAC={mac:.3f} < {mac_terminate_floor})"
        )
        c["lost"] = True
        _append_lost_row(c, V, nm, mac=mac, k=k)
        return True
    return False


def _track_root(
    c: dict,
    vals: np.ndarray,
    vecs: np.ndarray,
    osc: np.ndarray,
    V: float,
    nm: int,
    k: float,
    *,
    mac_update_floor: float,
    mac_terminate_floor: float,
) -> tuple[complex | None, np.ndarray | None, float]:
    """MAC-track; fall back to frequency continuity before declaring branch lost."""
    if c["ref"] is None:
        lam, vec, _j = _pick_by_frequency(vals, vecs, osc, c["seed_omega"])
        return lam, vec, 1.0

    lam, vec, _j, mac = _pick_by_mac(vals, vecs, osc, c["ref"])
    if lam is not None and np.isfinite(mac) and mac >= mac_terminate_floor:
        return lam, vec, mac

    lam_f, vec_f, _j = _pick_by_frequency(vals, vecs, osc, c["omega_ref"])
    if lam_f is not None and vec_f is not None:
        mac_f = _mac(c["ref"], vec_f)
        return lam_f, vec_f, mac_f

    if lam is not None and vec is not None:
        return lam, vec, float(mac) if np.isfinite(mac) else float("nan")

    print(
        f"[physical_sweep] Branch {c['branch_id']} lost at V={V / KN:.2f} kn "
        "(no oscillatory root in window)"
    )
    c["lost"] = True
    _append_lost_row(c, V, nm, mac=mac if np.isfinite(mac) else np.nan, k=k)
    return None, None, float("nan")


def sweep_sharpy_comparable_loci(
    build_fn,
    matrix_iter_state: dict,
    V_list: np.ndarray,
    nm: int,
    *,
    seed_freq_hz: tuple[float, ...] = DEFAULT_SEED_FREQ_HZ,
    const_omega_hz: float = DEFAULT_CONST_OMEGA_HZ,
    pk_omega_table: dict[float, list[float]] | None = None,
    freq_window_hz: tuple[float, float] = (0.3, 15.0),
    modes12_indices: tuple[int, ...] = (0, 1),
    mac_update_floor: float = 0.4,
    mac_terminate_floor: float = 0.35,
    k_policy: str = "sharpy_const_omega",
) -> dict:
    """
    Two-branch MAC loci comparable to SHARPy ``physical_root_loci_modes12``.

    Seeds roots closest to ``seed_freq_hz`` (default 1.86 + 7.45 Hz).
    Participation uses 1B+1T modal energy (``part``); ``part_bend`` is 1B only.
    """
    V_list = np.asarray(V_list, dtype=float).ravel()
    policy = str(k_policy).lower()
    use_pk_k = policy == "pk_converged"
    seed_desc = ", ".join(f"{f:.2f} Hz" for f in seed_freq_hz)
    if use_pk_k:
        k_desc = "k_b(V)=omega_PK/V from eigenvalue_data.csv"
    else:
        k_desc = f"k=omega_branch/V (Capytaine at omega_branch; target seeds {seed_desc})"
    print(
        f"[physical_sweep] SHARPy-comparable: {len(seed_freq_hz)} branches, "
        f"seeds=[{seed_desc}], {k_desc}, freq_window={freq_window_hz}"
    )

    chains: list[dict] = []
    seeded = False

    for iv, V in enumerate(V_list):
        # per-branch k: pk_converged or MAC-chained omega_branch / V
        if not seeded:
            chains = [
                dict(
                    branch_id=int(bi + 1),
                    name=f"Branch{bi + 1}",
                    seed_f_hz=float(f_hz),
                    seed_omega=float(f_hz) * 2.0 * np.pi,
                    omega_ref=float(f_hz) * 2.0 * np.pi,
                    ref=None,
                    rows=[],
                    lost=False,
                )
                for bi, f_hz in enumerate(seed_freq_hz)
            ]
            seeded = True

        for c in chains:
            if c.get("lost", False):
                _append_lost_row(c, V, nm)
                continue

            if use_pk_k and pk_omega_table is not None:
                om_pk = _nearest_pk_omega(pk_omega_table, V, c["branch_id"] - 1)
                if np.isfinite(om_pk):
                    k = max(om_pk / float(V), 1e-6)
                elif c["ref"] is None:
                    k = max(float(c["seed_omega"]) / float(V), 1e-6)
                else:
                    # PK branch lost (e.g. 1B overdamped) — keep MAC chain at ω_branch/V.
                    k = max(float(c["omega_ref"]) / float(V), 1e-6)
            elif c["ref"] is None:
                k = max(float(c["seed_omega"]) / float(V), 1e-6)
            else:
                k = max(float(c["omega_ref"]) / float(V), 1e-6)

            vals, vecs, osc, _ = _eigenspectrum(
                build_fn, matrix_iter_state, k, V, nm,
                freq_window_hz=freq_window_hz,
                mode_s=min(c["branch_id"] - 1, nm - 1),
            )

            if c["ref"] is None:
                lam, vec, _j = _pick_by_frequency(vals, vecs, osc, c["seed_omega"])
                mac = 1.0
            else:
                lam, vec, mac = _track_root(
                    c, vals, vecs, osc, V, nm, k,
                    mac_update_floor=mac_update_floor,
                    mac_terminate_floor=mac_terminate_floor,
                )
                if c.get("lost", False):
                    continue

            if lam is None or vec is None:
                _append_lost_row(c, V, nm, k=k)
                continue

            c["omega_ref"] = float(np.imag(lam))
            if np.isfinite(mac) and mac >= mac_update_floor:
                c["ref"] = vec.copy()
            _append_chain_row(c, V, nm, lam, vec, mac, k, modes12_indices)

        if iv == 0:
            ok = [c for c in chains if c["rows"] and np.isfinite(c["rows"][-1]["f_Hz"])]
            print(
                f"[physical_sweep] first V={V / KN:.2f} kn: "
                + ", ".join(
                    f"Branch {c['branch_id']} {c['rows'][-1]['f_Hz']:.2f} Hz "
                    f"(k={c['rows'][-1]['k']:.2f})"
                    for c in ok
                )
            )

    return {
        "velocities": V_list,
        "chains": chains,
        "nm": nm,
        "k_policy": policy,
        "modes12_indices": modes12_indices,
        "seed_freq_hz": seed_freq_hz,
        "const_omega_hz": const_omega_hz,
    }


def sweep_coalescence_loci(
    build_fn,
    matrix_iter_state: dict,
    V_list: np.ndarray,
    nm: int,
    omega_dry: np.ndarray,
    *,
    seed_mode_indices: tuple[int, ...] = (0, 1),
    freq_window_hz: tuple[float, float] = (0.3, 15.0),
    modes12_indices: tuple[int, ...] = (0, 1),
    mac_update_floor: float = 0.4,
    mac_terminate_floor: float = 0.35,
) -> dict:
    """Legacy per-branch sweep using dry-mode seeds (not SHARPy frequency targets)."""
    seed_hz = tuple(
        float(omega_dry[int(mi)]) / (2.0 * np.pi) if int(mi) < omega_dry.size else float("nan")
        for mi in seed_mode_indices
    )
    return sweep_sharpy_comparable_loci(
        build_fn,
        matrix_iter_state,
        V_list,
        nm,
        seed_freq_hz=seed_hz,
        freq_window_hz=freq_window_hz,
        modes12_indices=modes12_indices,
        mac_update_floor=mac_update_floor,
        mac_terminate_floor=mac_terminate_floor,
        k_policy="per_branch",
    )


def sweep_physical_roots(
    build_fn,
    matrix_iter_state: dict,
    V_list: np.ndarray,
    nm: int,
    omega_dry: np.ndarray,
    *,
    part_floor: float = 1e-4,
    freq_window_hz: tuple[float, float] = (0.3, 35.0),
    chain_mode_indices: tuple[int, ...] = (0, 1),
    torsion_mode: int = 1,
    k_policy: str = "sharpy_const_omega",
    seed_mode_indices: tuple[int, ...] | None = None,
    seed_freq_hz: tuple[float, ...] | None = None,
    const_omega_hz: float = DEFAULT_CONST_OMEGA_HZ,
    pk_omega_table: dict[float, list[float]] | None = None,
    mac_update_floor: float = 0.4,
) -> dict:
    """Dispatch physical-root sweep policy."""
    policy = str(k_policy).lower()
    seeds_hz = seed_freq_hz if seed_freq_hz is not None else DEFAULT_SEED_FREQ_HZ
    win_hi = min(freq_window_hz[1], 15.0)

    if policy in ("sharpy_const_omega", "const_omega", "pk_converged"):
        return sweep_sharpy_comparable_loci(
            build_fn,
            matrix_iter_state,
            V_list,
            nm,
            seed_freq_hz=seeds_hz,
            const_omega_hz=const_omega_hz,
            pk_omega_table=pk_omega_table,
            freq_window_hz=(freq_window_hz[0], win_hi),
            modes12_indices=chain_mode_indices,
            mac_update_floor=mac_update_floor,
            k_policy=policy,
        )

    if policy in ("per_branch", "coalescence"):
        smi = seed_mode_indices if seed_mode_indices is not None else (0, 1)
        return sweep_coalescence_loci(
            build_fn,
            matrix_iter_state,
            V_list,
            nm,
            omega_dry,
            seed_mode_indices=smi,
            freq_window_hz=(freq_window_hz[0], win_hi),
            modes12_indices=chain_mode_indices,
            mac_update_floor=mac_update_floor,
        )

    # legacy torsion_over_v
    V_list = np.asarray(V_list, dtype=float).ravel()
    omega_dry = np.asarray(omega_dry, dtype=float).ravel()
    omega_track = float(omega_dry[torsion_mode]) if torsion_mode < omega_dry.size else float(omega_dry[-1])

    spectra_lam: list[np.ndarray] = []
    spectra_vec: list[np.ndarray] = []
    spectra_part: list[np.ndarray] = []
    chains: list[dict] = []
    seeded = False

    for iv, V in enumerate(V_list):
        k = max(omega_track / float(V), 1e-6)
        vals, vecs, osc, _ = _eigenspectrum(
            build_fn, matrix_iter_state, k, V, nm,
            freq_window_hz=freq_window_hz, mode_s=int(torsion_mode),
        )
        part = np.array([_dominant_modal_fraction(vecs[:, kk], nm) for kk in range(vecs.shape[1])])
        spectra_lam.append(vals)
        spectra_vec.append(vecs)
        spectra_part.append(part)

        if not seeded:
            seed_idx = np.flatnonzero(osc & (part >= part_floor))
            if seed_idx.size == 0:
                seed_idx = np.flatnonzero(osc)
            seed_idx = sorted(seed_idx, key=lambda kk: np.imag(vals[kk]))
            for kk in seed_idx:
                q = _modal_content(vecs[:, kk], nm)
                name = MODE_NAMES[int(np.argmax(q))] if int(np.argmax(q)) < len(MODE_NAMES) else f"m{int(np.argmax(q))}"
                chains.append(dict(name=name, mode_idx=int(np.argmax(q)), ref=vecs[:, kk].copy(), rows=[]))
            seeded = True
            print(f"[physical_sweep] legacy seeded {len(chains)} chains at V={V / KN:.2f} kn")

        for c in chains:
            lam, vec, _j, mac = _pick_by_mac(vals, vecs, osc, c["ref"])
            if lam is None:
                continue
            q = _modal_content(vec, nm)
            c["rows"].append(
                dict(
                    V_kn=float(V) / KN,
                    V_m_s=float(V),
                    f_Hz=float(np.imag(lam) / (2.0 * np.pi)),
                    sigma=float(np.real(lam)),
                    g=float(np.real(lam) / max(abs(np.imag(lam)), 1e-30)),
                    part=float(part[_j]) if _j is not None else np.nan,
                    mac=float(mac),
                    q=q,
                )
            )
            if mac > mac_update_floor:
                c["ref"] = vec.copy()

        if osc.any():
            torsion_cands = np.flatnonzero(osc)
            k_t = int(torsion_cands[np.argmax(part[torsion_cands])])
            omega_track = 0.7 * omega_track + 0.3 * float(np.imag(vals[k_t]))

    return {
        "velocities": V_list,
        "lam": np.array(spectra_lam, dtype=object),
        "struct_vecs": np.array(spectra_vec, dtype=object),
        "participation": np.array(spectra_part, dtype=object),
        "chains": chains,
        "nm": nm,
        "k_policy": "torsion_over_v",
    }


def write_physical_root_artifacts(
    sweep: dict,
    out_dir: Path,
    *,
    chain_mode_indices: tuple[int, ...] = (0, 1),
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chains = sweep["chains"]
    nm = int(sweep["nm"])
    policy = str(sweep.get("k_policy", "")).lower()
    sharpy_style = policy in (
        "sharpy_const_omega", "const_omega", "pk_converged", "per_branch",
        "coalescence", "fullstate_sharpy",
    )

    names_keep = {MODE_NAMES[i] for i in chain_mode_indices if i < len(MODE_NAMES)}
    chains_all = chains
    if sharpy_style:
        chains_m12 = list(chains)
    else:
        chains_m12 = [c for c in chains if c.get("name") in names_keep]

    csv_path = out_dir / "physical_root_loci.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        hdr = ["branch", "V_kn", "f_Hz", "sigma", "g", "part", "mac"]
        if sharpy_style:
            hdr.extend(["part_bend", "k", "branch_id", "seed_f_hz"])
        w.writerow(hdr)
        for c in chains_all:
            for r in c["rows"]:
                row = [
                    c.get("name", f"Branch{c.get('branch_id', 0)}"),
                    f"{r['V_kn']:.3f}",
                    f"{r['f_Hz']:.5f}" if np.isfinite(r["f_Hz"]) else "",
                    f"{r['sigma']:.5f}" if np.isfinite(r.get("sigma", np.nan)) else "",
                    f"{r['g']:.6f}" if np.isfinite(r.get("g", np.nan)) else "",
                    f"{r['part']:.3e}" if np.isfinite(r.get("part", np.nan)) else "",
                    f"{r['mac']:.4f}" if np.isfinite(r.get("mac", np.nan)) else "",
                ]
                if sharpy_style:
                    row.extend(
                        [
                            f"{r.get('part_bend', float('nan')):.3e}"
                            if np.isfinite(r.get("part_bend", np.nan))
                            else "",
                            f"{r.get('k', float('nan')):.5f}" if np.isfinite(r.get("k", np.nan)) else "",
                            int(c.get("branch_id", 0)),
                            f"{c.get('seed_f_hz', float('nan')):.3f}",
                        ]
                    )
                w.writerow(row)

    npz_path = out_dir / f"gino_physical_roots_nm{nm}.npz"
    save_kw = dict(
        velocities=np.asarray(sweep["velocities"], float),
        nm=nm,
        k_policy=str(sweep.get("k_policy", "")),
        seed_freq_hz=np.asarray(sweep.get("seed_freq_hz", DEFAULT_SEED_FREQ_HZ), float),
    )
    if "const_omega_hz" in sweep:
        save_kw["const_omega_hz"] = float(sweep["const_omega_hz"])
    for i, c in enumerate(chains):
        rows = c["rows"]
        save_kw[f"branch{i + 1}_V_kn"] = np.array([r["V_kn"] for r in rows], float)
        save_kw[f"branch{i + 1}_f_Hz"] = np.array([r["f_Hz"] for r in rows], float)
        save_kw[f"branch{i + 1}_g"] = np.array([r["g"] for r in rows], float)
        save_kw[f"branch{i + 1}_part"] = np.array([r.get("part", np.nan) for r in rows], float)
        save_kw[f"branch{i + 1}_part_bend"] = np.array([r.get("part_bend", np.nan) for r in rows], float)
    np.savez_compressed(npz_path, **save_kw)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    branch_col = ["#2a78d6", "#eda100", "#1baf7a", "#c33d69"]
    col = {"1B": "#2a78d6", "1T": "#eda100", "2B": "#1baf7a", "m4": "#c33d69"}

    def _chain_color(c, idx):
        if sharpy_style:
            return branch_col[idx % len(branch_col)]
        return col.get(c.get("name", ""), "k")

    def _chain_label(c):
        if sharpy_style and "seed_f_hz" in c:
            return f"Branch {c['branch_id']} (seed {c['seed_f_hz']:.2f} Hz)"
        return c.get("name", "?")

    def _plot_loci(subchains, png_name, title_suffix, *, part_key="part", part_ylabel=None):
        fig, (axf, axg, axp) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        for idx, c in enumerate(subchains):
            rows_ok = [r for r in c["rows"] if np.isfinite(r["f_Hz"])]
            if not rows_ok:
                continue
            v = [r["V_kn"] for r in rows_ok]
            color = _chain_color(c, idx)
            lab = _chain_label(c)
            axf.plot(v, [r["f_Hz"] for r in rows_ok], "-", color=color, lw=1.8, label=lab)
            axg.plot(v, [r["g"] for r in rows_ok], "-", color=color, lw=1.8)
            pvals = [max(r.get(part_key, r["part"]), 1e-7) for r in rows_ok]
            axp.semilogy(v, pvals, "-", color=color, lw=1.8)
        axg.axhline(0, color="k", lw=0.8)
        axp.axhline(1e-4, color="gray", ls=":", lw=1, label="part floor 1e-4")
        axp.axhline(1e-3, color="k", ls=":", lw=1, label="tracker floor 1e-3")
        if sharpy_style:
            axp.axhline(2e-4, color="gray", ls=":", lw=1, alpha=0.7)
        axf.set_ylabel("frequency [Hz]")
        axg.set_ylabel("g = σ/ω [–]")
        axp.set_ylabel(part_ylabel or "dominant modal fraction [–]")
        axp.set_xlabel("V [kn]")
        axf.legend(fontsize=9, ncol=2)
        axp.legend(fontsize=8)
        for ax in (axf, axg, axp):
            ax.grid(alpha=0.3)
        title = "GINO PK — physical root MAC loci"
        if policy == "fullstate_sharpy":
            title = "Model2t delta hybrid — full-state loci (SHARPy ss.A, GINO post-proc)"
        elif sharpy_style:
            title = "Model2t delta hybrid — filter-free loci of the two physical roots (GINO)"
        fig.suptitle(f"{title} ({title_suffix})")
        fig.tight_layout()
        fig.savefig(out_dir / png_name, dpi=150)
        plt.close(fig)

    if policy == "fullstate_sharpy":
        part_key = "part"
        part_ylabel = "beam participation [–]"
    elif sharpy_style:
        part_key = "part_bend"
        part_ylabel = "1B modal share (q) [–]"
    else:
        part_key = "part"
        part_ylabel = None
    _plot_loci(chains_all, "physical_root_loci.png", "all branches", part_key=part_key, part_ylabel=part_ylabel)
    _plot_loci(
        chains_m12,
        "physical_root_loci_modes12.png",
        "modes 1B + 1T coalescence partners",
        part_key=part_key,
        part_ylabel=part_ylabel,
    )

    fig, axes = plt.subplots(nm, 1, figsize=(10, 2.2 * nm), sharex=True)
    if nm == 1:
        axes = [axes]
    for idx, c in enumerate(chains_m12):
        rows_ok = [r for r in c["rows"] if np.isfinite(r["f_Hz"])]
        if not rows_ok:
            continue
        v = [r["V_kn"] for r in rows_ok]
        color = _chain_color(c, idx)
        lab = _chain_label(c)
        for m in range(nm):
            axes[m].plot(
                v,
                [r["q"][m] for r in rows_ok],
                "-",
                color=color,
                lw=1.6,
                label=lab if m == 0 else None,
            )
    for m, ax in enumerate(axes):
        ax.set_ylabel(f"|q_{MODE_NAMES[m]}|")
        ax.grid(alpha=0.3)
        if m == 0:
            ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("V [kn]")
    fig.suptitle("GINO — modal content of coalescence MAC chains")
    fig.tight_layout()
    fig.savefig(out_dir / "root_modal_content_modes12.png", dpi=150)
    plt.close(fig)

    print(f"[physical_sweep] wrote {out_dir / 'physical_root_loci_modes12.png'}")
    print(f"[physical_sweep] wrote {out_dir / 'root_modal_content_modes12.png'}")
    print(f"[physical_sweep] wrote {csv_path}")
    print(f"[physical_sweep] wrote {npz_path}")
