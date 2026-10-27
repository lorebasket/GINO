#!/usr/bin/env python3
"""
Track Abramson SHARPy flutter branches by **modal DOF content** in eigenvectors.

Each branch follows modal index ``m`` (q_m, qdot_m): at every velocity the mode is
the eigenvector with the strongest MAC on the 2-DOF slice [q_m, qdot_m], with a
frequency-continuity penalty so wake modes cannot hijack the branch.

  python gino_bem/modal_dof_track.py --npz <eigvecs.npz> --out <png>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))

import numpy as np

from gino_bem.abramson_tracking_config import ABRAMSON_V_MIN_KN, at_or_above_m_s

V_EXP_KN, F_EXP_HZ = 48.1, 17.5
DEFAULT_BRANCHES = (
    (0, "bending", 4.0, 11.0),
    (1, "torsion", 13.0, 22.0),
)


def modal_dof_slice(vec: np.ndarray, dof: int, nm: int) -> np.ndarray:
    """2-vector [q_dof, qdot_dof] extracted from the structural eigenvector block."""
    return np.array([vec[dof], vec[nm + dof]], dtype=complex)


def modal_dof_fraction(vec: np.ndarray, dof: int, nm: int) -> float:
    """Share of structural-block energy in (q_dof, qdot_dof)."""
    e = float(np.vdot(vec, vec).real)
    if e < 1e-30:
        return 0.0
    q = vec[dof]
    qd = vec[nm + dof]
    return float((abs(q) ** 2 + abs(qd) ** 2) / e)


def mac(a: np.ndarray, b: np.ndarray) -> float:
    num = np.abs(np.vdot(a, b)) ** 2
    den = float(np.vdot(a, a).real) * float(np.vdot(b, b).real)
    return num / den if den > 1e-30 else 0.0


def dof_mac(vec_a: np.ndarray, vec_b: np.ndarray, dof: int, nm: int) -> float:
    return mac(modal_dof_slice(vec_a, dof, nm), modal_dof_slice(vec_b, dof, nm))


def _branch_score(
    *,
    frac: float,
    m_dof: float,
    freq_hz: float,
    f_target: float | None,
    freq_sigma_hz: float,
) -> float:
    score = 0.55 * frac + 0.45 * m_dof
    if f_target is not None and freq_sigma_hz > 0:
        score -= ((freq_hz - f_target) / freq_sigma_hz) ** 2
    return score


def track_branches_by_modal_dof(
    velocities: np.ndarray,
    lam: np.ndarray,
    struct_vecs: np.ndarray,
    participation: np.ndarray,
    *,
    branches: Sequence[tuple[int, str, float, float]] = DEFAULT_BRANCHES,
    v_min_m_s: float | None = None,
    f_min_hz: float = 0.5,
    f_max_hz: float = 40.0,
    dof_frac_floor: float = 0.15,
    part_floor: float = 0.0,
    freq_sigma_hz: float = 3.0,
) -> dict[str, list[dict]]:
    """
    Return per-branch lists of dicts with keys V_m_s, real_1_s, freq_Hz, mode_index,
    dof_fraction, participation, mac_to_prev.
    """
    order = np.argsort(velocities)
    vels = np.asarray(velocities, dtype=float)[order]
    lam = np.asarray(lam)[order]
    SV = np.asarray(struct_vecs)[order]
    P = np.asarray(participation)[order]

    nV, Ns = lam.shape
    nb = SV.shape[1]
    nm = nb // 2
    freq = np.abs(lam.imag) / (2.0 * np.pi)

    if v_min_m_s is None:
        v_min_m_s = ABRAMSON_V_MIN_KN * 0.514444

    out: dict[str, list[dict]] = {name: [] for _, name, _, _ in branches}
    prev_vec: dict[str, np.ndarray] = {}
    f_target: dict[str, float] = {}

    for i in range(nV):
        v = float(vels[i])
        if v < v_min_m_s - 1e-9:
            continue

        used: set[int] = set()
        for dof, name, f_lo, f_hi in branches:
            if dof >= nm:
                continue
            osc = [
                k
                for k in range(Ns)
                if lam[i, k].imag > 1e-6
                and max(f_lo, f_min_hz) <= freq[i, k] <= min(f_hi, f_max_hz)
                and (part_floor <= 0.0 or P[i, k] >= part_floor)
                and k not in used
            ]
            if not osc:
                continue

            scored: list[tuple[float, float, float, int]] = []
            for k in osc:
                frac = modal_dof_fraction(SV[i, :, k], dof, nm)
                if frac < dof_frac_floor:
                    continue
                m_d = dof_mac(prev_vec[name], SV[i, :, k], dof, nm) if name in prev_vec else frac
                sc = _branch_score(
                    frac=frac,
                    m_dof=m_d,
                    freq_hz=float(freq[i, k]),
                    f_target=f_target.get(name),
                    freq_sigma_hz=freq_sigma_hz,
                )
                scored.append((sc, frac, m_d, k))
            if not scored:
                for k in osc:
                    frac = modal_dof_fraction(SV[i, :, k], dof, nm)
                    m_d = dof_mac(prev_vec[name], SV[i, :, k], dof, nm) if name in prev_vec else frac
                    sc = _branch_score(
                        frac=frac,
                        m_dof=m_d,
                        freq_hz=float(freq[i, k]),
                        f_target=f_target.get(name),
                        freq_sigma_hz=freq_sigma_hz,
                    )
                    scored.append((sc, frac, m_d, k))
            if not scored:
                continue
            scored.sort(reverse=True, key=lambda t: t[0])
            _, frac, m_d, k = scored[0]
            used.add(k)
            m_prev_val = dof_mac(prev_vec[name], SV[i, :, k], dof, nm) if name in prev_vec else 1.0
            prev_vec[name] = SV[i, :, k].copy()
            f_target[name] = float(freq[i, k])
            out[name].append(
                {
                    "V_m_s": v,
                    "real_1_s": float(lam[i, k].real),
                    "freq_Hz": float(freq[i, k]),
                    "mode_index": int(k),
                    "dof_fraction": float(frac),
                    "dof_mac": float(m_d),
                    "participation": float(P[i, k]),
                    "mac_to_prev": float(m_prev_val),
                }
            )

    return out


def plot_tracked_branches(
    tracked: dict[str, list[dict]],
    *,
    out_png: Path,
    title: str,
    v_min_kn: float,
    v_max_kn: float,
    kn: float,
) -> tuple[float | None, float | None, str | None]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = {"bending": "#1f77b4", "torsion": "#ff7f0e"}
    fig, (axg, axf) = plt.subplots(1, 2, figsize=(13, 5))
    vfl = None

    for name, pts in tracked.items():
        if not pts:
            continue
        v = np.array([p["V_m_s"] for p in pts])
        re = np.array([p["real_1_s"] for p in pts])
        fz = np.array([p["freq_Hz"] for p in pts])
        axg.plot(v / kn, re, "-", color=cols.get(name, "k"), lw=2, label=name)
        axf.plot(v / kn, fz, "-", color=cols.get(name, "k"), lw=2, label=name)
        cx = np.where(np.diff(np.sign(re)) > 0)[0]
        if len(cx) and name == "torsion":
            j = int(cx[0])
            a = re[j] / (re[j] - re[j + 1])
            vc = (v[j] + a * (v[j + 1] - v[j])) / kn
            fc = fz[j] + a * (fz[j + 1] - fz[j])
            if vfl is None or vc < vfl[0]:
                vfl = (vc, fc, name)

    axg.axhline(0, color="k", lw=0.8)
    axg.axvline(V_EXP_KN, color="green", ls=":", lw=1.6, label=f"exp {V_EXP_KN} kn")
    if vfl:
        axg.axvline(vfl[0], color="red", ls="--", lw=1.3, label=f"flutter {vfl[0]:.1f} kn ({vfl[2]})")
    axg.set_xlabel("V [kn]")
    axg.set_ylabel("damping: Re(λ) [1/s]")
    axg.set_title("V-g (modal DOF q₀ / q₁ tracked)")
    axg.set_xlim(v_min_kn, v_max_kn)
    axg.set_ylim(-500, 50)
    axg.grid(alpha=0.3)
    axg.legend(fontsize=8)

    axf.plot(V_EXP_KN, F_EXP_HZ, "*", color="green", ms=15, label=f"exp {F_EXP_HZ} Hz")
    axf.axhline(20.14, color="gray", ls=":", lw=1, label="dry torsion 20.1 Hz")
    axf.axhline(9.93, color="lightgray", ls=":", lw=1, label="dry bending 9.9 Hz")
    axf.set_xlabel("V [kn]")
    axf.set_ylabel("frequency [Hz]")
    axf.set_title("V-f (modal DOF tracked)")
    axf.set_xlim(v_min_kn, v_max_kn)
    axf.set_ylim(0, 25)
    axf.grid(alpha=0.3)
    axf.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return vfl


def write_csv(tracked: dict[str, list[dict]], path: Path, kn: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(
            [
                "branch",
                "modal_dof",
                "V_m_s",
                "V_kn",
                "real_1_s",
                "freq_Hz",
                "mode_index",
                "dof_fraction",
                "dof_mac",
                "participation",
                "mac_to_prev",
            ]
        )
        for dof, name, _, _ in DEFAULT_BRANCHES:
            for p in tracked.get(name, []):
                wri.writerow(
                    [
                        name,
                        dof,
                        p["V_m_s"],
                        p["V_m_s"] / kn,
                        p["real_1_s"],
                        p["freq_Hz"],
                        p["mode_index"],
                        p["dof_fraction"],
                        p.get("dof_mac", ""),
                        p["participation"],
                        p["mac_to_prev"],
                    ]
                )


def main() -> int:
    ap = argparse.ArgumentParser(description="Track SHARPy branches by modal DOF in eigenvectors")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--v-min-kn", type=float, default=ABRAMSON_V_MIN_KN)
    ap.add_argument("--v-max-kn", type=float, default=150.0)
    ap.add_argument("--dof-frac-floor", type=float, default=0.15,
                    help="min fraction of structural energy in target q_m")
    ap.add_argument("--part-floor", type=float, default=0.0,
                    help="optional global participation floor")
    ap.add_argument("--freq-sigma-hz", type=float, default=5.0,
                    help="Gaussian penalty width for frequency continuity")
    ap.add_argument("--title", type=str, default="Abramson SHARPy UVLM (modal DOF tracked)")
    args = ap.parse_args()

    d = np.load(args.npz)
    kn = float(d["kn"])
    tracked = track_branches_by_modal_dof(
        d["velocities"],
        d["lam"],
        d["struct_vecs"],
        d["participation"],
        v_min_m_s=args.v_min_kn * kn,
        dof_frac_floor=args.dof_frac_floor,
        part_floor=args.part_floor,
        freq_sigma_hz=args.freq_sigma_hz,
    )

    for name, pts in tracked.items():
        if pts:
            print(
                f"{name}: {len(pts)} points, "
                f"V=[{pts[0]['V_m_s']/kn:.1f}, {pts[-1]['V_m_s']/kn:.1f}] kn, "
                f"mean dof_frac={np.mean([p['dof_fraction'] for p in pts]):.3f}"
            )

    out = args.out or args.npz.with_name(
        args.npz.stem.replace("eigvecs", "modal_dof_tracked") + ".png"
    )
    vfl = plot_tracked_branches(
        tracked,
        out_png=out,
        title=args.title,
        v_min_kn=args.v_min_kn,
        v_max_kn=args.v_max_kn,
        kn=kn,
    )
    write_csv(tracked, out.with_suffix(".csv"), kn)
    print("saved plot:", out)
    print("saved CSV:", out.with_suffix(".csv"))
    if vfl:
        print(f"flutter (modal DOF): {vfl[0]:.2f} kn @ {vfl[1]:.2f} Hz ({vfl[2]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
