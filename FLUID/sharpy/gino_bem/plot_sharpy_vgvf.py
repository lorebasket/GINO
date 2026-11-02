#!/usr/bin/env python3
"""V–g / V–f plots with frequency-consistent branch tracking."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KN_TO_MS = 0.514444
DRY_F1, DRY_F2 = 10.0, 20.5

# Case-specific literature references for overlay lines
CASE_REFS = {
    "abramson": {"v_ref": 48.1, "v_unit": "kn", "f_ref": 17.5, "v_label": "Exp. Vf=48.1 kn"},
    "goland": {"v_ref": 166.0, "v_unit": "m/s", "f_ref": None, "v_label": "SHARPy ref. Vf≈166 m/s"},
    # Besch & Liu Fig. 11b flutter point (Model 2T config G)
    "model2t": {"v_ref": 18.1, "v_unit": "kn", "f_ref": 6.4, "v_label": "Exp. Vf=18.1 kn"},
}

# Tracked-branch settings per case: (label, f_min [Hz], max_jump [Hz], method).
# "forward": seed from the SHARPy dry modal frequency (--modal-dir) and walk up
# in V. "backward": start from the most-unstable root at V_max (unambiguously
# the flutter branch) and walk down — robust against the lightly damped wake
# roots whose frequency scales with V and pollutes dry-seeded tracking at low V.
CASE_BRANCHES = {
    "abramson": [
        ("Branch 1 (1st bending)", 5.0, 12.0, "forward"),
        ("Branch 2 (torsion)", 8.0, 12.0, "forward"),
    ],
    "model2t": [
        ("Branch 1 (1st bending)", 0.5, 4.0, "forward"),
        ("Branch 2 (torsion — flutter)", 3.0, 3.0, "backward"),
        ("Branch 3 (2nd bending)", 12.0, 10.0, "forward"),
    ],
}
CASE_FMAX = {"abramson": 35.0, "model2t": 30.0}


def _case_key_from_stem(stem: str) -> str:
    s = stem.lower()
    if "goland" in s:
        return "goland"
    if "model2t" in s:
        return "model2t"
    if "abramson" in s:
        return "abramson"
    return "abramson"


def _plot_velocity(row: dict, *, use_ms: bool) -> float:
    return float(row["V_m_s"] if use_ms else row["V_kn"])


def _oscillatory(df: pd.DataFrame, imag_min: float = 0.5) -> pd.DataFrame:
    out = df[np.abs(df["imag"]) > imag_min].copy()
    out["f_hz"] = np.abs(out["imag"]) / (2.0 * np.pi)
    out["g"] = -out["real"] / np.abs(out["imag"])
    return out


def branch_from_mode_index(df: pd.DataFrame, mode_index: int, imag_min: float = 10.0) -> list[dict]:
    sub = df[df["mode_index"] == mode_index].sort_values("V_m_s")
    pts = []
    for _, row in sub.iterrows():
        imag = float(row["imag"])
        if abs(imag) < imag_min:
            continue
        pts.append(
            {
                "V_m_s": float(row["V_m_s"]),
                "V_kn": float(row["V_kn"]),
                "f_hz": abs(imag) / (2.0 * np.pi),
                "g": -float(row["real"]) / abs(imag),
                "real": float(row["real"]),
                "imag": imag,
                "mode_index": mode_index,
            }
        )
    return pts


def track_branch_frequency(
    df: pd.DataFrame,
    *,
    seed_hz: float,
    imag_min: float = 10.0,
    f_min: float = 5.0,
    max_jump_hz: float = 12.0,
    used_by_velocity: dict[float, set[int]] | None = None,
    g_weight_hz: float = 5.0,
) -> list[dict]:
    """Track a branch by nearest frequency; optional per-V mode_index de-duplication.

    ``g_weight_hz`` penalises damping jumps (Hz-equivalent per unit g) so the
    tracker does not hop onto heavily damped lag/real-axis roots that happen to
    sit at the same frequency as the structural branch.
    """
    velocities = sorted(df["V_m_s"].unique())
    pts: list[dict] = []
    target = seed_hz
    # Seed damping continuity from the dry (V=0, g=0) point so the very first
    # velocity sample does not lock onto a heavily damped lag/aero root.
    g_prev: float | None = 0.0
    if used_by_velocity is None:
        used_by_velocity = {}

    for v in velocities:
        osc = _oscillatory(df[np.isclose(df["V_m_s"], v)], imag_min=imag_min)
        osc = osc[osc["f_hz"] >= f_min]
        used = used_by_velocity.setdefault(float(v), set())
        osc = osc[~osc["mode_index"].isin(used)]
        if osc.empty:
            continue
        osc = osc.copy()
        osc["cost"] = np.abs(osc["f_hz"] - target)
        osc = osc[osc["cost"] <= max_jump_hz]
        if osc.empty:
            continue
        if g_prev is not None and g_weight_hz > 0.0:
            osc["cost"] = osc["cost"] + g_weight_hz * np.abs(osc["g"] - g_prev)
        row = osc.loc[osc["cost"].idxmin()]
        mi = int(row["mode_index"])
        used.add(mi)
        target = float(row["f_hz"])
        g_prev = float(row["g"])
        pts.append(
            {
                "V_m_s": float(v),
                "V_kn": float(v) / KN_TO_MS,
                "f_hz": target,
                "g": float(row["g"]),
                "real": float(row["real"]),
                "imag": float(row["imag"]),
                "mode_index": mi,
            }
        )

    return pts


def track_flutter_branch_backward(
    df: pd.DataFrame,
    *,
    imag_min: float = 3.0,
    f_min: float = 0.5,
    max_jump_hz: float = 3.0,
    g_weight_hz: float = 5.0,
    used_by_velocity: dict[float, set[int]] | None = None,
) -> list[dict]:
    """
    Track the flutter branch from V_max downward. The starting root (largest
    real part at the highest velocity) is unambiguously the flutter branch;
    walking down with frequency + damping continuity avoids the wake/aero
    roots that cross the structural branches at low velocity.
    Marks both conjugates of each picked root in ``used_by_velocity``.
    """
    velocities = sorted(df["V_m_s"].unique(), reverse=True)
    if used_by_velocity is None:
        used_by_velocity = {}
    pts: list[dict] = []
    target: float | None = None
    g_prev: float | None = None

    for v in velocities:
        osc = _oscillatory(df[np.isclose(df["V_m_s"], v)], imag_min=imag_min)
        osc = osc[osc["f_hz"] >= f_min]
        used = used_by_velocity.setdefault(float(v), set())
        osc = osc[~osc["mode_index"].isin(used)]
        if osc.empty:
            continue
        osc = osc.copy()
        if target is None:
            row = osc.loc[osc["real"].idxmax()]
        else:
            osc["cost"] = np.abs(osc["f_hz"] - target)
            osc = osc[osc["cost"] <= max_jump_hz]
            if osc.empty:
                continue
            osc["cost"] = osc["cost"] + g_weight_hz * np.abs(osc["g"] - g_prev)
            row = osc.loc[osc["cost"].idxmin()]
        # claim the chosen root AND its complex conjugate
        conj = df[np.isclose(df["V_m_s"], v)]
        same = conj[
            np.isclose(np.abs(conj["imag"]), abs(float(row["imag"])))
            & np.isclose(conj["real"], float(row["real"]))
        ]
        used.update(int(mi) for mi in same["mode_index"])
        target = float(row["f_hz"])
        g_prev = float(row["g"])
        pts.append(
            {
                "V_m_s": float(v),
                "V_kn": float(v) / KN_TO_MS,
                "f_hz": target,
                "g": g_prev,
                "real": float(row["real"]),
                "imag": float(row["imag"]),
                "mode_index": int(row["mode_index"]),
            }
        )

    return list(reversed(pts))


def prepend_dry_points(branches: list[list[dict]], dry_hz: tuple[float, ...]) -> list[list[dict]]:
    out: list[list[dict]] = []
    for bi, pts in enumerate(branches):
        f0 = dry_hz[bi] if bi < len(dry_hz) else (pts[0]["f_hz"] if pts else 0.0)
        dry = {
            "V_m_s": 0.0,
            "V_kn": 0.0,
            "f_hz": f0,
            "g": 0.0,
            "real": 0.0,
            "imag": 2.0 * np.pi * f0,
            "mode_index": -1,
        }
        out.append([dry] + pts)
    return out


def _read_dry_hz(modal_dir: Path | None, n: int = 2) -> tuple[float, ...]:
    fallback = (DRY_F1, DRY_F2, 2.0 * DRY_F2)[:n]
    if modal_dir is None:
        return fallback
    p = modal_dir / "frequencies.dat"
    if not p.is_file():
        return fallback
    omega = np.atleast_1d(np.loadtxt(p)).astype(float)
    hz = sorted((omega / (2.0 * np.pi)).tolist())
    if len(hz) >= n:
        return tuple(float(x) for x in hz[:n])
    return fallback


def export_tracked_csv(branches: list[list[dict]], path: Path) -> None:
    rows = []
    for bi, pts in enumerate(branches):
        for p in pts:
            rows.append({"branch": bi + 1, **p})
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_raw_mode_indices(
    csv_path: Path,
    out_path: Path,
    *,
    mode_indices: tuple[int, ...] = (0, 1, 2, 3),
    imag_min: float = 1.0,
    title: str | None = None,
    case_key: str | None = None,
) -> Path:
    """Plot V–g / V–f for fixed SHARPy ``mode_index`` rows (no branch tracking)."""
    df = pd.read_csv(csv_path)
    ck = case_key or _case_key_from_stem(csv_path.stem)
    refs = CASE_REFS.get(ck, CASE_REFS["abramson"])
    use_ms = refs["v_unit"] == "m/s"
    v_label = "Velocity [m/s]" if use_ms else "Velocity [kn]"

    fig, (ax_g, ax_f) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    flutter_v = None
    for i, mi in enumerate(mode_indices):
        pts = branch_from_mode_index(df, mi, imag_min=imag_min)
        if not pts:
            continue
        c = colors[i % len(colors)]
        label = f"mode_index {mi}"
        v_plot = [_plot_velocity(p, use_ms=use_ms) for p in pts]
        g = [p["g"] for p in pts]
        f = [p["f_hz"] for p in pts]
        ax_g.plot(v_plot, g, "-", color=c, lw=1.5, ms=2, label=label)
        ax_f.plot(v_plot, f, "-", color=c, lw=1.5, ms=2, label=label)
        if mi == 0:
            for j in range(1, len(pts)):
                if pts[j - 1]["g"] > 0.0 >= pts[j]["g"]:
                    t = pts[j - 1]["g"] / (pts[j - 1]["g"] - pts[j]["g"])
                    flutter_v = v_plot[j - 1] + t * (v_plot[j] - v_plot[j - 1])
                    break

    ax_g.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax_g.axvline(refs["v_ref"], color="red", ls="--", lw=1.2, alpha=0.85, label=refs["v_label"])
    if refs.get("f_ref") is not None:
        ax_f.axhline(refs["f_ref"], color="red", ls="--", lw=1.2, alpha=0.85, label=f"Exp. f={refs['f_ref']} Hz")
    ax_f.axvline(refs["v_ref"], color="red", ls="--", lw=1.2, alpha=0.85)
    if flutter_v is not None:
        unit = "m/s" if use_ms else "kn"
        ax_g.axvline(flutter_v, color="purple", ls=":", lw=1.2, alpha=0.9, label=f"mode 0 g=0 ≈{flutter_v:.1f} {unit}")
        ax_f.axvline(flutter_v, color="purple", ls=":", lw=1.2, alpha=0.9)

    ax_g.set_xlabel(v_label)
    ax_g.set_ylabel("Damping g = −σ/|ω| [–]")
    ax_g.set_title("V–g — raw mode_index (no tracking)")
    ax_g.set_xlim(0, None)
    ax_g.grid(True, alpha=0.3)
    ax_g.legend(loc="best", fontsize=8)

    ax_f.set_xlabel(v_label)
    ax_f.set_ylabel("Frequency [Hz]")
    ax_f.set_title("V–f — raw mode_index (no tracking)")
    ax_f.set_xlim(0, None)
    ax_f.set_ylim(0, 45)
    ax_f.grid(True, alpha=0.3)
    ax_f.legend(loc="best", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_vgvf(
    csv_path: Path,
    out_path: Path,
    *,
    modal_dir: Path | None = None,
    title: str | None = None,
    case_key: str | None = None,
) -> Path:
    df = pd.read_csv(csv_path)
    ck = case_key or _case_key_from_stem(csv_path.stem)
    branch_specs = CASE_BRANCHES.get(ck, CASE_BRANCHES["abramson"])
    dry = _read_dry_hz(modal_dir, n=len(branch_specs))
    used: dict[float, set[int]] = {}
    # Backward-tracked branches claim their roots first so the dry-seeded
    # forward trackers cannot steal points from the flutter branch.
    order = sorted(range(len(branch_specs)), key=lambda i: branch_specs[i][3] != "backward")
    branches: list[list[dict]] = [[] for _ in branch_specs]
    for bi in order:
        _, f_min, max_jump, method = branch_specs[bi]
        if method == "backward":
            raw = track_flutter_branch_backward(
                df, f_min=f_min, max_jump_hz=max_jump, used_by_velocity=used
            )
        else:
            raw = track_branch_frequency(
                df, seed_hz=dry[bi], f_min=f_min, max_jump_hz=max_jump, used_by_velocity=used
            )
        branches[bi] = prepend_dry_points([raw], (dry[bi],))[0]

    tracked_csv = out_path.with_name(out_path.stem + "_branches.csv")
    export_tracked_csv(branches, tracked_csv)

    fig, (ax_g, ax_f) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = [spec[0] for spec in branch_specs]

    # Flutter = first g zero-crossing on ANY tracked branch
    flutter_v_kn = None
    for bi, pts in enumerate(branches):
        if not pts:
            continue
        v_kn = [p["V_kn"] for p in pts]
        g = [p["g"] for p in pts]
        f = [p["f_hz"] for p in pts]
        ax_g.plot(v_kn, g, "-", color=colors[bi % len(colors)], lw=2.0, label=labels[bi], zorder=3)
        ax_f.plot(v_kn, f, "-", color=colors[bi % len(colors)], lw=2.0, label=labels[bi], zorder=3)
        for i in range(1, len(pts)):
            g0, g1 = pts[i - 1]["g"], pts[i]["g"]
            if g0 > 0.0 >= g1:
                t = g0 / (g0 - g1)
                v_cross = pts[i - 1]["V_kn"] + t * (pts[i]["V_kn"] - pts[i - 1]["V_kn"])
                if flutter_v_kn is None or v_cross < flutter_v_kn:
                    flutter_v_kn = v_cross
                break

    refs = CASE_REFS.get(ck, CASE_REFS["abramson"])
    ax_g.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax_g.axvline(refs["v_ref"], color="red", ls="--", lw=1.2, alpha=0.85, label=refs["v_label"])
    if refs.get("f_ref") is not None:
        ax_f.axhline(refs["f_ref"], color="red", ls="--", lw=1.2, alpha=0.85, label=f"Exp. f={refs['f_ref']} Hz")
    ax_f.axvline(refs["v_ref"], color="red", ls="--", lw=1.2, alpha=0.85)
    if flutter_v_kn is not None:
        ax_g.axvline(flutter_v_kn, color="purple", ls=":", lw=1.2, alpha=0.9, label=f"SHARPy Vf≈{flutter_v_kn:.1f} kn")
        ax_f.axvline(flutter_v_kn, color="purple", ls=":", lw=1.2, alpha=0.9)

    ax_g.set_xlabel("Velocity [kn]")
    ax_g.set_ylabel("Damping g = −σ/|ω| [–]")
    ax_g.set_title("V–g (SHARPy UVLM, ρ=997)")
    ax_g.set_xlim(0, None)
    ax_g.grid(True, alpha=0.3)
    ax_g.legend(loc="best", fontsize=8)

    ax_f.set_xlabel("Velocity [kn]")
    ax_f.set_ylabel("Frequency [Hz]")
    ax_f.set_title("V–f (SHARPy UVLM, ρ=997)")
    ax_f.set_xlim(0, None)
    ax_f.set_ylim(0, CASE_FMAX.get(ck, 35.0))
    ax_f.grid(True, alpha=0.3)
    ax_f.legend(loc="best", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Tracked CSV: {tracked_csv}")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description="Plot SHARPy V-g / V-f")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--modal-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--raw-modes",
        action="store_true",
        help="Plot fixed mode_index 0..N-1 without branch tracking",
    )
    p.add_argument("--n-modes", type=int, default=4, help="Number of raw mode_index rows to plot")
    args = p.parse_args()
    stem = args.csv.stem.replace("_velocity_eigs", "")
    if "goland" in stem.lower():
        case_title = "Goland wing — SHARPy stock UVLM (raw eigenvalues)"
    elif "model2t" in stem.lower():
        case_title = "Besch & Liu Model 2T (config G) — SHARPy stock UVLM (no BEM, raw eigenvalues)"
    elif "abramson" in stem.lower():
        case_title = "Abramson 1965 — SHARPy stock UVLM (no BEM, raw eigenvalues)"
    else:
        case_title = f"{stem} — SHARPy UVLM (raw eigenvalues)"
    case_key = _case_key_from_stem(stem)
    if args.raw_modes:
        out = args.out or args.csv.with_name(f"{stem}_vg_vf_raw{args.n_modes}modes.png")
        path = plot_raw_mode_indices(
            args.csv,
            out,
            mode_indices=tuple(range(args.n_modes)),
            title=case_title,
            case_key=case_key,
        )
    else:
        out = args.out or args.csv.with_name(f"{stem}_vg_vf_combined.png")
        path = plot_vgvf(
            args.csv,
            out,
            modal_dir=args.modal_dir,
            title=case_title.replace("raw eigenvalues", "tracked branches"),
            case_key=case_key,
        )
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
