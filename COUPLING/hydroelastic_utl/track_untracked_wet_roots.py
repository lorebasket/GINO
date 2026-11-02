"""
MAC-based multi-branch tracker for the untracked wet roots.

Reads ``pk_untracked_wet_roots_vectors.csv`` (written by
``post_processing.plot_pk_converged_wet_eigenvectors`` once the flutter analysis
is re-run with eigenvector export) and stitches the per-velocity roots into
continuous physical branches using shape (MAC) + frequency continuity, instead
of the unstable ``spectrum_index`` grouping that produces the vertical jumps in
``pk_untracked_wet_roots.png``.

At each airspeed every active branch is matched to the unused root that minimises

    cost = w_mac * (1 - MAC(u_branch, u_root)) + w_freq * |Δω| / ω

subject to a relative frequency-jump guard. Roots that no branch can claim seed
new branches; branches that go unmatched are closed. Short branches (fewer than
``--min-points`` samples) are dropped as noise.

Usage
-----
    python -m COUPLING.hydroelastic_utl.track_untracked_wet_roots \
        --dir output_data/Model2t/00_baseline_sensitivity/_pk_capytaine_gj0464_finev_2mode

Outputs (in the same directory):
    pk_untracked_wet_roots_tracked.csv   branch_id per (V, root)
    pk_untracked_wet_roots_tracked.png   omega(V), g(V) coloured by branch
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy is expected but stay robust
    _HAVE_SCIPY = False


def _compute_mac(u1: np.ndarray, u2: np.ndarray, n_structural_dof=None) -> float:
    """MAC between two complex vectors (phase/scale invariant), local copy."""
    if u1 is None or u2 is None:
        return 0.0
    u1 = np.asarray(u1, dtype=complex).ravel()
    u2 = np.asarray(u2, dtype=complex).ravel()
    if n_structural_dof:
        n = int(n_structural_dof)
        u1 = u1[:n] if u1.size >= n else u1
        u2 = u2[:n] if u2.size >= n else u2
    num = np.abs(np.vdot(u1, u2)) ** 2
    den = np.vdot(u1, u1).real * np.vdot(u2, u2).real
    if den < 1e-30:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def _load_roots(csv_path: str, kinds: set[str] | None):
    """Return dict {V: list of root dicts} with reconstructed complex vectors."""
    df = pd.read_csv(csv_path)
    vec_re_cols = sorted(
        [c for c in df.columns if c.startswith("vec_re_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    vec_im_cols = sorted(
        [c for c in df.columns if c.startswith("vec_im_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    if not vec_re_cols:
        raise ValueError(
            f"{csv_path} has no vec_re_* columns — re-run the flutter analysis "
            "with the updated pk_method/post_processing to export eigenvectors."
        )

    by_v: dict[float, list] = {}
    for _, row in df.iterrows():
        kind = str(row.get("kind", ""))
        if kinds is not None and kind not in kinds:
            continue
        n = int(row["state_dim"]) if "state_dim" in df.columns else len(vec_re_cols)
        re = row[vec_re_cols[:n]].to_numpy(dtype=float)
        im = row[vec_im_cols[:n]].to_numpy(dtype=float)
        vec = re + 1j * im
        V = float(row["Velocity_m_s"])
        by_v.setdefault(V, []).append(
            {
                "V": V,
                "index": int(row["spectrum_index"]),
                "kind": kind,
                "omega": float(row["omega_rad_s"]),
                "gamma": float(row["gamma"]),
                "vec": vec,
            }
        )
    return by_v


def track_branches(
    by_v: dict,
    *,
    w_mac: float = 0.8,
    w_freq: float = 0.4,
    max_freq_jump: float = 0.30,
    mac_floor: float = 0.5,
    n_structural_dof=None,
):
    """
    Greedy/optimal assignment of per-velocity roots into continuous branches.

    Returns a list of branches; each branch is a list of point dicts with keys
    V, omega, gamma, index, kind, mac_prev.
    """
    velocities = sorted(by_v.keys())
    branches: list[dict] = []  # active: {"pts": [...], "vec": last_vec, "omega": last_omega}
    finished: list[list] = []

    for V in velocities:
        roots = by_v[V]
        if not branches:
            for r in roots:
                branches.append(
                    {
                        "pts": [
                            {**{k: r[k] for k in ("V", "omega", "gamma", "index", "kind")},
                             "mac_prev": np.nan}
                        ],
                        "vec": r["vec"],
                        "omega": r["omega"],
                    }
                )
            continue

        nb, nr = len(branches), len(roots)
        BIG = 1e6
        cost = np.full((nb, nr), BIG)
        mac_mat = np.zeros((nb, nr))
        for bi, b in enumerate(branches):
            for ri, r in enumerate(roots):
                mac = _compute_mac(b["vec"], r["vec"], n_structural_dof)
                mac_mat[bi, ri] = mac
                dom = abs(r["omega"] - b["omega"]) / max(b["omega"], 1e-6)
                c = w_mac * (1.0 - mac) + w_freq * dom
                # guard: implausible jump AND shape mismatch -> forbid
                if dom > max_freq_jump and mac < mac_floor:
                    c = BIG
                cost[bi, ri] = c

        if _HAVE_SCIPY:
            rows, cols = linear_sum_assignment(cost)
            pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < BIG]
        else:  # greedy fallback
            pairs = []
            used_b, used_r = set(), set()
            order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
            for bi, ri in order:
                bi, ri = int(bi), int(ri)
                if bi in used_b or ri in used_r or cost[bi, ri] >= BIG:
                    continue
                used_b.add(bi)
                used_r.add(ri)
                pairs.append((bi, ri))

        matched_b = {bi for bi, _ in pairs}
        matched_r = {ri for _, ri in pairs}

        for bi, ri in pairs:
            b, r = branches[bi], roots[ri]
            b["pts"].append(
                {**{k: r[k] for k in ("V", "omega", "gamma", "index", "kind")},
                 "mac_prev": float(mac_mat[bi, ri])}
            )
            b["vec"] = r["vec"]
            b["omega"] = r["omega"]

        # close unmatched branches
        for bi in range(nb - 1, -1, -1):
            if bi not in matched_b:
                finished.append(branches[bi]["pts"])
                branches.pop(bi)

        # seed new branches from unmatched roots
        for ri, r in enumerate(roots):
            if ri not in matched_r:
                branches.append(
                    {
                        "pts": [
                            {**{k: r[k] for k in ("V", "omega", "gamma", "index", "kind")},
                             "mac_prev": np.nan}
                        ],
                        "vec": r["vec"],
                        "omega": r["omega"],
                    }
                )

    finished.extend(b["pts"] for b in branches)
    return finished


def _plot(branches, out_png: str, title: str):
    fig, (ax_o, ax_g) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    cmap = plt.cm.turbo
    n = max(len(branches), 1)
    for bi, pts in enumerate(branches):
        V = [p["V"] for p in pts]
        om = [p["omega"] for p in pts]
        g = [p["gamma"] for p in pts]
        color = cmap(bi / n)
        ax_o.plot(V, om, "o-", color=color, ms=3, lw=1.2, label=f"branch {bi} (n={len(pts)})")
        ax_g.plot(V, g, "o-", color=color, ms=3, lw=1.2)

    ax_g.axhline(0.0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax_o.set_ylabel(r"$\omega = |\mathrm{Im}(\lambda)|$ [rad/s]")
    ax_g.set_ylabel(r"$g = \sigma/\omega$ [-]")
    ax_g.set_xlabel("Airspeed V [m/s]")
    ax_o.grid(True, alpha=0.3)
    ax_g.grid(True, alpha=0.3)
    if len(branches) <= 20:
        ax_o.legend(fontsize=8, ncol=2, loc="best")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="directory with pk_untracked_wet_roots_vectors.csv")
    ap.add_argument("--w-mac", type=float, default=0.8)
    ap.add_argument("--w-freq", type=float, default=0.4)
    ap.add_argument("--max-freq-jump", type=float, default=0.30,
                    help="reject a match if |Δω|/ω exceeds this AND MAC < --mac-floor")
    ap.add_argument("--mac-floor", type=float, default=0.5)
    ap.add_argument("--min-points", type=int, default=3,
                    help="drop tracked branches shorter than this many samples")
    ap.add_argument("--struct-dof", type=int, default=None,
                    help="slice eigenvectors to first N DOF before MAC (e.g. 2*n_modes for RFA lag states)")
    ap.add_argument("--include-kinds", default="untracked,structural_near",
                    help="comma-separated root kinds to track (use 'all' for every root)")
    args = ap.parse_args()

    csv_path = os.path.join(args.dir, "pk_untracked_wet_roots_vectors.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(
            f"Not found: {csv_path}\n"
            "Re-run the flutter analysis first so eigenvectors get exported."
        )

    kinds = None if args.include_kinds.strip().lower() == "all" else set(
        k.strip() for k in args.include_kinds.split(",") if k.strip()
    )
    by_v = _load_roots(csv_path, kinds)
    branches = track_branches(
        by_v,
        w_mac=args.w_mac,
        w_freq=args.w_freq,
        max_freq_jump=args.max_freq_jump,
        mac_floor=args.mac_floor,
        n_structural_dof=args.struct_dof,
    )
    branches = [b for b in branches if len(b) >= args.min_points]
    branches.sort(key=lambda b: b[0]["V"])

    rows = []
    for bi, pts in enumerate(branches):
        for p in pts:
            rows.append(
                {
                    "branch_id": bi,
                    "Velocity_m_s": p["V"],
                    "omega_rad_s": p["omega"],
                    "gamma": p["gamma"],
                    "spectrum_index": p["index"],
                    "kind": p["kind"],
                    "mac_prev": p["mac_prev"],
                }
            )
    out_csv = os.path.join(args.dir, "pk_untracked_wet_roots_tracked.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.6e")

    out_png = os.path.join(args.dir, "pk_untracked_wet_roots_tracked.png")
    _plot(branches, out_png, "Untracked wet roots — MAC-tracked branches")

    print(f"Tracked {len(branches)} branch(es) from {len(by_v)} velocities.")
    for bi, pts in enumerate(branches):
        print(
            f"  branch {bi}: {len(pts)} pts, "
            f"V=[{pts[0]['V']:.2f}, {pts[-1]['V']:.2f}] m/s, "
            f"ω=[{min(p['omega'] for p in pts):.1f}, {max(p['omega'] for p in pts):.1f}] rad/s"
        )
    print(f"Saved {out_csv}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
