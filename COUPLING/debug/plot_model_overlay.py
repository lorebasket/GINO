#!/usr/bin/env python3
"""Overlay of the three discretisations of a case: GINO beam (FEM nodes), UVLM lattice (SHARPy
zeta vertices) and BEM hull (Capytaine mesh), in three orthogonal views + 3D.

Any of the three can be omitted.  Each source lives in its own frame: bring it into a common
one with an affine map  p' = R p + o  given as 12 numbers "R11,R12,...,R33,ox,oy,oz".

    python COUPLING/debug/plot_model_overlay.py --beam output_data/X/dry_modes.npz \\
        --uvlm grid.npz --bem FLUID/capytaine/X/X_mesh.vtu -o overlay.png \\
        --uvlm-affine 1,0,0,0,-1,0,0,0,-1,0,0,-0.2 --wl -0.1

  --beam  dry_modes.npz (pos, sub) from utl/export_modes.py, or a *_dry_egv_nodes.csv
  --uvlm  npz with n_surf and zeta<i> (3, M+1, N+1), e.g. gino_bem/dump_tfoil_simple_grid.py
  --bem   any meshio-readable mesh (.vtu, .stl, .obj, ...)
"""
import argparse

import numpy as np


def _affine(text):
    if not text:
        return np.eye(3), np.zeros(3)
    v = np.array([float(x) for x in text.split(",")])
    return v[:9].reshape(3, 3), v[9:12]


def _apply(p, aff):
    R, o = aff
    return p @ R.T + o


def load_beam(path):
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        sub = d["sub"].astype(str) if "sub" in d else np.full(len(d["pos"]), "beam")
        return np.asarray(d["pos"], float), sub
    import csv
    rows = list(csv.DictReader(open(path)))
    keys = [k for k in rows[0] if k.lower() in ("x", "y", "z")]
    pos = np.array([[float(r[k]) for k in keys] for r in rows])
    return pos, np.full(len(pos), "beam")


def load_uvlm(path):
    g = np.load(path)
    quads = []
    for i in range(int(g["n_surf"])):
        z = np.moveaxis(np.asarray(g[f"zeta{i}"]), 0, -1)        # (M+1, N+1, 3)
        for a in range(z.shape[0] - 1):
            for b in range(z.shape[1] - 1):
                quads.append([z[a, b], z[a + 1, b], z[a + 1, b + 1], z[a, b + 1]])
    return np.array(quads)


def load_bem(path):
    import meshio
    m = meshio.read(path)
    faces = [c.data for c in m.cells if c.type in ("triangle", "quad")]
    polys = []
    for f in faces:
        polys += [m.points[ff] for ff in f]
    return polys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beam")
    ap.add_argument("--uvlm")
    ap.add_argument("--bem")
    for k in ("beam", "uvlm", "bem"):
        ap.add_argument(f"--{k}-affine", help="R11..R33,ox,oy,oz")
    ap.add_argument("--wl", type=float, help="free-surface height (vertical coordinate)")
    ap.add_argument("--vertical", type=int, default=2, help="index of the vertical axis")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    C_BEAM, C_UVLM, C_BEM = "k", "#E4572E", "#13A89E"
    beam = sub = uvlm = None
    bem = []
    if a.beam:
        beam, sub = load_beam(a.beam)
        beam = _apply(beam, _affine(a.beam_affine))
    if a.uvlm:
        uvlm = _apply(load_uvlm(a.uvlm), _affine(a.uvlm_affine))
    if a.bem:
        bem = [_apply(p, _affine(a.bem_affine)) for p in load_bem(a.bem)]

    views = [("x-z", 0, 2), ("y-z", 1, 2), ("x-y", 0, 1)]
    fig = plt.figure(figsize=(20, 6))
    for k, (tit, i, j) in enumerate(views):
        ax = fig.add_subplot(1, 4, k + 1)
        if bem:
            ax.add_collection(PolyCollection([p[:, [i, j]] for p in bem], facecolor=C_BEM, edgecolor=C_BEM,
                                             alpha=0.18, lw=0.3))
        if uvlm is not None:
            ax.add_collection(PolyCollection(uvlm[:, :, [i, j]], facecolor="none", edgecolor=C_UVLM, lw=0.5))
        if beam is not None:
            for s in sorted(set(sub)):
                p = beam[sub == s]
                o = np.argsort(np.linalg.norm(p - p[0], axis=1))
                ax.plot(p[o, i], p[o, j], "-o", color=C_BEAM, ms=1.8, lw=1.4)
        if a.wl is not None and a.vertical in (i, j):
            (ax.axhline if j == a.vertical else ax.axvline)(a.wl, color="royalblue", ls="--", lw=1.2)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.grid(alpha=.3)
        ax.set_xlabel("xyz"[i] + " [m]")
        ax.set_ylabel("xyz"[j] + " [m]")
        ax.set_title(tit)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    pts = []
    if bem:
        ax.add_collection3d(Poly3DCollection(bem, facecolor=C_BEM, edgecolor="none", alpha=0.25))
        pts.append(np.vstack(bem))
    if uvlm is not None:
        ax.add_collection3d(Line3DCollection([q[[0, 1, 2, 3, 0]] for q in uvlm], colors=C_UVLM, lw=0.3))
        pts.append(uvlm.reshape(-1, 3))
    if beam is not None:
        for s in sorted(set(sub)):
            p = beam[sub == s]
            o = np.argsort(np.linalg.norm(p - p[0], axis=1))
            ax.plot(p[o, 0], p[o, 1], p[o, 2], "-", color=C_BEAM, lw=1.6)
        pts.append(beam)
    if pts:
        P = np.vstack(pts)
        lo, hi = P.min(0), P.max(0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(np.maximum(hi - lo, 1e-3))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    hand = []
    if beam is not None:
        hand.append(Line2D([0], [0], color=C_BEAM, marker="o", ms=3, label=f"GINO beam ({len(beam)} nodes)"))
    if uvlm is not None:
        hand.append(Line2D([0], [0], color=C_UVLM, label=f"UVLM lattice ({len(uvlm)} panels)"))
    if bem:
        hand.append(Line2D([0], [0], color=C_BEM, lw=6, alpha=.4, label=f"BEM hull ({len(bem)} panels)"))
    if a.wl is not None:
        hand.append(Line2D([0], [0], color="royalblue", ls="--", label=f"free surface {a.wl:g} m"))
    fig.legend(handles=hand, loc="upper center", ncol=len(hand), frameon=False)
    fig.suptitle(a.title, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(a.out, dpi=140)
    print(a.out)


if __name__ == "__main__":
    main()
