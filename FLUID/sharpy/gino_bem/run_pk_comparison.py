#!/usr/bin/env python3
"""
Abramson 1965 — UVLM apparent-mass removal x Capytaine BEM comparison matrix.

Runs run_abramson_pk_bem.py for {none, gamma_dot, full} x {BEM on, off} in
subprocesses (SHARPy monkey-patches are process-global), then tabulates the
flutter crossings against experiment (48.1 kn) and GINO (48.75 kn).

  conda activate sharpy
  cd sharpy
  python gino_bem/run_pk_comparison.py --coarse
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
_FSI_ROOT = _SHARPY_ROOT.parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))

from gino_bem.run_abramson_pk_bem import KN, V_EXP_KN, V_GINO_KN, flutter_crossing  # noqa: E402

COMBOS = [
    ("none", False),        # stock SHARPy baseline (regression anchor ~39.8 kn)
    ("none", True),         # double counting (illustration)
    ("gamma_dot", False),
    ("gamma_dot", True),    # naive replace-hybrid (over-removes V-coupled nc damping)
    ("full", False),
    ("full", True),
    ("delta", True),        # UVLM intact + (A_capy - A_flat_plate) correction
]


def run_combo(removal: str, bem: bool, *, grid_flag: str, nm: int, stride: int,
              v_max_kn: float, extra: list[str]) -> tuple[str, Path]:
    tag = f"{removal}_{'bem' if bem else 'nobem'}_nm{nm}_{grid_flag.lstrip('-') or 'default'}"
    cmd = [sys.executable, str(_SHARPY_ROOT / "gino_bem" / "run_abramson_pk_bem.py"),
           "--num-modes", str(nm),
           "--stride", str(stride), "--v-max-kn", str(v_max_kn), "--tag", tag]
    if removal == "delta":
        cmd += ["--bem-mode", "delta", "--added-mass-removal", "none"]
    else:
        cmd += ["--added-mass-removal", removal]
    if grid_flag:
        cmd.append(grid_flag)
    if not bem:
        cmd.append("--no-bem")
    cmd += extra
    print(f"\n>>> [{tag}] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(_SHARPY_ROOT))
    if proc.returncode != 0:
        print(f"!! combo {tag} FAILED (rc={proc.returncode})", flush=True)
    outdir = _FSI_ROOT / "output_data" / "ABRAMSON1965" / "sharpy_pk_bem" / tag
    return tag, outdir / f"abramson_pk_bem_tracked_nm{nm}.csv"


def read_rows(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            rows.append(dict(branch=r["branch"], V_m_s=float(r["V_m_s"]),
                             V_kn=float(r["V_kn"]), g=float(r["g"]),
                             freq_Hz=float(r["freq_Hz"])))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--fine", action="store_true")
    ap.add_argument("--num-modes", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--v-max-kn", type=float, default=60.0)
    ap.add_argument("--skip-run", action="store_true", help="only tabulate existing CSVs")
    ap.add_argument("--extra", type=str, default="", help="extra flags passed through to the runner")
    args = ap.parse_args()

    grid_flag = "--coarse" if args.coarse else ("--fine" if args.fine else "")
    grid_name = grid_flag.lstrip("-") or "default"
    nm = args.num_modes
    extra = [x for x in args.extra.split() if x]

    results = []
    for removal, bem in COMBOS:
        if args.skip_run:
            tag = f"{removal}_{'bem' if bem else 'nobem'}_nm{nm}_{grid_name}"
            csv_path = (_FSI_ROOT / "output_data" / "ABRAMSON1965" / "sharpy_pk_bem" / tag
                        / f"abramson_pk_bem_tracked_nm{nm}.csv")
        else:
            tag, csv_path = run_combo(removal, bem, grid_flag=grid_flag, nm=nm,
                                      stride=args.stride, v_max_kn=args.v_max_kn, extra=extra)
        rows = read_rows(csv_path)
        vf_best, ff_best, branch_best = None, None, None
        for branch in sorted({r["branch"] for r in rows}):
            vf, ff = flutter_crossing(rows, branch)
            if vf is not None and (vf_best is None or vf < vf_best):
                vf_best, ff_best, branch_best = vf, ff, branch
        results.append(dict(removal=removal, bem=bem, tag=tag, V_f_m_s=vf_best,
                            f_f_Hz=ff_best, branch=branch_best))

    outdir = _FSI_ROOT / "output_data" / "ABRAMSON1965" / "sharpy_pk_bem"
    outdir.mkdir(parents=True, exist_ok=True)
    summary_csv = outdir / f"pk_comparison_summary_nm{nm}_{grid_name}.csv"
    with summary_csv.open("w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["removal", "bem", "branch", "V_f_kn", "f_f_Hz",
                      "err_vs_exp_pct", "err_vs_gino_pct"])
        for r in results:
            vf_kn = None if r["V_f_m_s"] is None else r["V_f_m_s"] / KN
            wri.writerow([
                r["removal"], "on" if r["bem"] else "off", r["branch"],
                f"{vf_kn:.2f}" if vf_kn else "-",
                f"{r['f_f_Hz']:.2f}" if r["f_f_Hz"] else "-",
                f"{100 * (vf_kn - V_EXP_KN) / V_EXP_KN:+.1f}" if vf_kn else "-",
                f"{100 * (vf_kn - V_GINO_KN) / V_GINO_KN:+.1f}" if vf_kn else "-",
            ])

    print(f"\n{'removal':>10s} {'BEM':>4s} {'branch':>8s} {'V_f [kn]':>9s} {'f_f [Hz]':>9s} "
          f"{'vs exp':>8s} {'vs GINO':>8s}")
    print("-" * 62)
    for r in results:
        vf_kn = None if r["V_f_m_s"] is None else r["V_f_m_s"] / KN
        if vf_kn is None:
            print(f"{r['removal']:>10s} {'on' if r['bem'] else 'off':>4s} {'-':>8s} "
                  f"{'no cross':>9s}")
        else:
            print(f"{r['removal']:>10s} {'on' if r['bem'] else 'off':>4s} "
                  f"{r['branch']:>8s} {vf_kn:9.2f} {r['f_f_Hz']:9.2f} "
                  f"{100 * (vf_kn - V_EXP_KN) / V_EXP_KN:+7.1f}% "
                  f"{100 * (vf_kn - V_GINO_KN) / V_GINO_KN:+7.1f}%")
    print(f"\nexp = {V_EXP_KN} kn, GINO = {V_GINO_KN} kn")
    print(f"summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
