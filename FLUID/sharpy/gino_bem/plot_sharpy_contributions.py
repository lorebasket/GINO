#!/usr/bin/env python3
"""
Replot SharPy p-k + BEM stiffness / mass / damping contributions (GINO-style).

Uses ``stiffness_damping_contributions.csv`` written by ``run_abramson_pk_bem.py``.

  conda activate sharpy
  cd sharpy
  python gino_bem/plot_sharpy_contributions.py \\
    --csv-dir ../../output_data/ABRAMSON1965/07_sharpy_pk_bem/wetcheck
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
_FSI_ROOT = _SHARPY_ROOT.parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))
if str(_FSI_ROOT) not in sys.path:
    sys.path.insert(0, str(_FSI_ROOT))


def load_contributions_csv(path: Path) -> list[dict]:
    """Rebuild contributions_history from GINO-format CSV."""
    df = pd.read_csv(path)
    if df.empty:
        return []

    key_map = {
        "M_structural": "M_struct",
        "M_hydro_Capytaine": "M_hydro",
        "M_effective": "M_effective",
        "K_structural": "K_struct",
        "K_aerodynamic_B0": "K_aero",
        "K_hydro": "K_hydro",
        "K_effective": "K_effective",
        "C_structural": "C_struct",
        "C_aerodynamic_B1": "C_aero",
        "C_hydro_Capytaine": "C_hydro",
        "C_empirical": "C_empirical",
        "C_effective": "C_effective",
    }
    entries = []
    for _, row in df.iterrows():
        entry = {
            "V": float(row["Velocity_m_s"]),
            "branch_dof_indices": [0, 1],
            "omega_rad_s": float(row.get("omega_rad_s", np.nan)),
        }
        for csv_prefix, internal in key_map.items():
            mat = np.zeros((2, 2), dtype=float)
            for i in range(2):
                for j in range(2):
                    col = f"{csv_prefix}_{i}{j}"
                    if col in row.index:
                        mat[i, j] = float(row[col])
            entry[internal] = [mat, mat.copy()]
        entries.append(entry)
    return entries


def plot_contributions(history: list[dict], outdir: Path, *, show: bool = False) -> None:
    from gino_bem.contributions_plot import plot_stiffness_damping_contributions

    plot_stiffness_damping_contributions(history, outdir, show_plot=show)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True,
                    help="folder with stiffness_damping_contributions.csv")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    csv_path = args.csv_dir / "stiffness_damping_contributions.csv"
    if not csv_path.is_file():
        raise SystemExit(f"Missing {csv_path}")
    hist = load_contributions_csv(csv_path)
    if not hist:
        raise SystemExit("No rows in contributions CSV")
    plot_contributions(hist, args.csv_dir, show=args.show)
    print(f"Wrote {args.csv_dir / 'stiffness_damping_contributions.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
