#!/usr/bin/env python3
"""Optional: load GINO Capytaine NPZ and document BEM injection (see capy_bem_bridge.py)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
if str(_SHARPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_SHARPY_ROOT))

from gino_bem.capy_bem_bridge import capy_at_omega, default_capytaine_npz, load_capytaine_npz  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--capy-npz", type=Path, default=None)
    p.add_argument("--omega", type=float, default=110.0)
    p.add_argument("--n-modes", type=int, default=2)
    args = p.parse_args()

    path = args.capy_npz or default_capytaine_npz()
    capy = load_capytaine_npz(path)
    A, B = capy_at_omega(capy, args.omega, mode_slice=slice(0, args.n_modes))
    print(f"Loaded {path}")
    print(f"A(ω={args.omega}): diag = {np.diag(A)}")
    print(f"B(ω={args.omega}): diag = {np.diag(B)}")
    print("For stock SHARPy + rho only, use: python gino_bem/run_goland_water_rho.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
