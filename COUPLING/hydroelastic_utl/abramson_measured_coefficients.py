"""
Measured 3-D oscillatory hydrofoil coefficients (Abramson curve 3).

Data: DTIC TR-4 (Dec 1963), mid-span (50 %), quarter-chord EA form
(L_hh, L_hα, M_hh, M_αα) vs reduced velocity U = V/(b·ω) = 1/k.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_CSV = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "resources"
    / "benchmarks"
    / "abramson1965_report"
    / "abramson_measured_coefficients_span50.csv"
)


def _repo_csv_path() -> Path:
    fsi = Path(__file__).resolve().parents[2]
    p = fsi / "docs" / "resources" / "benchmarks" / "abramson1965_report"
    return p / "abramson_measured_coefficients_span50.csv"


def load_measured_coefficients_span50(csv_path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Load mid-span measured coefficients; return sorted by inv_k (= V/(bω))."""
    path = Path(csv_path) if csv_path is not None else _repo_csv_path()
    if not path.is_file():
        path = _DEFAULT_CSV
    if not path.is_file():
        raise FileNotFoundError(f"Measured coefficient table not found: {path}")

    df = pd.read_csv(path, comment="#")
    df = df.sort_values("inv_k").reset_index(drop=True)
    for col in df.columns:
        if col != "inv_k":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _complex_from_mag_phase(mag: float, phase_deg: float) -> complex:
    ph = np.deg2rad(float(phase_deg))
    return float(mag) * (np.cos(ph) + 1j * np.sin(ph))


def h_matrix_at_inv_k(df: pd.DataFrame, inv_k: float) -> np.ndarray:
    """
    2×2 complex flutter-derivative matrix at EA (quarter chord).

    Ordering: [bend (h), pitch (α)] — rows (L, M), cols (h, α).
    """
    u = float(inv_k)
    inv_arr = df["inv_k"].to_numpy(dtype=float)
    if u <= inv_arr[0]:
        row = df.iloc[0]
    elif u >= inv_arr[-1]:
        row = df.iloc[-1]
    else:
        row = df.iloc[int(np.argmin(np.abs(inv_arr - u)))]

    def c(name_mag, name_ph):
        return _complex_from_mag_phase(row[name_mag], row[name_ph])

    return np.array(
        [
            [c("L_hh_mag", "L_hh_phase_deg"), c("L_ha_mag", "L_ha_phase_deg")],
            [c("M_hh_mag", "M_hh_phase_deg"), c("M_aa_mag", "M_aa_phase_deg")],
        ],
        dtype=complex,
    )


def k_from_inv_k(inv_k: float, b: float) -> float:
    """
    Reduced frequency for PanelAero (k = ω/V [1/m]) from paper U = V/(bω).

    U = V/(bω) = 1/(k·b)  ⇒  k = 1/(U·b).
    """
    u = float(inv_k)
    b = float(b)
    if u <= 1e-12 or b <= 1e-12:
        return np.inf
    return 1.0 / (u * b)


def inv_k_from_k(k: float, b: float) -> float:
    """U = V/(bω) = 1/(k·b)."""
    k = float(k)
    b = float(b)
    if not np.isfinite(k) or k <= 1e-12 or b <= 1e-12:
        return np.inf
    return 1.0 / (k * b)


def interpolate_h_matrix(df: pd.DataFrame, inv_k: float) -> np.ndarray:
    """Linear interpolation of Re/Im of each entry vs inv_k."""
    u = float(inv_k)
    inv_arr = df["inv_k"].to_numpy(dtype=float)
    if u <= inv_arr[0]:
        return h_matrix_at_inv_k(df, inv_arr[0])
    if u >= inv_arr[-1]:
        return h_matrix_at_inv_k(df, inv_arr[-1])

    names = [
        ("L_hh_mag", "L_hh_phase_deg"),
        ("L_ha_mag", "L_ha_phase_deg"),
        ("M_hh_mag", "M_hh_phase_deg"),
        ("M_aa_mag", "M_aa_phase_deg"),
    ]
    H = np.zeros((2, 2), dtype=complex)
    slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (i, j), (nm, np_) in zip(slots, names):
        mags = df[nm].to_numpy(dtype=float)
        phs = np.deg2rad(df[np_].to_numpy(dtype=float))
        z = mags * (np.cos(phs) + 1j * np.sin(phs))
        H[i, j] = np.interp(u, inv_arr, z.real) + 1j * np.interp(u, inv_arr, z.imag)
    return H
