"""Shared Abramson SharPy UVLM post-processing velocity thresholds."""

from __future__ import annotations

KN_TO_MS = 0.514444

# First dense-sweep sample [m/s] where participation-tracked bending (~7 Hz) and
# torsion (~15 Hz) are separated; below this, wake modes pollute the spectrum.
# Equals ~6.8 kn (not 3.5 kn).
ABRAMSON_V_MIN_M_S = 3.5
ABRAMSON_V_MIN_KN = ABRAMSON_V_MIN_M_S / KN_TO_MS


def at_or_above_m_s(v_m_s: float) -> bool:
    return float(v_m_s) >= ABRAMSON_V_MIN_M_S - 1e-9


def at_or_above_kn(v_kn: float) -> bool:
    return float(v_kn) >= ABRAMSON_V_MIN_KN - 1e-9
