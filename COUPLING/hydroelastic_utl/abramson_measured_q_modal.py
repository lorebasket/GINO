"""
Build modal Q(k) from Abramson measured 3-D coefficients (curve 3) for Roger RFA.

Replaces DLM Q_modal on the bend–pitch 2×2 block; calibration maps dimensionless
H( U ) to Q_modal units using one DLM reference point (default U = 1).
"""

from __future__ import annotations

import numpy as np

from . import abramson_measured_coefficients as amc
from .abramson_imq_comparison import (
    _modal_scale_from_dlm_and_measured,
    compute_Q_modal_at_k,
)


def _embed_2x2(Q2: np.ndarray, n_modes: int, mode_indices=(0, 1)) -> np.ndarray:
    Q = np.zeros((n_modes, n_modes), dtype=complex)
    idx = [int(i) for i in mode_indices]
    if len(idx) != 2:
        raise ValueError("mode_indices must have length 2 for measured Abramson Q")
    Q[np.ix_(idx, idx)] = np.asarray(Q2, dtype=complex)
    return Q


def calibration_scale_matrix(
    config,
    structural_results,
    coupling_results,
    aerogrid,
    *,
    inv_k_cal: float = 1.0,
    V_for_k: float = 15.0,
    mode_indices=(0, 1),
    measured_csv: str | None = None,
) -> np.ndarray:
    """2×2 scale: |Q_DLM_ij| / |H_ij| at inv_k_cal."""
    df = amc.load_measured_coefficients_span50(measured_csv)
    b = float(config.chord) / 2.0
    k_cal = amc.k_from_inv_k(inv_k_cal, b)
    H_cal = amc.interpolate_h_matrix(df, inv_k_cal)
    Q_cal = compute_Q_modal_at_k(
        k_cal, V_for_k, config, structural_results, coupling_results, aerogrid,
        mode_indices=mode_indices,
    )
    _, scale = _modal_scale_from_dlm_and_measured(Q_cal, H_cal)
    return scale


def measured_q_modal_at_k(
    k: float,
    scale: np.ndarray,
    config,
    *,
    mode_indices=(0, 1),
    n_modes: int | None = None,
    measured_csv: str | None = None,
    im_flip: bool = True,
) -> np.ndarray:
    """Single Q_modal matrix at reduced frequency k [1/m]."""
    b = float(config.chord) / 2.0
    inv_k = amc.inv_k_from_k(float(k), b)
    df = amc.load_measured_coefficients_span50(measured_csv)
    H = amc.interpolate_h_matrix(df, inv_k)
    im = H.imag
    if im_flip:
        im = -im
    im_scale = float(getattr(config, "aero_measured_q_im_scale", 1.0))
    H = H.real + 1j * (im_scale * im)
    Q2 = np.asarray(scale, dtype=complex) * H
    nm = int(n_modes) if n_modes is not None else max(mode_indices) + 1
    return _embed_2x2(Q2, nm, mode_indices=mode_indices)


def build_measured_q_modal_array(
    k_list: np.ndarray,
    config,
    structural_results,
    coupling_results,
    aerogrid,
    *,
    n_modes_modal: int,
    mode_indices=(0, 1),
    inv_k_cal: float | None = None,
    V_for_k: float = 15.0,
    measured_csv: str | None = None,
    im_flip: bool | None = None,
    apply_conj: bool = True,
) -> np.ndarray:
    """
    Return Q_modal array shape (Nk, n_modes_modal, n_modes_modal) from measured H.

    ``apply_conj``: match DLM Roger path (conj before fit) for PK-consistent C_aero.
    """
    if inv_k_cal is None:
        inv_k_cal = float(getattr(config, "aero_measured_q_cal_inv_k", 1.0))
    if im_flip is None:
        im_flip = bool(getattr(config, "aero_measured_q_im_flip", True))

    scale = calibration_scale_matrix(
        config,
        structural_results,
        coupling_results,
        aerogrid,
        inv_k_cal=inv_k_cal,
        V_for_k=V_for_k,
        mode_indices=mode_indices,
        measured_csv=measured_csv,
    )
    print(
        f"  [measured Q] calibration at inv_k={inv_k_cal}: "
        f"scale diag = {np.diag(scale)}, im_flip={im_flip}"
    )

    k_list = np.asarray(k_list, dtype=float).ravel()
    out = np.zeros((k_list.size, n_modes_modal, n_modes_modal), dtype=complex)
    for i, k in enumerate(k_list):
        out[i] = measured_q_modal_at_k(
            k,
            scale,
            config,
            mode_indices=mode_indices,
            n_modes=n_modes_modal,
            measured_csv=measured_csv,
            im_flip=im_flip,
        )
    if apply_conj:
        out = np.conj(out)
    return out


def build_hybrid_q_modal_array(
    Q_dlm_array: np.ndarray,
    k_list: np.ndarray,
    config,
    structural_results,
    coupling_results,
    aerogrid,
    *,
    n_modes_modal: int,
    mode_indices=(0, 1),
    inv_k_cal: float | None = None,
    V_for_k: float = 15.0,
    measured_csv: str | None = None,
    im_flip: bool | None = None,
) -> np.ndarray:
    """
    Re(Q) from DLM, Im(Q) from measured Abramson H on the bend–pitch 2×2 block.

    ``aero_hybrid_im_weight`` blends Im: 0 = DLM only, 1 = measured Im only.
    Both inputs must already use the PK-consistent conj(Q) convention.
    """
    w = float(getattr(config, "aero_hybrid_im_weight", 1.0))
    w = max(0.0, min(1.0, w))
    Q_meas = build_measured_q_modal_array(
        k_list,
        config,
        structural_results,
        coupling_results,
        aerogrid,
        n_modes_modal=n_modes_modal,
        mode_indices=mode_indices,
        inv_k_cal=inv_k_cal,
        V_for_k=V_for_k,
        measured_csv=measured_csv,
        im_flip=im_flip,
        apply_conj=True,
    )
    out = np.asarray(Q_dlm_array, dtype=complex).copy()
    idx = [int(i) for i in mode_indices]
    sl = np.ix_(idx, idx)
    for i in range(out.shape[0]):
        Qd = out[i][sl]
        Qm = Q_meas[i][sl]
        out[i][sl] = Qd.real + 1j * ((1.0 - w) * Qd.imag + w * Qm.imag)
    print(
        f"  [hybrid Q] Re=DLM Im=measured blend w={w:.3f} on modes {idx}"
    )
    return out
