#!/usr/bin/env python3
"""
MAC-based modal-basis alignment: GINO FEA modes (basis of the Capytaine NPZ
matrices) vs SHARPy linear-beam modes.

The Capytaine modal added-mass/damping matrices are expressed in GINO's
mass-normalized FEA modal basis. SHARPy's linear beam uses its own
mass-normalized modal basis. Before injecting A(omega), B(omega) into SHARPy
the two bases must be matched in order and sign:

    q_gino = T q_sharpy   =>   A_sharpy = T.T @ A_gino @ T

with T a signed permutation obtained by MAC pairing of the [plunge, twist]
mode-shape profiles interpolated to common span stations. Axis indices
(span / plunge / twist) are auto-detected from the data rather than
hard-coded (different frameworks order node DOFs differently).
"""
from __future__ import annotations

import numpy as np

# semichord [m], balances translation vs rotation units in the MAC vectors
DEFAULT_THETA_SCALE = 0.1524


def _mac(a: np.ndarray, b: np.ndarray) -> float:
    num = np.abs(np.vdot(a, b)) ** 2
    den = float(np.vdot(a, a).real) * float(np.vdot(b, b).real)
    return num / den if den > 1e-30 else 0.0


def _detect_plunge_axis(disp: np.ndarray) -> int:
    """Translation component with max total energy over all modes.

    ``disp``: (n_modes, n_nodes, 3).
    """
    return int(np.argmax((disp ** 2).sum(axis=(0, 1))))


def gino_modal_displacements(capy: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract span stations and [plunge, twist] profiles from the NPZ source modes.

    Returns (y (n_nodes,), W (n_modes, n_nodes), TH (n_modes, n_nodes), freq_hz (n_modes,)).
    """
    vecs = np.asarray(capy["source_eigenvectors"], dtype=float)      # (n_modes, n_nodes*6)
    pos = np.asarray(capy["source_beam_positions"], dtype=float)     # (n_nodes, 3)
    n_modes = vecs.shape[0]
    n_nodes = pos.shape[0]
    if vecs.shape[1] != 6 * n_nodes:
        raise ValueError(f"source_eigenvectors {vecs.shape} inconsistent with {n_nodes} nodes x 6 DOF")
    span_axis = int(np.argmax(pos.max(axis=0) - pos.min(axis=0)))
    y = pos[:, span_axis]

    node_dofs = vecs.reshape(n_modes, n_nodes, 6)
    disp, rot = node_dofs[:, :, :3], node_dofs[:, :, 3:]
    i_w = _detect_plunge_axis(disp)
    # GINO FEA rotations are global-frame: twist = rotation about the span axis
    i_th = span_axis
    W = disp[:, :, i_w]
    TH = rot[:, :, i_th]

    lam = np.asarray(capy.get("source_eigenvalues", np.full(n_modes, np.nan)), dtype=float).ravel()
    freq_hz = np.sqrt(np.abs(lam[:n_modes])) / (2 * np.pi)

    print(f"[mode_alignment] GINO: span_axis={span_axis}, plunge_dof={i_w}, twist_dof={i_th + 3}, "
          f"dry f = {np.array2string(freq_hz, precision=2)} Hz")
    return y, W, TH, freq_hz


def sharpy_modal_displacements(data, num_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract span stations and [plunge, twist] profiles from SHARPy beam modes.

    Uses ``data.linear.linear_system.beam.sys.U`` (mass-normalized, vdof
    ordering: clamped nodes carry no DOFs; free nodes 6 DOFs each in node order).
    Frequencies may be in SHARPy's scaled time — only their ratios are used.
    """
    beam_sys = data.linear.linear_system.beam.sys
    U = np.asarray(beam_sys.U, dtype=float)[:, :num_modes]
    pos = np.asarray(data.linear.tsstruct0.pos, dtype=float)          # (n_nodes, 3)
    bcs = np.asarray(data.structure.boundary_conditions).ravel()
    free = bcs != 1                                                   # clamped nodes excluded from vdof
    n_free = int(free.sum())
    if U.shape[0] < 6 * n_free:
        raise ValueError(f"beam.sys.U has {U.shape[0]} rows < 6 x {n_free} free nodes")

    span_axis = int(np.argmax(pos.max(axis=0) - pos.min(axis=0)))
    y = pos[free, span_axis]

    node_dofs = U[: 6 * n_free, :].T.reshape(num_modes, n_free, 6)
    disp, rot = node_dofs[:, :, :3], node_dofs[:, :, 3:]
    i_w = _detect_plunge_axis(disp)
    # SHARPy CRV perturbations are material-frame; the beam is built with
    # x_b along span (abramson1965_wing.update_mass_stiff), so torsion = crv1
    i_th = 0
    W = disp[:, :, i_w]
    TH = rot[:, :, i_th]

    freq = np.asarray(beam_sys.freq_natural, dtype=float).ravel()[:num_modes]
    print(f"[mode_alignment] SHARPy: span_axis={span_axis}, plunge_dof={i_w}, twist_dof={i_th + 3}, "
          f"freq (possibly scaled) = {np.array2string(freq, precision=3)}")
    return y, W, TH, freq


def alignment_matrix(
    y_s: np.ndarray, W_s: np.ndarray, TH_s: np.ndarray, freq_s: np.ndarray,
    y_g: np.ndarray, W_g: np.ndarray, TH_g: np.ndarray, freq_g: np.ndarray,
    *,
    mac_floor: float = 0.6,
    theta_scale: float = DEFAULT_THETA_SCALE,
) -> tuple[np.ndarray, dict]:
    """Signed permutation T (q_gino = T q_sharpy) by greedy MAC pairing.

    GINO profiles are interpolated onto the SHARPy span stations; each mode is
    flattened to [w; theta_scale*theta] and unit-normalized. Pairs below
    ``mac_floor`` fall back to nearest frequency-ratio pairing with a warning.
    """
    n = min(W_s.shape[0], W_g.shape[0])
    # interpolate GINO onto SHARPy stations (handles reversed span ordering)
    order_g = np.argsort(y_g)
    vec_s, vec_g = [], []
    for j in range(n):
        vec_s.append(np.concatenate([W_s[j], theta_scale * TH_s[j]]))
        wg = np.interp(y_s, y_g[order_g], W_g[j][order_g])
        tg = np.interp(y_s, y_g[order_g], TH_g[j][order_g])
        vec_g.append(np.concatenate([wg, theta_scale * tg]))
    vec_s = [v / max(np.linalg.norm(v), 1e-30) for v in vec_s]
    vec_g = [v / max(np.linalg.norm(v), 1e-30) for v in vec_g]

    mac_tab = np.array([[_mac(vec_g[i], vec_s[j]) for j in range(n)] for i in range(n)])

    # greedy assignment: best MAC first
    T = np.zeros((n, n))
    pairs, used_g, used_s = [], set(), set()
    flat = sorted(((mac_tab[i, j], i, j) for i in range(n) for j in range(n)), reverse=True)
    for m, i, j in flat:
        if i in used_g or j in used_s:
            continue
        used_g.add(i)
        used_s.add(j)
        pairs.append((i, j, m))
    fallback = []
    fs = freq_s / max(freq_s[0], 1e-30)   # scale-invariant frequency ratios
    fg = freq_g / max(freq_g[0], 1e-30)
    for i, j, m in pairs:
        if m < mac_floor and np.isfinite(fg[i]):
            j_new = int(np.argmin(np.abs(fs - fg[i])))
            if j_new != j:
                fallback.append((i, j, j_new, m))
                j = j_new
        sign = np.sign(np.dot(vec_g[i], vec_s[j])) or 1.0
        T[i, j] = sign

    # column/row sanity: must remain a permutation after fallbacks
    if not (np.abs(T).sum(axis=0) == 1).all() or not (np.abs(T).sum(axis=1) == 1).all():
        raise RuntimeError(f"mode alignment failed to produce a permutation:\n{T}\nMAC:\n{mac_tab}")

    report = {"mac": mac_tab, "T": T, "pairs": pairs, "fallback": fallback,
              "freq_ratio_sharpy": fs, "freq_ratio_gino": fg}
    print("[mode_alignment] MAC table (rows=GINO, cols=SHARPy):")
    print(np.array2string(mac_tab, precision=3, suppress_small=True))
    print("[mode_alignment] T (q_gino = T q_sharpy):")
    print(np.array2string(T, precision=0, suppress_small=True))
    for i, j, j_new, m in fallback:
        print(f"[mode_alignment] WARNING: GINO mode {i} MAC={m:.2f} < {mac_floor} vs SHARPy {j}; "
              f"frequency-ratio fallback pairs it with SHARPy {j_new}")
    return T, report


def strip_added_mass_modal(y: np.ndarray, W: np.ndarray, TH: np.ndarray, *,
                           rho: float = 997.0, b: float = DEFAULT_THETA_SCALE,
                           a: float = -0.5) -> np.ndarray:
    """Analytic 2D flat-plate (Theodorsen non-circulatory) added mass in modal coords.

    Per unit span, in (w up, theta) coordinates about the elastic axis at
    ``a`` semichords aft of midchord (EA at quarter chord -> a = -0.5):

        M_am = pi rho b^2 [[1, b a], [b a, b^2 (1/8 + a^2)]]

    Projected with the (mass-normalized) mode profiles: A_ij = int [w_i th_i] M_am [w_j th_j]^T dy.
    The sign of the w-theta coupling depends on the theta sign convention; callers
    should check it against the BEM matrices (same physics) and flip if needed.
    """
    m = np.pi * rho * b ** 2
    Mam = m * np.array([[1.0, b * a], [b * a, b ** 2 * (0.125 + a ** 2)]])
    n = W.shape[0]
    order = np.argsort(y)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            pw = Mam[0, 0] * W[i] * W[j] + Mam[1, 1] * TH[i] * TH[j] \
                + Mam[0, 1] * (W[i] * TH[j] + TH[i] * W[j])
            A[i, j] = A[j, i] = np.trapz(pw[order], y[order])
    return A


def align_capy_to_sharpy(capy: dict, data, num_modes: int, *, mac_floor: float = 0.6) -> tuple[np.ndarray, dict]:
    """Convenience wrapper: full alignment from NPZ dict + SHARPy data object."""
    if "source_eigenvectors" not in capy:
        print("[mode_alignment] WARNING: NPZ has no source_eigenvectors — assuming identity alignment.")
        return np.eye(num_modes), {"mac": None, "T": np.eye(num_modes)}
    y_g, W_g, TH_g, f_g = gino_modal_displacements(capy)
    y_s, W_s, TH_s, f_s = sharpy_modal_displacements(data, num_modes)
    n = min(num_modes, W_g.shape[0])
    # NB: y_s / y_g are span stations (full length) — only the mode axis is truncated
    return alignment_matrix(y_s, W_s[:n], TH_s[:n], f_s[:n], y_g, W_g[:n], TH_g[:n], f_g[:n],
                            mac_floor=mac_floor)


def sharpy_modal_matrices(
    beam_sys,
    n_modes: int,
    *,
    v_ref: float | None = None,
    semichord: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Modal M, K and ω [rad/s] from SHARPy ``beam.sys``.

    After linear assembly the beam may be time-normalised (``freq_natural`` in
    reduced units).  Pass ``v_ref`` [m/s] and ``semichord`` [m] to recover
    physical rad/s via ``omega = freq_natural * v_ref / semichord`` (call *before*
    ``linear_system.update(V)`` re-scales at a new speed).
    """
    from scipy.linalg import eigh

    n_modes = min(int(n_modes), int(beam_sys.num_modes))
    phi = np.asarray(beam_sys.U[:, :n_modes], dtype=float)
    m_str = np.asarray(beam_sys.Mstr, dtype=float)
    ref = getattr(beam_sys, "scaled_reference_matrices", {}) or {}
    if "K" in ref:
        k_str = np.asarray(ref["K"], dtype=float)
    else:
        k_str = np.asarray(beam_sys.Kstr, dtype=float)

    m_hat = phi.T @ m_str @ phi
    k_hat = phi.T @ k_str @ phi

    freq = np.asarray(beam_sys.freq_natural[:n_modes], dtype=float).ravel()
    if v_ref is not None and semichord is not None and float(semichord) > 0.0:
        scale = float(v_ref) / float(semichord)
        omega = freq * scale
        # Rescale K from SHARPy reduced-time units to physical rad/s, preserving
        # off-diagonal bend–torsion coupling ratios.
        m_diag = np.maximum(np.real(np.diag(m_hat)), 1e-30)
        omega_bar = np.sqrt(np.maximum(np.real(np.diag(k_hat)) / m_diag, 0.0))
        omega_bar = np.maximum(omega_bar, 1e-30)
        scale_vec = omega / omega_bar
        k_hat = k_hat * np.outer(scale_vec, scale_vec)
    else:
        omega2, _ = eigh(k_hat, m_hat)
        omega = np.sqrt(np.maximum(np.real(omega2), 0.0))

    order = np.argsort(omega)
    omega = omega[order]
    perm = order
    m_hat = m_hat[perm][:, perm]
    k_hat = k_hat[perm][:, perm]
    return m_hat, k_hat, omega


def goland_twist_sign(
    y_sh: np.ndarray,
    W_sh: np.ndarray,
    TH_sh: np.ndarray,
    y_g: np.ndarray,
    W_g: np.ndarray,
    TH_g: np.ndarray,
    *,
    theta_scale: float,
    forced: float | None = None,
) -> float:
    """Pick SHARPy twist sign (+1 or -1) that maximizes GINO↔SHARPy pairing MAC."""
    if forced is not None:
        return float(forced)
    order_g = np.argsort(y_g)
    nm_g = W_g.shape[0]
    nm_s = W_sh.shape[0]

    def _mac_sum(sgn: float) -> float:
        tot = 0.0
        for g in range(nm_g):
            wg = np.interp(y_sh, y_g[order_g], W_g[g][order_g])
            tg = np.interp(y_sh, y_g[order_g], TH_g[g][order_g])
            v_g = np.concatenate([wg, theta_scale * tg])
            v_g /= max(np.linalg.norm(v_g), 1e-30)
            best = 0.0
            for s in range(nm_s):
                v_s = np.concatenate([W_sh[s], sgn * theta_scale * TH_sh[s]])
                v_s /= max(np.linalg.norm(v_s), 1e-30)
                best = max(best, float(np.dot(v_g, v_s) ** 2))
            tot += best
        return tot

    return 1.0 if _mac_sum(1.0) >= _mac_sum(-1.0) else -1.0


def build_goland_congruence(
    y_sh: np.ndarray,
    W_sh: np.ndarray,
    TH_sh: np.ndarray,
    y_g: np.ndarray,
    W_g: np.ndarray,
    TH_g: np.ndarray,
    *,
    theta_scale: float,
    projection: str = "w_lsq",
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """
    Build GINO←SHARPy modal congruence ``C`` (``q_sh = C @ q_gino`` amplitude map).

    ``projection``:
      - ``identity``: signed permutation from plunge MAC only (±1, no LSQ scale).
      - ``w_lsq``: LSQ scalar from plunge channel only (ignores twist mismatch on bend).
      - ``full_lsq``: legacy [w; θ] combined LSQ (can mis-scale bend modes).
    """
    projection = str(projection).lower()
    if projection not in ("identity", "w_lsq", "full_lsq"):
        raise ValueError(f"projection must be identity, w_lsq, or full_lsq; got {projection!r}")

    nm_g = W_g.shape[0]
    nm_s = W_sh.shape[0]
    order_g = np.argsort(y_g)
    c = np.zeros((nm_g, nm_s))
    pairs: list[tuple[int, int, float]] = []
    used_s: set[int] = set()

    def _vec(w, th):
        v = np.concatenate([w, theta_scale * th])
        return v / max(np.linalg.norm(v), 1e-30)

    for g in range(nm_g):
        wg = np.interp(y_sh, y_g[order_g], W_g[g][order_g])
        tg = np.interp(y_sh, y_g[order_g], TH_g[g][order_g])
        v_g_unit = _vec(wg, tg)
        macs = []
        for s in range(nm_s):
            v_s_unit = _vec(W_sh[s], TH_sh[s])
            macs.append(float(np.dot(v_g_unit, v_s_unit) ** 2))
        s_best = int(np.argmax([m if s not in used_s else -1.0 for s, m in enumerate(macs)]))
        used_s.add(s_best)
        pairs.append((g, s_best, macs[s_best]))

        w_sh = W_sh[s_best]
        if projection == "identity":
            sign = float(np.sign(np.dot(wg, w_sh)) or 1.0)
            c[g, s_best] = sign
        elif projection == "w_lsq":
            c[g, s_best] = float(w_sh @ wg) / max(float(w_sh @ w_sh), 1e-30)
        else:
            v_s = np.concatenate([w_sh, theta_scale * TH_sh[s_best]])
            v_g = np.concatenate([wg, theta_scale * tg])
            c[g, s_best] = float(v_s @ v_g) / max(float(v_s @ v_s), 1e-30)

    return c, pairs
