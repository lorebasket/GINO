"""
Native aeroelastic state-space coupling (no p-k, no Roger refit).

Couples a precomputed SHARPy linear-UVLM aerodynamic ROM (states, sampled
once in the ``sharpy`` conda env by ``FLUID/sharpy/gino_bem/precompute_uvlm_rom_goland.py``
or the Model2t/Abramson equivalent) with an arbitrary modal structural
operator (M_hat, K_hat — SHARPy's own for parity checks, GINO/SONATA's own
for production) using the exact same recipe SHARPy itself uses internally
(``sharpy.linear.src.lingebm.newmark_ss`` + ``sharpy.linear.src.libss.couple``),
ported here so the coupling + eigenvalue sweep runs with no SHARPy runtime
dependency (numpy/scipy only).

Validated 2026-07-20 on the Goland case: reproduces SHARPy's own native
coupled eigenvalues (both structural branches, all velocities) to <0.01
rad/s, for both SHARPy's own structural basis and GINO's own basis
(congruence-mapped aero ROM).

Physics/scaling recap (from ``sharpy.linear.src.lingebm.FlexDynamic``):
  - the aero ROM (A, B_in, C_out, D_out) is built ONCE at a reference
    velocity and stays fixed in *nondimensional* time (fixed ``dt``);
  - only the structural operator is rescaled per velocity via
    ``t_ref = length_ref / V``: ``K_nondim = K_hat * t_ref**2``,
    ``C_nondim = C_hat * t_ref``, ``M_hat`` unchanged (valid for a
    mass-normalized modal basis, M_hat = I);
  - the aero<->structure coupling gains (K12 mapping structural [q, q_dot]
    into the aero ROM's inputs, K21 mapping the aero ROM's modal force
    output back into the structural force input) are FIXED, V-independent
    constants — the V-dependence lives entirely in the structural
    nondimensionalization above, not in the coupling gains.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import linalg as la


def newmark_ss_forced(M, C, K, dt, num_damp=1e-4, *, M_is_spd=False):
    """
    Forced discrete-time state-space for ``M q_ddot + C q_dot + K q = f``.

    Direct port of ``sharpy.linear.src.lingebm.newmark_ss`` (a standalone
    function with no dependency on SHARPy's ``FlexDynamic`` object/state).
    State is ``[q; q_dot]`` (2n,); output ``y = [q; q_dot]`` (``Css = I``);
    input is the generalized force ``f`` (n,).

    Returns
    -------
    Ass, Bss, Css, Dss : ndarray
        Discrete state-space matrices, ``x_{k+1} = Ass x_k + Bss f_k``,
        ``y_k = Css x_k + Dss f_k``.
    """
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    th1 = 0.5 + float(num_damp)
    th2 = 0.0625 + 0.25 * (th1 + th1 ** 2)

    dt2 = dt ** 2
    a1 = th2 * dt2
    a0 = 0.5 * dt2 - a1
    b1 = th1 * dt
    b0 = dt - b1

    n = K.shape[0]
    imat = np.eye(n)

    M_factors = la.cho_factor(M) if M_is_spd else la.lu_factor(M)

    def m_solve(b):
        return la.cho_solve(M_factors, b) if M_is_spd else la.lu_solve(M_factors, b)

    minv = m_solve(imat)
    minv_k = m_solve(K)
    minv_c = m_solve(C)

    ass0 = np.block([[imat - a0 * minv_k, dt * imat - a0 * minv_c],
                      [-b0 * minv_k, imat - b0 * minv_c]])
    ass1 = np.block([[imat + a1 * minv_k, a1 * minv_c],
                      [b1 * minv_k, imat + b1 * minv_c]])

    ass1_factors = la.lu_factor(ass1)
    ass = la.lu_solve(ass1_factors, ass0)
    bss0 = la.lu_solve(ass1_factors, np.block([[a0 * minv], [b0 * minv]]))
    bss1 = la.lu_solve(ass1_factors, np.block([[a1 * minv], [b1 * minv]]))

    # eliminate the u_{n+1} predictor term (libss.SSconv with B0=bss0, B1=bss1,
    # C=I, D=0): Bh = B0 + A @ B1 ; Dh = D + C @ B1 = B1
    bss = bss0 + ass @ bss1
    css = np.eye(2 * n)
    dss = bss1
    return ass, bss, css, dss


def couple_A(A1, B1, C1, D1, A2, B2, C2, D2, K12, K21):
    """
    Dynamics matrix ``Ac`` of two coupled discrete-time state-space systems
    (ss1, ss2), through gains ``K12`` (ss2 output -> ss1 input) and ``K21``
    (ss1 output -> ss2 input) — the ``A``-block of the algebra in
    ``sharpy.linear.src.libss.couple`` (dense path only; ``Bc``/``Dc`` are
    not needed here since only the free-vibration eigenvalue problem, with
    no external forcing, is used throughout this module).
    """
    k11 = K12 @ (D2 @ K21)
    k22 = K21 @ (D1 @ K12)
    l1 = np.eye(k11.shape[0]) - k11 @ D1
    l2 = np.eye(k22.shape[0]) - k22 @ D2
    cpl_12 = np.linalg.solve(l1, K12)
    cpl_21 = np.linalg.solve(l2, K21)
    cpl_11 = cpl_12 @ (D2 @ K21)
    cpl_22 = cpl_21 @ (D1 @ K12)
    return np.block([
        [A1 + B1 @ cpl_11 @ C1, B1 @ cpl_12 @ C2],
        [B2 @ cpl_21 @ C1, A2 + B2 @ cpl_22 @ C2],
    ])


def load_uvlm_rom(npz_path: str) -> dict:
    """Load a precomputed UVLM aero ROM (see ``precompute_uvlm_rom_goland.py``)."""
    d = np.load(npz_path, allow_pickle=True)
    return {
        "A": np.asarray(d["A"], dtype=float),
        "B_in": np.asarray(d["B_in"], dtype=float),
        "C_out": np.asarray(d["C_out"], dtype=float),
        "D_out": np.asarray(d["D_out"], dtype=float),
        "dt_nondim": float(d["dt_nondim"]),
        "length_ref": float(d["length_ref"]),
        "V_construction": float(d["V_construction"]),
        "nm": int(d["nm"]),
        "K12": np.asarray(d["K12"], dtype=float),
        "K21": np.asarray(d["K21"], dtype=float),
    }


def apply_basis_congruence(rom: dict, C_cong: np.ndarray, surface_factor: float = 1.0) -> dict:
    """
    Re-express an aero ROM built in SHARPy's own modal basis into a
    different (but same-size) structural basis via a linear congruence
    ``C_cong`` (``nm x nm``; diagonal for the Goland w_lsq projection, dense
    in general), following the same convention already validated for the
    frequency-sampled Quvlm tables: ``Q_new(k) = surface_factor * C_cong @
    Q_sharpy(k) @ C_cong.T``.

    ``K12``/``K21`` (unit-conversion gains, not modal-basis-dependent) are
    carried over unchanged from ``rom``.

    Returns a new ROM dict (same aero states/A, remapped B_in/C_out/D_out).
    """
    nm = rom["nm"]
    if C_cong.shape != (nm, nm):
        raise ValueError(f"C_cong shape {C_cong.shape} != (nm, nm)=({nm}, {nm})")
    Cblock = np.block([
        [C_cong, np.zeros((nm, nm))],
        [np.zeros((nm, nm)), C_cong],
    ])
    new = dict(rom)
    new["B_in"] = rom["B_in"] @ Cblock.T
    new["C_out"] = surface_factor * (C_cong @ rom["C_out"])
    new["D_out"] = surface_factor * (C_cong @ rom["D_out"] @ Cblock.T)
    return new


def native_coupled_eigs_at_V(
    V: float,
    rom: dict,
    M_hat: np.ndarray,
    K_hat: np.ndarray,
    C_hat: Optional[np.ndarray] = None,
    newmark_damp: float = 0.5e-4,
    *,
    return_participation: bool = False,
):
    """
    Physical-units eigenvalues (rad/s, complex ``sigma + i*omega``) of the
    coupled aero+structure system at a single velocity ``V``.

    ``M_hat``/``K_hat``/``C_hat`` are the (mass-normalized: M_hat = I is the
    simplest valid choice) modal structural operator in PHYSICAL units —
    from SHARPy's own beam for validation, or from GINO/SONATA in production.

    If ``return_participation`` is True, also returns each eigenvalue's
    structural participation ratio (``||eigenvector on the [q, q_dot] block||
    / ||eigenvector||``) — needed to discriminate physical structural
    branches from the aero ROM's own (Krylov-lag) poles, which sit at
    similar frequencies but carry negligible structural participation
    (empirically ~1e-4 vs ~0.05-0.5 for genuine structural branches) — and
    the (unit-normalized) structural [q, q_dot] sub-block of each
    eigenvector, for MAC-based branch tracking across velocity steps
    (:func:`sweep_native_coupling_with_hydro`) — plain nearest-frequency
    continuity is not robust once two branches' frequencies approach each
    other, or a spurious aero-ROM pole briefly sits close in frequency to a
    genuine branch.
    """
    if C_hat is None:
        C_hat = np.zeros_like(M_hat)

    t_ref = rom["length_ref"] / float(V)
    K_nd = K_hat * t_ref ** 2
    C_nd = C_hat * t_ref

    Ass_b, Bss_b, Css_b, Dss_b = newmark_ss_forced(
        M_hat, C_nd, K_nd, rom["dt_nondim"], newmark_damp,
    )

    Ac = couple_A(
        rom["A"], rom["B_in"], rom["C_out"], rom["D_out"],
        Ass_b, Bss_b, Css_b, Dss_b,
        rom["K12"], rom["K21"],
    )

    if not return_participation:
        mu = np.linalg.eigvals(Ac)
        s_nd = np.log(mu.astype(complex)) / rom["dt_nondim"]
        return s_nd / t_ref

    mu, vecs = np.linalg.eig(Ac)
    s_nd = np.log(mu.astype(complex)) / rom["dt_nondim"]
    s_phys = s_nd / t_ref
    n_aero = rom["A"].shape[0]
    vec_norm = np.linalg.norm(vecs, axis=0)
    struct_vecs = vecs[n_aero:, :]
    struct_norm = np.linalg.norm(struct_vecs, axis=0)
    participation = struct_norm / np.maximum(vec_norm, 1e-300)
    struct_vecs_unit = struct_vecs / np.maximum(struct_norm, 1e-300)
    return s_phys, participation, struct_vecs_unit


def evaluate_rom_Q_nondim(rom: dict, w_s: float) -> np.ndarray:
    """
    Raw (pre-``K21``) modal frequency response of the aero ROM at a given
    NONDIMENSIONAL angular frequency ``w_s`` (``= k_pa * length_ref``,
    independent of any reference velocity — see module docstring scaling
    recap), combining its separate ``q``/``q_dot`` input channels via the
    harmonic relation ``q_dot = i*w_s*q`` (both expressed in the SAME shared
    nondimensional time the coupled system uses): ``Q_raw(w_s) = H_q(w_s) +
    i*w_s*H_qdot(w_s)``, ``H(w_s) = D_out + C_out @ (zI-A)^-1 @ B_in`` (shape
    ``(nm, 2*nm)``, columns ``[q | q_dot]``), ``z = exp(i*w_s*dt_nondim)``.
    Shape ``(nm, nm)``.

    This is the ROM's raw output, in its own internal (nondimensional-time)
    units — NOT yet the physical generalized force (:func:`fit_rom_apparent_mass`
    applies the ``K21`` unit-conversion gain and undoes the extra ``t_ref**2``
    baked into the coupled system's RHS — see that function's docstring for
    the full derivation). The coupled-eigenvalue path itself never needs
    this function: it consumes the ROM's states directly through
    ``couple_A``, which already applies ``K21`` correctly for that separate
    (V-parametrized-via-t_ref) formulation.
    """
    n = rom["A"].shape[0]
    nm = rom["nm"]
    z = np.exp(1j * w_s * rom["dt_nondim"])
    H = rom["D_out"] + rom["C_out"] @ np.linalg.solve(
        z * np.eye(n) - rom["A"], rom["B_in"]
    )
    H_q, H_qdot = H[:, :nm], H[:, nm:]
    return H_q + 1j * w_s * H_qdot


def fit_rom_apparent_mass(rom: dict, w_s_high: Optional[float] = None) -> np.ndarray:
    """
    The aero ROM's own non-circulatory (apparent) mass, in the SAME physical
    (mass-normalized-modal) units as ``M_hat`` — directly addable to it.

    Derivation (native-coupling framework, NOT the old qinf*Q(k) p-k
    convention — the two are different mathematical formulations and do not
    share intermediate quantities): nondimensionalizing the physical EOM
    ``M_hat*q_ddot_phys + K_hat*q = f_phys`` by ``tau = t_phys/t_ref`` gives
    ``M_hat*q'' + K_hat*t_ref**2*q = f_phys*t_ref**2`` (this is exactly the
    ``K_nd = K_hat*t_ref**2`` already used in :func:`native_coupled_eigs_at_V`
    — so by construction the coupled system's forcing RHS equals
    ``f_phys*t_ref**2``, not ``f_phys`` directly). The aero ROM's own
    harmonic response at nondimensional frequency ``w_s`` (``= omega_phys *
    t_ref``) supplies ``K21 @ Q_raw(w_s) @ q`` on that same RHS. At high
    frequency this response is dominated by the non-circulatory (in phase
    with ``-q_ddot``, i.e. real, mass-like) term: writing ``f_phys ~=
    M_apparent * omega_phys**2 * q`` and substituting
    ``omega_phys = w_s/t_ref`` gives, after equating both sides and the
    ``t_ref**2`` factors cancel exactly (independent of V, as it must for a
    ROM whose own dynamics are fixed and V-independent):

        M_apparent = Re(K21 @ Q_raw(w_s)) / w_s**2

    No extra ``rho`` or ``qinf``-like factor: ``K21`` already carries the
    density used when the SHARPy case was linearised (it is the same fixed
    gain validated bit-for-bit against SHARPy's own native coupling).

    ``w_s_high`` defaults to 75% of the ROM's own Nyquist limit
    (``pi/dt_nondim``) — high enough to be past the circulatory/lag
    dynamics, safely clear of aliasing.
    """
    if w_s_high is None:
        w_s_high = 0.75 * np.pi / rom["dt_nondim"]
    Q_phys = rom["K21"] @ evaluate_rom_Q_nondim(rom, w_s_high)
    M_app = np.real(Q_phys) / (w_s_high ** 2)
    return 0.5 * (M_app + M_app.T)


def _select_tracked_branches(
    lam: np.ndarray,
    struct_vecs: np.ndarray,
    part: np.ndarray,
    targets_omega: np.ndarray,
    prev_vecs: list,
    *,
    omega_phys_min: float,
    participation_min: float,
):
    """
    Select ``len(targets_omega)`` branches from one velocity step's candidate
    eigenvalues: MAC against ``prev_vecs[b]`` (structural sub-block from the
    previous step) when available, else nearest-frequency to
    ``targets_omega[b]`` (first step / lost branch). Processed in ascending
    target-frequency order with an exclusivity constraint — once an
    eigenvalue is claimed by one branch it is unavailable to the rest — so
    two branches never collapse onto the same root, a real failure mode of
    plain per-branch nearest-frequency tracking once two branches approach
    each other in frequency or a spurious aero-ROM pole briefly sits close
    in frequency to a genuine one.

    Returns (real, imag, vecs) each length ``len(targets_omega)`` (``vecs``
    is a list, ``None`` entries where no candidate was available).
    """
    from .mode_matching import compute_mac

    mask = (np.imag(lam) > omega_phys_min) & (part > participation_min)
    lam_f = lam[mask]
    vecs_f = struct_vecs[:, mask]

    n_track = len(targets_omega)
    real_out = np.full(n_track, np.nan)
    imag_out = np.full(n_track, np.nan)
    vecs_out: list = [None] * n_track
    claimed = np.zeros(lam_f.size, dtype=bool)

    for b in np.argsort(targets_omega):
        avail = np.flatnonzero(~claimed)
        if avail.size == 0:
            break
        if prev_vecs[b] is not None:
            macs = np.array([compute_mac(prev_vecs[b], vecs_f[:, i]) for i in avail])
            best = avail[int(np.argmax(macs))]
        else:
            best = avail[int(np.argmin(np.abs(np.imag(lam_f[avail]) - targets_omega[b])))]
        real_out[b] = float(np.real(lam_f[best]))
        imag_out[b] = float(np.imag(lam_f[best]))
        vecs_out[b] = vecs_f[:, best]
        claimed[best] = True

    return real_out, imag_out, vecs_out


def native_coupled_eigs_with_hydro(
    V: float,
    rom: dict,
    M_hat: np.ndarray,
    K_hat: np.ndarray,
    M_hydro: np.ndarray,
    capytaine_damping_fn,
    *,
    C_hat: Optional[np.ndarray] = None,
    newmark_damp: float = 0.5e-4,
    n_track: Optional[int] = None,
    track_seed_omegas: Optional[np.ndarray] = None,
    prev_vecs: Optional[list] = None,
    n_iter: int = 2,
    omega_phys_min: float = 1.0,
    participation_min: float = 1e-2,
) -> dict:
    """
    Single-velocity coupled eigenvalue solve with an external (Capytaine)
    hydrodynamic contribution: constant added mass ``M_hydro`` (already
    embedded on the full modal DOF space — folded directly into ``M_hat``,
    valid since Capytaine's added mass is essentially frequency-independent
    over the flutter band on Model2t, <0.1% variation, verified 2026-07-20)
    plus radiation damping evaluated via ``capytaine_damping_fn(omega) ->
    (n,n) ndarray`` (embedded on the full modal DOF space), which IS
    strongly frequency-dependent (~25x drop over the flutter band) and is
    therefore solved **per branch**, mirroring p-k's own per-branch
    structure, but with only ``n_iter`` light fixed-point evaluations
    (not full k-convergence) — the aero ROM's own circulatory dynamics
    already capture k-dependence exactly through its states, so only the
    damping table lookup needs iterating.

    Branch identity within each per-branch iteration (and across branches,
    via the exclusivity constraint) is resolved by MAC against
    ``prev_vecs[b]`` when available (see :func:`_select_tracked_branches`),
    not by nearest-frequency alone.

    Returns
    -------
    dict with keys "real", "imag" (each shape (n_track,)), "omega_iter"
    (converged per-branch frequency used for the final damping lookup), and
    "vecs" (list of structural eigenvectors, for the next velocity step).
    """
    n_modes = M_hat.shape[0]
    if n_track is None:
        n_track = n_modes
    if track_seed_omegas is None:
        track_seed_omegas = np.sqrt(np.abs(np.diag(K_hat) / np.maximum(np.diag(M_hat), 1e-30)))
    track_seed_omegas = np.asarray(track_seed_omegas, dtype=float)[:n_track]
    if prev_vecs is None:
        prev_vecs = [None] * n_track

    M_hat_eff = M_hat + M_hydro
    omega_est = track_seed_omegas.copy()
    real = np.full(n_track, np.nan)
    imag = np.full(n_track, np.nan)
    vecs_out: list = list(prev_vecs)

    for _ in range(max(1, n_iter)):
        # One shared damping table lookup per branch's CURRENT frequency
        # estimate, but the eigenvalue problem itself (aero ROM + structure)
        # is a single system — build it once per unique-ish estimate is not
        # worth the complexity here; solve per branch, mirroring p-k.
        any_iter_changed = False
        for b in range(n_track):
            C_hydro = capytaine_damping_fn(float(omega_est[b]))
            C_hat_eff = (np.zeros_like(M_hat) if C_hat is None else C_hat) + C_hydro
            lam, part, struct_vecs = native_coupled_eigs_at_V(
                V, rom, M_hat_eff, K_hat, C_hat_eff, newmark_damp,
                return_participation=True,
            )
            r, i, v = _select_tracked_branches(
                lam, struct_vecs, part,
                targets_omega=omega_est, prev_vecs=vecs_out,
                omega_phys_min=omega_phys_min, participation_min=participation_min,
            )
            if np.isfinite(i[b]):
                if abs(i[b] - omega_est[b]) > 1e-9:
                    any_iter_changed = True
                omega_est[b] = i[b]
                real[b], imag[b] = r[b], i[b]
                vecs_out[b] = v[b]
        if not any_iter_changed:
            break

    return {"real": real, "imag": imag, "omega_iter": omega_est, "vecs": vecs_out}


def sweep_native_coupling_with_hydro(
    V_list: np.ndarray,
    rom: dict,
    M_hat: np.ndarray,
    K_hat: np.ndarray,
    M_hydro: np.ndarray,
    capytaine_damping_fn,
    *,
    C_hat: Optional[np.ndarray] = None,
    newmark_damp: float = 0.5e-4,
    n_track: Optional[int] = None,
    track_seed_omegas: Optional[np.ndarray] = None,
    n_iter: int = 2,
) -> dict:
    """Sweep :func:`native_coupled_eigs_with_hydro` over ``V_list``, tracking
    branches by MAC continuity (seeded from ``track_seed_omegas`` at the
    first step, then chained via the previous velocity step's converged
    frequency AND eigenvector)."""
    n_modes = M_hat.shape[0]
    if n_track is None:
        n_track = n_modes
    if track_seed_omegas is None:
        track_seed_omegas = np.sqrt(np.abs(np.diag(K_hat) / np.maximum(np.diag(M_hat), 1e-30)))
    seed = np.asarray(track_seed_omegas, dtype=float)[:n_track].copy()
    prev_vecs: list = [None] * n_track

    V_arr = np.asarray(V_list, dtype=float)
    real = np.full((n_track, V_arr.size), np.nan)
    imag = np.full((n_track, V_arr.size), np.nan)

    for iv, V in enumerate(V_arr):
        res = native_coupled_eigs_with_hydro(
            V, rom, M_hat, K_hat, M_hydro, capytaine_damping_fn,
            C_hat=C_hat, newmark_damp=newmark_damp, n_track=n_track,
            track_seed_omegas=seed, prev_vecs=prev_vecs, n_iter=n_iter,
        )
        real[:, iv] = res["real"]
        imag[:, iv] = res["imag"]
        valid = np.isfinite(res["imag"])
        seed[valid] = res["imag"][valid]
        for b in range(n_track):
            if valid[b]:
                prev_vecs[b] = res["vecs"][b]

    return {"V": V_arr, "real": real, "imag": imag}


def sweep_native_coupling(
    V_list: np.ndarray,
    rom: dict,
    M_hat: np.ndarray,
    K_hat: np.ndarray,
    *,
    C_hat: Optional[np.ndarray] = None,
    newmark_damp: float = 0.5e-4,
    n_track: Optional[int] = None,
    track_seed_omegas: Optional[np.ndarray] = None,
    omega_phys_min: float = 1.0,
    participation_min: float = 1e-2,
) -> dict:
    """
    Sweep ``native_coupled_eigs_at_V`` over ``V_list`` and track ``n_track``
    branches by MAC continuity (seeded from ``track_seed_omegas``, e.g. the
    dry structural frequencies, at the first step; nearest-frequency only as
    a fallback when a branch has no previous eigenvector — e.g. lost/re-
    acquired), restricted to eigenvalues with structural participation
    above ``participation_min`` — the aero ROM's own (Krylov-lag) poles can
    sit at frequencies very close to a genuine structural branch
    (empirically within ~1 rad/s on Goland) but carry negligible structural
    participation (~1e-4 vs ~0.05-0.5); without this filter, tracking can
    silently lock onto a spurious aero pole. An exclusivity constraint
    additionally prevents two branches from ever claiming the same
    eigenvalue (see :func:`_select_tracked_branches`).

    Returns
    -------
    dict with keys "V", "real" (n_track, nV), "imag" (n_track, nV).
    """
    n_modes = M_hat.shape[0]
    if n_track is None:
        n_track = n_modes
    if track_seed_omegas is None:
        track_seed_omegas = np.sqrt(np.abs(np.diag(K_hat) / np.maximum(np.diag(M_hat), 1e-30)))
    targets = np.asarray(track_seed_omegas, dtype=float)[:n_track].copy()
    prev_vecs: list = [None] * n_track

    V_arr = np.asarray(V_list, dtype=float)
    real = np.full((n_track, V_arr.size), np.nan)
    imag = np.full((n_track, V_arr.size), np.nan)

    for iv, V in enumerate(V_arr):
        lam, part, struct_vecs = native_coupled_eigs_at_V(
            V, rom, M_hat, K_hat, C_hat, newmark_damp, return_participation=True,
        )
        r, i, v = _select_tracked_branches(
            lam, struct_vecs, part, targets, prev_vecs,
            omega_phys_min=omega_phys_min, participation_min=participation_min,
        )
        real[:, iv] = r
        imag[:, iv] = i
        valid = np.isfinite(i)
        targets[valid] = i[valid]
        for b in range(n_track):
            if valid[b]:
                prev_vecs[b] = v[b]

    return {"V": V_arr, "real": real, "imag": imag}
