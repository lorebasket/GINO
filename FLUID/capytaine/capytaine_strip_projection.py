"""
Project Capytaine rigid-body added mass onto structural modes the same way as strip theory.

Strip theory path
-----------------
1. Build a physical lumped matrix ``M_add`` on transverse translational beam DOFs.
2. Modal generalized mass: ``M_hat_add = Phi^T M_add Phi`` (mass-normalized ``Phi``).

Capytaine BEM modal radiation (``run_modal_radiation.py``) instead solves face fields per
mode and returns ``A_ij(omega)`` directly in modal space. That is **not** comparable to strip
when normal velocity on the hull is small.

This module takes the **asymptotic rigid-body added-mass tensor** ``A_∞`` (6×6) from
Capytaine, maps the translational block into the structural reference frame, converts each
transverse diagonal to an effective ``m_a' = A_ii / L``, and reuses the **same lumping and
projection** as ``FEA.structural_analysis.added_mass_projection``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import numpy as np

_RIGID_DOF_NAMES = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")


def _depth_npz_suffix(depth: float) -> str:
    return f"_depth_{depth:g}".replace(".", "p").replace("-", "m")


def default_rigid_radiation_npz_path(config, depth: float | None = None) -> str:
    """Default path for rigid-body Capytaine radiation NPZ."""
    capy_dir = str(getattr(config, "capytaine_results_dir", "") or "").strip()
    rigid_dir = str(
        getattr(config, "capytaine_rigid_results_dir", "") or os.path.join(capy_dir, "rigid_body")
    ).strip()
    if depth is None:
        depth = float(np.asarray(getattr(config, "depth", 0.0)).ravel()[int(getattr(config, "depth_index", 0))])
    return os.path.join(rigid_dir, f"modal_radiation_rigid_AB{_depth_npz_suffix(depth)}.npz")


def asymptotic_added_mass_6x6(capy_data: dict[str, Any], *, omega_cut_fraction: float = 0.85) -> np.ndarray:
    """
    High-ω plateau of the Capytaine added-mass matrix (n_modes × n_modes), symmetrized.
    """
    omega = np.asarray(capy_data["omega"], dtype=float).ravel()
    A = np.asarray(capy_data["added_mass"], dtype=float)
    if A.ndim != 3:
        raise ValueError(f"Expected added_mass (n_omega, n, n), got {A.shape}")
    if omega.size == 0:
        raise ValueError("Empty omega grid in Capytaine data.")
    i0 = int(max(0, min(omega.size - 1, round(omega_cut_fraction * (omega.size - 1)))))
    A_tail = 0.5 * (A[i0:] + np.transpose(A[i0:], (0, 2, 1)))
    return np.mean(A_tail, axis=0)


def load_rigid_capytaine_npz(path: str) -> dict[str, Any]:
    from modal_radiation_io import load_modal_radiation_npz

    data = load_modal_radiation_npz(path)
    names = [str(x) for x in np.asarray(data.get("mode_names", [])).ravel().tolist()]
    n = int(np.asarray(data["added_mass"]).shape[1])
    if n != 6:
        raise ValueError(f"Expected 6 rigid-body modes in {path}, got n_modes={n}.")
    if names and names != list(_RIGID_DOF_NAMES):
        print(
            f"Warning: {path} mode_names={names!r} differ from Capytaine rigid labels; "
            f"using first 6×6 block anyway."
        )
    return data


def m_a_prime_per_axis_from_rigid_added_mass(
    A6: np.ndarray,
    config,
    *,
    R_beam_to_global: np.ndarray | None = None,
) -> np.ndarray:
    """
    Effective strip-style ``m_a'`` [kg/m] per global reference axis from Capytaine ``A_∞`` (6×6).

    Uses the translational 3×3 block, mapped to the structural reference frame via
    ``R_beam_to_global.T @ M_t @ R_beam_to_global`` (same ``R`` as beam→hydro mesh).
    """
    A6 = np.asarray(A6, dtype=float)
    if A6.shape != (6, 6):
        raise ValueError(f"Expected (6, 6) rigid added mass, got {A6.shape}.")
    A6 = 0.5 * (A6 + A6.T)
    M_t_hydro = A6[:3, :3]

    if R_beam_to_global is None:
        from run_modal_radiation import _beam_to_global_rotation_from_cfg

        R_beam_to_global = _beam_to_global_rotation_from_cfg(config)
    R = np.asarray(R_beam_to_global, dtype=float).reshape(3, 3)
    M_t_ref = R.T @ M_t_hydro @ R

    L = float(config.beam_length)
    if L <= 0.0:
        raise ValueError("config.beam_length must be positive.")

    span_axis = int(getattr(config, "strip_theory_span_axis", 1))
    m_a_prime = np.zeros(3, dtype=float)
    for ax in range(3):
        if ax == span_axis:
            continue
        m_a_prime[ax] = float(M_t_ref[ax, ax]) / L
    return m_a_prime, M_t_ref, M_t_hydro


def mass_matrix_capytaine_strip_style(
    M_global,
    config,
    A6_rigid_asymptotic: np.ndarray,
    *,
    scale: float | None = None,
) -> np.ndarray:
    """
    Physical added-mass matrix lumped like strip theory, with ``m_a'`` from Capytaine rigid body.
    """
    from FEA.structural_analysis import mass_matrix_lumped_transverse

    m_a_prime, M_t_ref, M_t_hydro = m_a_prime_per_axis_from_rigid_added_mass(
        A6_rigid_asymptotic, config
    )
    if scale is None:
        scale = float(getattr(config, "added_mass_strip_scale", 1.0))
    print(
        "[capytaine strip-style] rigid translational A_∞ (hydro frame) diag "
        f"({M_t_hydro[0,0]:g}, {M_t_hydro[1,1]:g}, {M_t_hydro[2,2]:g}) kg"
    )
    print(
        "[capytaine strip-style] mapped to structural ref, m_a' [kg/m] "
        f"(X,Y,Z)=({m_a_prime[0]:g}, {m_a_prime[1]:g}, {m_a_prime[2]:g})"
    )
    return mass_matrix_lumped_transverse(
        M_global,
        config,
        m_a_prime,
        scale=float(scale),
        log_prefix="[capytaine strip-style]",
    )


def capytaine_added_mass_projection_like_strip(
    structural_results,
    config,
    capy_rigid_data: dict[str, Any] | None = None,
    *,
    rigid_npz_path: str | None = None,
    mode_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ``Phi^T M_add Phi`` with ``M_add`` from Capytaine rigid-body ``A_∞`` lumped like strip.

    Returns
    -------
    M_hat_added, Phi_for_reduction
    """
    from FEA.structural_analysis import added_mass_projection

    if capy_rigid_data is None:
        path = rigid_npz_path or default_rigid_radiation_npz_path(config)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Rigid-body Capytaine NPZ not found: {path}\n"
                "Run: python FLUID/capytaine/run_modal_radiation.py --case-name <CASE> --rigid-body "
                f"--out-dir {os.path.dirname(path)}"
            )
        capy_rigid_data = load_rigid_capytaine_npz(path)

    A6 = asymptotic_added_mass_6x6(capy_rigid_data)
    M_physical = mass_matrix_capytaine_strip_style(
        structural_results.M_global, config, A6
    )

    if mode_indices is None:
        n = int(getattr(config, "num_modes_egv", structural_results.dry_eigenvectors.shape[1]))
        mode_indices = list(range(n))

    apply_dihedral = (
        abs(float(getattr(config, "dihedral_angle", 0.0))) > 1e-12
        and bool(getattr(config, "dihedral_rotate_beam", True))
    )

    return added_mass_projection(
        structural_results.M_global,
        structural_results.dry_eigenvectors,
        mode_indices,
        config,
        structural_results.total_dof,
        constrained_dofs=structural_results.constrained_dofs,
        apply_structural_dihedral=apply_dihedral,
        M_added_physical=M_physical,
        method_label="Capytaine (strip-style projection)",
    )


def default_stripstyle_export_npz_path(config) -> str:
    """Cached strip-style generalized added-mass matrix for flutter (RFA_PK)."""
    capy_dir = str(getattr(config, "capytaine_results_dir", "") or "").strip()
    return os.path.join(capy_dir, "capytaine_stripstyle_M_hat_added.npz")


def load_or_compute_stripstyle_M_hat_added(
    structural_results,
    config,
    *,
    rigid_npz_path: str | None = None,
    export_npz: bool = True,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Generalized added-mass matrix ``M_hat_added`` from Capytaine rigid ``A_∞`` + strip lumping.

    Returns ``(M_hat_added, Phi_for_reduction, rigid_npz_path_used)``.
    Optionally writes ``capytaine_stripstyle_M_hat_added.npz`` under ``capytaine_results_dir``.
    """
    rigid_path = rigid_npz_path or default_rigid_radiation_npz_path(config)
    M_hat_added, Phi = capytaine_added_mass_projection_like_strip(
        structural_results,
        config,
        rigid_npz_path=rigid_path,
    )
    if export_npz:
        out = default_stripstyle_export_npz_path(config)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        np.savez(
            out,
            M_hat_added=np.asarray(M_hat_added, dtype=float),
            Phi_for_reduction=np.asarray(Phi, dtype=float),
            modal_diag=np.diag(M_hat_added),
            rigid_npz_path=rigid_path,
            method="capytaine_stripstyle_projection",
            added_mass_modal_basis="mass_normalized",
        )
        print(f"[capytaine strip-style] exported M_hat_added → {out}")
    return M_hat_added, Phi, rigid_path


def compare_strip_bem_stripstyle_projection(
    structural_results,
    config,
    *,
    capy_modal_data: dict[str, Any] | None = None,
    capy_modal_npz: str | None = None,
    capy_rigid_data: dict[str, Any] | None = None,
    capy_rigid_npz: str | None = None,
    n_modes: int | None = None,
) -> dict[str, Any]:
    """
    Compare three modal added-mass diagonals on the same ``Phi``:

    - strip: ``m_a' = rho pi c^2 / 4`` lumped + projected
    - capy_bem: native Capytaine modal radiation ``A_∞`` (BEM face fields)
    - capy_strip: Capytaine rigid ``A_∞`` lumped + projected like strip
    """
    from FEA.structural_analysis import added_mass_projection
    from modal_radiation_io import load_modal_radiation_npz

    if n_modes is None:
        n_modes = int(getattr(config, "num_modes_egv", 4))
    mode_indices = list(range(n_modes))

    M_strip, _ = added_mass_projection(
        structural_results.M_global,
        structural_results.dry_eigenvectors,
        mode_indices,
        config,
        structural_results.total_dof,
        constrained_dofs=structural_results.constrained_dofs,
        method_label="strip theory",
    )
    diag_strip = np.diag(M_strip)[:n_modes]

    M_capy_strip = None
    diag_capy_strip = np.full(n_modes, np.nan)
    capy_strip_over_strip = [float("nan")] * n_modes
    if capy_rigid_data is not None or (
        capy_rigid_npz is not None and os.path.isfile(capy_rigid_npz)
    ):
        M_capy_strip, _ = capytaine_added_mass_projection_like_strip(
            structural_results,
            config,
            capy_rigid_data=capy_rigid_data,
            rigid_npz_path=capy_rigid_npz,
            mode_indices=mode_indices,
        )
        diag_capy_strip = np.diag(M_capy_strip)[:n_modes]
        capy_strip_over_strip = [
            float(c / s) if abs(s) > 1e-14 else float("nan")
            for s, c in zip(diag_strip, diag_capy_strip)
        ]

    if capy_modal_data is None:
        if capy_modal_npz is None:
            from capytaine_workflow import resolve_capytaine_npz_path

            capy_modal_npz = resolve_capytaine_npz_path(config)
        capy_modal_data = load_modal_radiation_npz(capy_modal_npz)

    A_bem = asymptotic_added_mass_6x6(capy_modal_data)
    n_bem = min(n_modes, A_bem.shape[0])
    diag_bem = np.diag(A_bem)[:n_bem]

    out = {
        "mode_indices": mode_indices,
        "strip_modal_diag_kg": diag_strip.tolist(),
        "capy_strip_proj_modal_diag_kg": np.asarray(diag_capy_strip, dtype=float).tolist(),
        "capy_bem_modal_diag_kg": diag_bem.tolist(),
        "capy_strip_over_strip": capy_strip_over_strip,
        "capy_bem_over_strip": [
            float(b / s) if abs(s) > 1e-14 else float("nan")
            for s, b in zip(diag_strip[:n_bem], diag_bem)
        ],
        "capy_bem_over_capy_strip": [
            float(b / c) if abs(c) > 1e-14 else float("nan")
            for b, c in zip(diag_bem, diag_capy_strip[:n_bem])
        ],
    }
    return out


def run_rigid_capytaine_radiation(
    config,
    *,
    case_name: str | None = None,
    out_dir: str | None = None,
    depth: float | None = None,
    check: bool = True,
) -> str:
    """
    Run ``run_modal_radiation.py --rigid-body`` into ``capytaine_rigid_results_dir``.
    """
    case_name = case_name or str(getattr(config, "name", "case"))
    if out_dir is None:
        out_dir = os.path.dirname(default_rigid_radiation_npz_path(config, depth=depth))
    os.makedirs(out_dir, exist_ok=True)

    script = os.path.join(os.path.dirname(__file__), "run_modal_radiation.py")
    cmd = [
        sys.executable,
        script,
        "--case-name",
        case_name,
        "--rigid-body",
        "--out-dir",
        out_dir,
        "--no-plots",
    ]
    if depth is not None:
        cmd.extend(["--depth", str(depth)])

    print("[capytaine_strip_projection] Running:", " ".join(cmd))
    subprocess.run(cmd, check=check)
    return default_rigid_radiation_npz_path(config, depth=depth)


def ensure_rigid_capytaine_npz(config, *, run_if_missing: bool = True) -> str:
    path = default_rigid_radiation_npz_path(config)
    if os.path.isfile(path):
        return path
    if not run_if_missing:
        raise FileNotFoundError(path)
    run_rigid_capytaine_radiation(config)
    if not os.path.isfile(path):
        alt = os.path.join(os.path.dirname(path), f"modal_radiation_AB{_depth_npz_suffix(float(np.asarray(getattr(config, 'depth', 0.0)).ravel()[int(getattr(config, 'depth_index', 0))]))}.npz")
        if os.path.isfile(alt):
            return alt
        raise FileNotFoundError(f"Rigid Capytaine run finished but NPZ not found at {path}")
    return path
