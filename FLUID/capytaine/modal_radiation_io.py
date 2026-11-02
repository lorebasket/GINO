from __future__ import annotations

import os
from typing import Any

import numpy as np


def save_modal_radiation_npz(
    *,
    path: str,
    omega: np.ndarray,
    mode_names: list[str],
    added_mass: np.ndarray,
    added_damping: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Save Capytaine modal radiation results in a stable workflow format.

    Stored arrays:
      - omega: (n_omega,)
      - mode_names: (n_modes,)
      - added_mass: (n_omega, n_modes, n_modes)
      - added_damping: (n_omega, n_modes, n_modes)

    Compatibility aliases are also written:
      - radiation_damping (same as added_damping)
    """
    omega = np.asarray(omega, dtype=float).ravel()
    A = np.asarray(added_mass, dtype=float)
    B = np.asarray(added_damping, dtype=float)

    if A.shape != B.shape:
        raise ValueError(f"added_mass and added_damping shape mismatch: {A.shape} vs {B.shape}")
    if A.ndim != 3:
        raise ValueError(f"Expected 3D matrices (n_omega, n_modes, n_modes), got {A.shape}")
    if A.shape[0] != omega.size:
        raise ValueError(f"omega size {omega.size} not consistent with matrix shape {A.shape}")
    if A.shape[1] != A.shape[2]:
        raise ValueError(f"Matrices must be square in mode-space, got {A.shape}")
    if len(mode_names) != A.shape[1]:
        raise ValueError(f"mode_names length {len(mode_names)} does not match matrix shape {A.shape}")

    payload: dict[str, Any] = {
        "omega": omega,
        "mode_names": np.asarray(mode_names, dtype=object),
        "added_mass": A,
        "added_damping": B,
        "radiation_damping": B,  # Backward-compatible alias.
    }
    if metadata:
        payload.update(metadata)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_modal_radiation_npz(path: str) -> dict[str, Any]:
    """
    Load modal radiation matrices saved by ``save_modal_radiation_npz``.
    """
    data = np.load(path, allow_pickle=True)

    if "omega" not in data or "added_mass" not in data:
        raise KeyError(f"{path} is missing mandatory keys 'omega' and/or 'added_mass'.")

    omega = np.asarray(data["omega"], dtype=float).ravel()
    A = np.asarray(data["added_mass"], dtype=float)
    if "added_damping" in data:
        B = np.asarray(data["added_damping"], dtype=float)
    elif "radiation_damping" in data:
        B = np.asarray(data["radiation_damping"], dtype=float)
    else:
        raise KeyError(f"{path} is missing 'added_damping' (or alias 'radiation_damping').")

    if A.shape != B.shape:
        raise ValueError(f"added_mass and added_damping shape mismatch: {A.shape} vs {B.shape}")
    if A.ndim != 3 or A.shape[0] != omega.size:
        raise ValueError(
            f"Inconsistent shapes in {path}: omega={omega.shape}, added_mass={A.shape}, added_damping={B.shape}"
        )

    mode_names = [str(x) for x in data["mode_names"].tolist()] if "mode_names" in data else [
        f"mode_{i + 1}" for i in range(A.shape[1])
    ]

    out: dict[str, Any] = {
        "omega": omega,
        "mode_names": mode_names,
        "added_mass": A,
        "added_damping": B,
    }
    for key in data.files:
        if key not in {"omega", "mode_names", "added_mass", "added_damping", "radiation_damping"}:
            out[key] = data[key]
    return out
