"""
Conditional MAC tracking for an emergent orphan wet root (Roger RFA / PK post-spectrum).

Free spectrum-index tracking runs first. After the orphan branch is present with a
continuous frequency trajectory for at least ``orphan_mac_min_span_m_s``, MAC tracking
locks onto the eigenvector at the segment start (first appearance of that continuous run).
"""

from __future__ import annotations

import numpy as np

from . import roger_mode_tracking


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=complex).ravel()
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def pick_orphan_root_free(
    wet_spectrum: dict,
    *,
    tracked_index: int | None,
    omega_prev: float | None,
) -> tuple[dict | None, int | None]:
    """
    Select orphan root from a classified wet spectrum (no MAC).

    Uses lowest_untracked when available, else continues ``tracked_index``, else
  frequency-closest untracked / structural_near root.
    """
    roots = list(wet_spectrum.get("roots") or [])
    if not roots:
        return None, tracked_index

    low = wet_spectrum.get("lowest_untracked")
    if low is not None:
        idx = int(low["index"])
        return low, idx

    if tracked_index is not None:
        for r in roots:
            if int(r["index"]) == int(tracked_index):
                return r, tracked_index

    candidates = [r for r in roots if r.get("kind") in ("untracked", "structural_near")]
    if not candidates:
        candidates = [r for r in roots if r.get("kind") != "pk_tracked"]
    if not candidates:
        return None, tracked_index

    if omega_prev is not None and omega_prev > 0:
        candidates.sort(
            key=lambda r: abs(float(r["omega"]) - omega_prev) / omega_prev
        )
    else:
        untracked = [r for r in candidates if r.get("kind") == "untracked"]
        candidates = untracked if untracked else candidates
        candidates = sorted(candidates, key=lambda r: float(r["omega"]))

    pick = candidates[0]
    return pick, int(pick["index"])


def _is_frequency_continuous(
    omega_prev: float,
    omega_new: float,
    max_rel_jump: float,
) -> bool:
    if omega_prev <= 0 or omega_new <= 0:
        return False
    return abs(omega_new - omega_prev) / omega_prev <= float(max_rel_jump)


def pick_orphan_root_mac(
    candidates: list[dict],
    ref_vec: np.ndarray,
    *,
    omega_prev: float | None,
    max_rel_jump: float,
    n_structural_dof: int | None,
    used_indices: set[int] | None = None,
) -> dict | None:
    """Pick orphan root by MAC to reference eigenvector (structural DOF block)."""
    if not candidates or ref_vec is None:
        return None

    ref_vec = _normalize_vec(ref_vec)
    used_indices = used_indices or set()

    best = None
    best_cost = np.inf
    for r in candidates:
        if int(r["index"]) in used_indices:
            continue
        vec = r.get("vec")
        if vec is None:
            continue
        om = float(r["omega"])
        if om <= 0:
            continue
        mac = roger_mode_tracking.compute_mac(
            vec, ref_vec, n_structural_dof=n_structural_dof
        )
        cost = 1.0 - mac
        if omega_prev is not None and omega_prev > 0:
            freq_err = abs(om - omega_prev) / omega_prev
            if freq_err > max_rel_jump:
                cost += 10.0 + freq_err
            else:
                cost += 0.15 * freq_err
        if cost < best_cost:
            best_cost = cost
            best = r
    return best


class OrphanBranchTracker:
    """
    State machine: off → free (spectrum track) → mac (after min span + continuity).

    ``ref_vec`` is taken at the first velocity of the current continuous segment
    (when MAC activates, and kept fixed thereafter).
    """

    def __init__(self, config):
        self.enabled = bool(getattr(config, "orphan_conditional_mac", False))
        self.min_span_m_s = float(getattr(config, "orphan_mac_min_span_m_s", 5.0))
        self.max_freq_jump = float(
            getattr(config, "orphan_mac_max_freq_jump_frac", 0.30)
        )
        self.n_structural_dof = (
            None
            if not getattr(config, "rfa_mac_structural_dof", True)
            else None  # set per step from n_modes_modal
        )

        self.phase = "off"  # off | free | mac
        self.V_segment_start: float | None = None
        self.ref_vec: np.ndarray | None = None
        self.ref_vec_segment: np.ndarray | None = None
        self.tracked_index: int | None = None
        self.omega_prev: float | None = None
        self.mac_activated = False
        self.V_mac_start: float | None = None

    def step(
        self,
        V: float,
        wet_spectrum: dict,
        *,
        n_structural_dof: int | None,
        tracked_branch_indices: set[int] | None = None,
    ) -> dict:
        """
        Advance orphan tracking one velocity step.

        Returns dict with keys: g, omega, phase, mac_active, spectrum_index, kind.
        Values are NaN / off when no orphan is selected.
        """
        empty = {
            "g": np.nan,
            "omega": np.nan,
            "phase": self.phase,
            "mac_active": self.mac_activated,
            "spectrum_index": -1,
            "kind": "",
            "V_segment_start": self.V_segment_start,
            "V_mac_start": self.V_mac_start,
        }
        if not self.enabled:
            return empty

        tracked_branch_indices = tracked_branch_indices or set()

        if self.phase == "mac" and self.ref_vec is not None:
            candidates = [
                r
                for r in wet_spectrum.get("roots", [])
                if int(r["index"]) not in tracked_branch_indices
            ]
            pick = pick_orphan_root_mac(
                candidates,
                self.ref_vec,
                omega_prev=self.omega_prev,
                max_rel_jump=self.max_freq_jump,
                n_structural_dof=n_structural_dof,
            )
            if pick is None:
                return empty
            self.omega_prev = float(pick["omega"])
            self.tracked_index = int(pick["index"])
            return {
                "g": float(pick["gamma"]),
                "omega": float(pick["omega"]),
                "phase": "mac",
                "mac_active": True,
                "spectrum_index": int(pick["index"]),
                "kind": str(pick.get("kind", "")),
                "V_segment_start": self.V_segment_start,
                "V_mac_start": self.V_mac_start,
            }

        # free or off: spectrum-index tracking
        pick, idx = pick_orphan_root_free(
            wet_spectrum,
            tracked_index=self.tracked_index,
            omega_prev=self.omega_prev,
        )
        if pick is None:
            self._reset_segment()
            return empty

        om = float(pick["omega"])
        if self.omega_prev is not None and self.V_segment_start is not None:
            continuous = _is_frequency_continuous(
                self.omega_prev, om, self.max_freq_jump
            )
        else:
            continuous = True

        if self.V_segment_start is None:
            self.V_segment_start = float(V)
            self.tracked_index = idx
            self.phase = "free"
            self.ref_vec_segment = (
                _normalize_vec(pick["vec"]) if pick.get("vec") is not None else None
            )
        elif not continuous:
            self.V_segment_start = float(V)
            self.tracked_index = idx
            self.phase = "free"
            self.ref_vec_segment = (
                _normalize_vec(pick["vec"]) if pick.get("vec") is not None else None
            )
            self.mac_activated = False
            self.V_mac_start = None
            self.ref_vec = None

        span_ok = (
            self.V_segment_start is not None
            and float(V) - float(self.V_segment_start) >= self.min_span_m_s
        )

        if (
            span_ok
            and continuous
            and self.ref_vec_segment is not None
            and not self.mac_activated
        ):
            self.phase = "mac"
            self.mac_activated = True
            self.V_mac_start = float(V)
            self.ref_vec = self.ref_vec_segment
            print(
                f"  [orphan MAC] Activated at V={V:.3g} m/s "
                f"(segment since V={self.V_segment_start:.3g} m/s, "
                f"ref eigenvector from segment start, ω={om:.2f} rad/s, index={idx})"
            )
            mac_pick = pick_orphan_root_mac(
                [
                    r
                    for r in wet_spectrum.get("roots", [])
                    if int(r["index"]) not in tracked_branch_indices
                ],
                self.ref_vec,
                omega_prev=self.omega_prev,
                max_rel_jump=self.max_freq_jump,
                n_structural_dof=n_structural_dof,
            )
            if mac_pick is not None:
                pick = mac_pick
                om = float(pick["omega"])
                idx = int(pick["index"])

        self.omega_prev = om
        self.tracked_index = idx

        return {
            "g": float(pick["gamma"]),
            "omega": om,
            "phase": self.phase,
            "mac_active": self.mac_activated,
            "spectrum_index": int(pick["index"]),
            "kind": str(pick.get("kind", "")),
            "V_segment_start": self.V_segment_start,
            "V_mac_start": self.V_mac_start,
        }

    def _reset_segment(self):
        if self.phase == "mac":
            return
        self.phase = "off"
        self.V_segment_start = None
        self.ref_vec_segment = None
        self.tracked_index = None
        self.omega_prev = None
