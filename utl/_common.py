"""Shared helpers for the utl/ scripts: repo paths, case config loading, case file paths.

Every utl script works on any case in ``examples/<CASE>/config_<CASE>.py``.  File paths
(dry modes, UVLM GAF table, BEM radiation, output folder) come from the case module's
optional ``paths(FSI)`` function when it exists, otherwise from the command line.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KN = 0.514444          # m/s per knot
SHARPY_ROOT = REPO / "FLUID" / "sharpy"


def setup_paths():
    """Same sys.path layout used by main.py (FEA, panelaero_utl, aerodynamic_model, ...)."""
    for p in (REPO / "FLUID" / "capytaine", REPO / "FLUID", REPO / "FLUID" / "PanelAero",
              REPO / "STRUCTURE" / "FEA" / "fea_utl", REPO / "STRUCTURE", REPO):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.environ.setdefault("MPLBACKEND", "Agg")


def case_module(case: str):
    """Import examples/<case>/config_<case>.py as a plain module (for paths(), geometry, ...)."""
    f = REPO / "examples" / case / f"config_{case}.py"
    if not f.is_file():
        raise SystemExit(f"case '{case}' not found: {f}")
    spec = importlib.util.spec_from_file_location(f"examples.{case}.config_{case}", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_config(case: str):
    """AnalysisConfig of the case, exactly as main.py builds it."""
    setup_paths()
    from examples import config
    return config.get_config(case)


def case_paths(case: str, **override) -> dict:
    """Case file paths: ``paths(FSI)`` from the case config if defined, then CLI overrides.

    Keys used by the scripts: base (output folder), dry, gaf, bem, anc, nm, tag.
    """
    mod = case_module(case) if case else None
    P = dict(mod.paths(str(REPO))) if mod is not None and hasattr(mod, "paths") else {}
    P.setdefault("base", str(REPO / "output_data" / (case or "unnamed")))
    P.setdefault("tag", case)
    for k, v in override.items():
        if v is not None:
            P[k] = v
    P.setdefault("dry", os.path.join(P["base"], "dry_modes.npz"))
    P.setdefault("anc", os.path.join(P["base"], "A_uvlm_nc.npz"))
    return P


def parse_value(text: str):
    """CLI literal -> python value (int, float, bool, None, list, str)."""
    import ast
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text
