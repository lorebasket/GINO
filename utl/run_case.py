#!/usr/bin/env python3
"""Run the GINO flutter pipeline (main.run_flutter_analysis_single) on any case with config overrides.

Replaces the per-case ``run_baseline_*``, ``run_*_sweep`` and ``run_method_matrix`` drivers:
each variant is just a set of AnalysisConfig attributes.

    # one run, output in its own folder
    python utl/run_case.py Model2t --out output_data/Model2t/uvlm_only \\
        --set aero_method='"UVLM"' --set aero_b2_removal=False --set V_list='np.linspace(2,20.58,240)'

    # overrides from a JSON file (dict of attribute -> value), env vars for env-driven configs
    python utl/run_case.py TFOIL_SIMPLE --json variant.json --env TFS_STAGE=bem

    # one-at-a-time sweep: one sub-folder per value, summary table at the end
    python utl/run_case.py GOLAND_water --out output_data/GOLAND_water/sweeps/gj \\
        --sweep gj_frac=0.8,1.0,1.2

Values are python literals; ``np.`` expressions are evaluated with numpy.  Each run folder gets
run.log, run_config.json and the usual vg_data / vf_data / flutter_summary files.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

TAGGED = ("vg_data", "vf_data", "vlambda_data", "flutter_summary", "vg_vf_combined",
          "stiffness_damping_contributions")


def _value(text):
    if "np." in text:
        import numpy as np
        return eval(text, {"np": np})                      # noqa: S307 (local CLI only)
    return C.parse_value(text)


def _pairs(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = _value(v)
    return out


def _untag(folder: Path, tag: str):
    for stem in TAGGED:
        for ext in (".csv", ".png"):
            t, c = folder / f"{stem}_{tag}{ext}", folder / f"{stem}{ext}"
            if t.is_file():
                if c.is_file():
                    c.unlink()
                shutil.move(str(t), str(c))


def run_one(case, out: Path, overrides: dict, alpha=0.0, force=False, quiet=True):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "flutter_summary.csv").is_file() and not force:
        print(f"skip (exists, use --force): {out}")
        return None
    C.setup_paths()
    import examples.config as cfg_mod
    from main import run_flutter_analysis_single

    orig = cfg_mod.get_config

    def patched(name):
        cfg = orig(name)
        cfg.output_dir = str(out)
        cfg.modal_dir = str(out)
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                print(f"  warning: AnalysisConfig has no attribute '{k}' (set anyway)")
            setattr(cfg, k, v)
        if "num_modes_flutter_egv" in overrides:
            n = cfg.num_modes_flutter_egv
            cfg.modes_to_analyze = overrides.get("modes_to_analyze", list(range(n)))
            cfg.modes_to_visualize = overrides.get("modes_to_visualize", list(range(n)))
        return cfg

    cfg_mod.get_config = patched
    tag = out.name
    t0 = time.time()
    try:
        with open(out / "run.log", "w", encoding="utf-8") as log:
            red = (contextlib.redirect_stdout(log), contextlib.redirect_stderr(log)) if quiet else ()
            with contextlib.ExitStack() as st:
                for r in red:
                    st.enter_context(r)
                res, _ = run_flutter_analysis_single(case, alpha, vg_vf_output_tag=tag)
    except Exception:
        with open(out / "run.log", "a", encoding="utf-8") as log:
            log.write(traceback.format_exc())
        raise
    finally:
        cfg_mod.get_config = orig
    _untag(out, tag)
    vf = getattr(res, "flutter_speed", None) if res else None
    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(dict(case=case, alpha_deg=alpha, elapsed_s=time.time() - t0,
                       overrides={k: repr(v) for k, v in overrides.items()},
                       env={k: os.environ[k] for k in RUN_ENV},
                       flutter_speed_m_s=vf), f, indent=2)
    print(f"{out}: V_f = {'%.2f kn' % (vf / C.KN) if vf else 'none'}  ({time.time() - t0:.0f} s)")
    return vf


RUN_ENV: list = []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("--out", type=Path, help="run folder (default output_data/<case>/run)")
    ap.add_argument("--set", action="append", metavar="ATTR=VALUE")
    ap.add_argument("--json", type=Path, help="JSON dict of overrides")
    ap.add_argument("--env", action="append", metavar="VAR=VALUE", help="env vars read by the case config")
    ap.add_argument("--sweep", metavar="ATTR=V1,V2,...", help="one sub-folder per value")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print the pipeline output instead of run.log")
    a = ap.parse_args()

    for e in a.env or []:
        k, _, v = e.partition("=")
        os.environ[k] = v
        RUN_ENV.append(k)
    ov = json.loads(a.json.read_text()) if a.json else {}
    ov.update(_pairs(a.set))
    out = a.out or (C.REPO / "output_data" / a.case / "run")

    if not a.sweep:
        run_one(a.case, out, ov, a.alpha, a.force, not a.verbose)
        return
    key, _, vals = a.sweep.partition("=")
    rows = []
    for txt in vals.split(","):
        v = _value(txt)
        sub = out / f"{key}_{txt.strip()}"
        vf = run_one(a.case, sub, {**ov, key: v}, a.alpha, a.force, not a.verbose)
        rows.append((txt.strip(), vf))
    with open(out / "sweep_summary.csv", "w") as f:
        f.write(f"{key},Flutter_Speed_m_s,Flutter_Speed_knots\n")
        for t, vf in rows:
            f.write(f"{t},{vf or ''},{vf / C.KN if vf else ''}\n")
    print(f"summary: {out / 'sweep_summary.csv'}")


if __name__ == "__main__":
    main()
