#!/bin/bash
# Full chain for any case whose config defines paths(FSI) (run from anywhere):
#   dry modes -> mode content -> UVLM GAF -> BEM radiation -> A_UVLM_nc -> scans uvlm + delta
#
#   CASE=TFOIL_SIMPLE \
#   GAF_CMD='FLUID/sharpy/gino_bem/precompute_quvlm_tfoil_simple.py --gino {dry} --nm {nm} --out {gaf}' \
#   BEM_CMD='FLUID/capytaine/run_tfoil_simple_radiation.py --modes {dry} --nm {nm} --out {bem}' \
#   [case env vars, e.g. TFS_NM=10 TFS_XEA=0.35] bash utl/run_chain.sh
#
# {dry} {gaf} {bem} {nm} are replaced with the case paths().  GAF_CMD runs in the sharpy env,
# BEM_CMD in the GINO env.  SKIP_GAF=1 / SKIP_BEM=1 reuse existing tables — only if the dry
# frequencies did not change (always regenerate the GAF table otherwise).
# PK=1 also runs main.py (tracked p-k) after the scans.
set -e
cd "$(dirname "$0")/.."
: "${CASE:?set CASE}"
PY="conda run -n GINO python"; SP="conda run -n sharpy python"
# paths relative to the repo root: the absolute ones contain spaces on this disk
P=$($PY -c "import os, sys; sys.path.insert(0,'utl'); import _common as C; P=C.case_paths('$CASE')
r=lambda k: os.path.relpath(P[k]) if P.get(k) else '-'
print(r('base'), r('dry'), r('gaf'), r('bem'), P.get('nm') or 10)")
read -r BASE DRY GAF BEM NM <<< "$P"
mkdir -p "$BASE"
sub(){ local s=$1; s=${s//\{dry\}/$DRY}; s=${s//\{gaf\}/$GAF}; s=${s//\{bem\}/$BEM}; echo "${s//\{nm\}/$NM}"; }
step(){ echo "=== $(date +%H:%M) $CASE $(basename "$BASE"): $*"; }

step "dry modes";   $PY utl/export_modes.py "$CASE" | tail -n 1
$PY utl/classify_modes.py "$DRY" --nm "$NM" > "$BASE/modes.txt"
if [ -n "$GAF_CMD" ] && { [ -z "$SKIP_GAF" ] || [ ! -f "$GAF" ]; }; then
  step "UVLM GAF"; $SP $(sub "$GAF_CMD") > "$BASE/gaf.log" 2>&1
fi
if [ -n "$BEM_CMD" ] && { [ -z "$SKIP_BEM" ] || [ ! -f "$BEM" ]; }; then
  step "BEM";      $PY $(sub "$BEM_CMD") > "$BASE/bem.log" 2>&1
fi
step "A_UVLM_nc";   $PY utl/uvlm_nc_mass.py --case "$CASE"
step "scan UVLM";   $PY utl/scan.py "$CASE" uvlm  > "$BASE/scan_uvlm.log" 2>&1
[ -f "$BEM" ] && { step "scan UVLM+BEM (delta)"; $PY utl/scan.py "$CASE" delta > "$BASE/scan_delta.log" 2>&1; }
$PY utl/flutter_from_scan.py "$BASE"/scan_*.txt
if [ -n "$PK" ]; then
  step "p-k (main.py)"; $PY main.py "$CASE" > "$BASE/pk.log" 2>&1
fi
step END
