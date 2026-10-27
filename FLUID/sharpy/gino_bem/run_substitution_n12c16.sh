#!/usr/bin/env bash
# Abramson added-mass substitution matrix — n12c16 production grid.
set -euo pipefail
source "${HOME}/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate sharpy

FSI="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
BASE="$FSI/output_data/ABRAMSON1965/07_sharpy_pk_bem"
LOG="$BASE/_run_substitution_n12c16.log"
cd "$FSI/FLUID/sharpy"

run_case() {
  local removal="$1" folder="$2"
  local out="$BASE/$folder"
  echo "========== $(date -Iseconds) START $removal -> $out ==========" | tee -a "$LOG"
  python gino_bem/run_abramson_pk_bem.py \
    --grid n12c16 \
    --added-mass-removal "$removal" \
    --v-step-kn 1 \
    --outdir "$out" \
    --tag "$folder" \
    2>&1 | tee -a "$LOG"
  echo "========== $(date -Iseconds) DONE $removal rc=${PIPESTATUS[0]} ==========" | tee -a "$LOG"
}

run_case none  _baseline_added_mass_substitution_none
run_case gamma_dot _baseline_added_mass_substitution_gammadot
run_case full   _baseline_added_mass_substitution_full

echo "========== $(date -Iseconds) ALL DONE ==========" | tee -a "$LOG"
