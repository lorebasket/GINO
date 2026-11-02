#!/usr/bin/env bash
# Vf(injected A_capy mass) trend on intact UVLM (control-B config, const-A):
# 25% / 50% / diag-only variants of the Capytaine added mass.
source "${HOME}/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate sharpy
set -eo pipefail
FSI="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
BASE="$FSI/output_data/ABRAMSON1965/07_sharpy_pk_bem"
SCRATCH="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI/FLUID/capytaine/ABRAMSON1965/results_modal_radiation/doctored_masstrend"
LOG="$BASE/_run_mass_trend.log"
cd "$FSI/FLUID/sharpy"
run_v() {
  local name="$1"
  echo "========== $(date -Iseconds) START $name ==========" | tee -a "$LOG"
  python gino_bem/run_abramson_pk_bem.py \
    --grid n12c16 --v-min-kn 8 --v-max-kn 100 --v-step-kn 2 \
    --const-a-omega 47 --bem-mode replace --added-mass-removal none \
    --capy-npz "$SCRATCH/capy_${name}.npz" \
    --outdir "$BASE/_masstrend_${name}" --tag "_masstrend_${name}" \
    2>&1 | tee -a "$LOG"
  echo "========== $(date -Iseconds) DONE $name rc=${PIPESTATUS[0]} ==========" | tee -a "$LOG"
}
run_v scale025
run_v scale050
run_v diagonly
echo "========== $(date -Iseconds) MASS TREND DONE ==========" | tee -a "$LOG"
