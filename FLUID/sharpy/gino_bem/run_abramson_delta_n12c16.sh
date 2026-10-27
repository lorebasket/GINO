#!/usr/bin/env bash
# Abramson SHARPy UVLM + Capytaine BEM, DELTA injection — n12c16 production grid.
# const-A eigensweep at two omega* (wet 1B ~47 rad/s, wet 1T ~107 -> clips to
# grid edge 60): matching V_f between them proves A(omega) flatness (Model2t
# validation pattern), each sweep is one eigensolve per velocity (no p-k).
source "${HOME}/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate sharpy
set -eo pipefail

FSI="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
BASE="$FSI/output_data/ABRAMSON1965/07_sharpy_pk_bem"
LOG="$BASE/_run_delta_n12c16.log"
cd "$FSI/FLUID/sharpy"

run_const_a() {
  local w="$1"
  local out="$BASE/_baseline_sharpy_deltainjection_plusbem_constA_w${w}"
  echo "========== $(date -Iseconds) START delta constA w=$w -> $out ==========" | tee -a "$LOG"
  python gino_bem/run_abramson_pk_bem.py \
    --grid n12c16 \
    --bem-mode delta \
    --const-a-omega "$w" \
    --v-step-kn 1 \
    --outdir "$out" \
    --tag "_baseline_sharpy_deltainjection_plusbem_constA_w${w}" \
    2>&1 | tee -a "$LOG"
  echo "========== $(date -Iseconds) DONE delta constA w=$w rc=${PIPESTATUS[0]} ==========" | tee -a "$LOG"
}

run_const_a 47
run_const_a 107
echo "========== $(date -Iseconds) ALL DONE ==========" | tee -a "$LOG"
