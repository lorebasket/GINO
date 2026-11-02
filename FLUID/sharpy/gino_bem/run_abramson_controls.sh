#!/usr/bin/env bash
# Bug-localization controls for the Abramson delta no-flutter result.
# A: stock UVLM (no BEM) through the const-A eigensweep machinery — must
#    reproduce the known ~58.8 kn framework-comparison flutter.
# B: full A_capy,B_capy on intact UVLM (replace/none = June none_plusbem,
#    whose tracked CSV was empty) — isolates the delta A_offset's role.
source "${HOME}/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate sharpy
set -eo pipefail

FSI="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
BASE="$FSI/output_data/ABRAMSON1965/07_sharpy_pk_bem"
LOG="$BASE/_run_controls.log"
cd "$FSI/FLUID/sharpy"

run_ctl() {
  local name="$1"; shift
  local out="$BASE/$name"
  echo "========== $(date -Iseconds) START $name ==========" | tee -a "$LOG"
  python gino_bem/run_abramson_pk_bem.py \
    --grid n12c16 --v-min-kn 8 --v-max-kn 100 --v-step-kn 2 \
    --const-a-omega 47 \
    --outdir "$out" --tag "$name" "$@" \
    2>&1 | tee -a "$LOG"
  echo "========== $(date -Iseconds) DONE $name rc=${PIPESTATUS[0]} ==========" | tee -a "$LOG"
}

run_ctl _control_A_stock_uvlm_nobem_constA --no-bem
run_ctl _control_B_fullAcapy_intactUVLM_constA --bem-mode replace --added-mass-removal none
echo "========== $(date -Iseconds) CONTROLS DONE ==========" | tee -a "$LOG"
