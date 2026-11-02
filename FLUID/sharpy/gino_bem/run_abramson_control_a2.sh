#!/usr/bin/env bash
# Corrected Control A: TRUE stock UVLM (removal none, no BEM) via const-A machinery.
source "${HOME}/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate sharpy
set -eo pipefail
FSI="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
BASE="$FSI/output_data/ABRAMSON1965/07_sharpy_pk_bem"
LOG="$BASE/_run_controls.log"
cd "$FSI/FLUID/sharpy"
# wait for any running sweep (control B) to finish
while pgrep -f "run_abramson_pk_bem.py" > /dev/null; do sleep 30; done
name=_control_A2_stock_uvlm_none_nobem_constA
echo "========== $(date -Iseconds) START $name ==========" | tee -a "$LOG"
python gino_bem/run_abramson_pk_bem.py \
  --grid n12c16 --v-min-kn 8 --v-max-kn 100 --v-step-kn 2 \
  --const-a-omega 47 --no-bem --added-mass-removal none \
  --outdir "$BASE/$name" --tag "$name" \
  2>&1 | tee -a "$LOG"
echo "========== $(date -Iseconds) DONE $name rc=${PIPESTATUS[0]} ==========" | tee -a "$LOG"
