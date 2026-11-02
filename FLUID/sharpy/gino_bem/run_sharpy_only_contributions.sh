#!/usr/bin/env bash
# Stock SharPy UVLM (no Capytaine) — M/K/C contributions via p-k driver.
set -euo pipefail
source "${HOME}/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate sharpy

FSI="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
OUT="$FSI/output_data/ABRAMSON1965/07_sharpy_pk_bem/_baseline_sharpy_only"
LOG="$OUT/_run_contributions.log"
cd "$FSI/FLUID/sharpy"

echo "========== $(date -Iseconds) START sharpy-only contributions ==========" | tee "$LOG"
python gino_bem/run_abramson_pk_bem.py \
  --grid n12c16 \
  --added-mass-removal none \
  --no-bem \
  --v-step-kn 1 \
  --outdir "$OUT" \
  --tag _baseline_sharpy_only \
  2>&1 | tee -a "$LOG"
echo "========== $(date -Iseconds) DONE rc=${PIPESTATUS[0]} ==========" | tee -a "$LOG"
