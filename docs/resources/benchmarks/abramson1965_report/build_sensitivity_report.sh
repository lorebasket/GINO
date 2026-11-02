#!/usr/bin/env bash
# Build sensitivity rerun LaTeX report from FSI repository root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
TEX="docs/resources/benchmarks/abramson1965_report/sensitivity_rerun_report.tex"
OUT="docs/resources/benchmarks/abramson1965_report/build"
PDF="$OUT/sensitivity_rerun_report.pdf"
LINK="docs/resources/benchmarks/abramson1965_report/sensitivity_rerun_report.pdf"

cd "$ROOT"
# Regenerate .tex from latest CSV / flutter_summary
/home/lorenzo/anaconda3/envs/GINO/bin/python \
  examples/ABRAMSON1965/write_sensitivity_rerun_report_tex.py
mkdir -p "$OUT"
pdflatex -synctex=1 -interaction=nonstopmode -output-directory="$OUT" "$TEX" >/dev/null
pdflatex -synctex=1 -interaction=nonstopmode -output-directory="$OUT" "$TEX" >/dev/null
cp -f "$PDF" "$LINK"
echo "Built: $ROOT/$PDF"
echo "Linked: $ROOT/$LINK"
