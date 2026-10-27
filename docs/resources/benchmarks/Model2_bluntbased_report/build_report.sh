#!/usr/bin/env bash
# Build Model2-bluntbased report PDF from the FSI repository root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
TEX="docs/resources/benchmarks/Model2_bluntbased_report/model2_bluntbased_current_report.tex"
OUT="docs/resources/benchmarks/Model2_bluntbased_report/build"
PDF="$OUT/model2_bluntbased_current_report.pdf"
LINK="docs/resources/benchmarks/Model2_bluntbased_report/model2_bluntbased_current_report.pdf"

cd "$ROOT"
mkdir -p "$OUT"
# Figures resolve relative to repo root or report folder (see \reportpath in .tex)
pdflatex -synctex=1 -interaction=nonstopmode -output-directory="$OUT" "$TEX" >/dev/null || true
pdflatex -synctex=1 -interaction=nonstopmode -output-directory="$OUT" "$TEX" >/dev/null || true
cp -f "$PDF" "$LINK"
echo "Built: $ROOT/$PDF"
echo "Linked: $ROOT/$LINK"
