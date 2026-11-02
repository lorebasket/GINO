#!/usr/bin/env bash
# Build Abramson1965 report PDF from the FSI repository root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
TEX="docs/resources/benchmarks/abramson1965_report/abramson1965_current_report.tex"
OUT="docs/resources/benchmarks/abramson1965_report/build"
PDF="$OUT/abramson1965_current_report.pdf"
LINK="docs/resources/benchmarks/abramson1965_report/abramson1965_current_report.pdf"

cd "$ROOT"
mkdir -p "$OUT"
pdflatex -synctex=1 -interaction=nonstopmode -output-directory="$OUT" "$TEX" >/dev/null
pdflatex -synctex=1 -interaction=nonstopmode -output-directory="$OUT" "$TEX" >/dev/null
cp -f "$PDF" "$LINK"
echo "Built: $ROOT/$PDF"
echo "Linked: $ROOT/$LINK"
