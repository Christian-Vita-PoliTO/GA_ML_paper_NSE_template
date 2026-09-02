#!/usr/bin/env bash
# Compila il manoscritto (pdflatex -> bibtex -> pdflatex x2)
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error paper.tex > build.log 2>&1 || {
    echo "ERRORE in pdflatex — ultime righe:"; tail -25 build.log; exit 1; }
bibtex paper >> build.log 2>&1 || true
pdflatex -interaction=nonstopmode paper.tex >> build.log 2>&1 || true
pdflatex -interaction=nonstopmode paper.tex >> build.log 2>&1 || true
echo "pagine: $(pdfinfo paper.pdf 2>/dev/null | awk '/^Pages/{print $2}')"
echo "warning irrisolti:"; grep -c "Warning" build.log || true
grep -o "Citation \`[^']*' undefined" build.log | sort -u | head
