#!/usr/bin/env bash
# Crea il repository su GitHub e ci spinge il manoscritto.
# Da lanciare UNA VOLTA, dopo essersi autenticati:  gh auth login
#
#   ./setup_github.sh [nome-repo]        default: lfr-energy-grid-paper
set -euo pipefail
cd "$(dirname "$0")"
REPO="${1:-lfr-energy-grid-paper}"

command -v gh >/dev/null || { echo "gh non trovato. Apri un terminale nuovo,"
                              echo "oppure: export PATH=\"\$HOME/.local/bin:\$PATH\""; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "non sei autenticato: lancia  gh auth login"; exit 1; }

USER=$(gh api user --jq .login)
echo "account GitHub: $USER"
echo "autore dei commit: $(git log -1 --format='%an <%ae>')"
echo

if git remote | grep -qx origin; then
    echo "remote 'origin' già presente: $(git remote get-url origin)"
else
    gh repo create "$REPO" --private --source=. --remote=origin
    echo "creato $USER/$REPO (privato)"
fi

git push -u origin "$(git branch --show-current)"
echo
echo "fatto. Ora su Overleaf:"
echo "  New Project -> Import from GitHub -> $USER/$REPO"
echo "  (serve un piano Overleaf a pagamento: la sincronizzazione GitHub"
echo "   non è disponibile sul piano gratuito)"
