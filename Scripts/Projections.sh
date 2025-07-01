#!/usr/bin/env bash
set -euo pipefail

# Vai para o dir onde o .sh está (Scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Define o root do projeto (uma pasta acima de Scripts/)
PROJECT_ROOT="$SCRIPT_DIR/.."

echo "▶ Rodando projeções COM pontos existentes…"
python3 "$PROJECT_ROOT/Projection/projection_with_points.py"

echo "▶ Rodando projeções SEM pontos existentes…"
python3 "$PROJECT_ROOT/Projection/projection_without_points.py"

echo "✓ Todos os scripts executados com sucesso!"