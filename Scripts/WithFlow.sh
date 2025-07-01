#!/usr/bin/env bash
set -euo pipefail

# parte 1: identifica raiz do projeto (onde ficam Scripts, WithoutFlow, WithFlow)
ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"

# Experimentos e nomes de scripts
EXPS=(FourthExperiment)
SCRIPTS=(03_mip_solver.py)

echo "Executando todos os experimentos em WithFlow..."

for exp in "${EXPS[@]}"; do
    EXP_DIR="${ROOT_DIR}/WithFlow/${exp}"
    echo "→ Experiment: ${exp}"
    for script in "${SCRIPTS[@]}"; do
        SCRIPT_PATH="${EXP_DIR}/${script}"
        if [[ -f "${SCRIPT_PATH}" ]]; then
            echo "   Running ${SCRIPT_PATH}..."
            python "${SCRIPT_PATH}"
        else
            echo "   ⚠️  Não encontrou ${SCRIPT_PATH}, pulando."
        fi
    done
done

echo "✔ Todos os scripts em WithFlow foram executados."