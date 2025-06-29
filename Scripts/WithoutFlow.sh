#!/usr/bin/env bash
set -euo pipefail

# parte 1: identifica raiz do projeto (onde ficam Scripts, WithoutFlow, WithFlow)
ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"

# Experimentos e nomes de scripts
EXPS=(FirstExperiment SecondExperiment ThirdExperiment FourthExperiment)
SCRIPTS=(01_mip_solver.py 02_mip_solver.py 03_mip_solver.py)

echo "Executando todos os experimentos em WithoutFlow..."

for exp in "${EXPS[@]}"; do
    EXP_DIR="${ROOT_DIR}/WithoutFlow/${exp}"
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

echo "✔ Todos os scripts em WithoutFlow foram executados."