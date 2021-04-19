#!/bin/bash
set -euxo pipefail

conda create --yes -n "${CONDA_ENV}" python="${PYTHON_VERSION}"
source activate "${CONDA_ENV}"
