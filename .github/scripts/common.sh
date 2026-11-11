#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Source this file in every step: exports do not persist between Actions steps.
set -eo pipefail
source /opt/dtk/env.sh
set -u
cd "${GITHUB_WORKSPACE:?}"
export CI_PROJECT_DIR="${GITHUB_WORKSPACE}"
export CI_JOB_ID="${GITHUB_RUN_ID:?}_${GITHUB_RUN_ATTEMPT:?}_${GITHUB_JOB:?}_${DTK_VER:?}_${PYTHON_VERSION:?}_${TORCH_VERSION:?}"
export CI_WORK_DIR="${RUNNER_TEMP:?}/fastsafetensors-${CI_JOB_ID}"
export WHEEL_OUTPUT="${CI_PROJECT_DIR}/output/dtk${DTK_VER}/py${PYTHON_VERSION}/torch${TORCH_VERSION}"
export MAX_JOBS="${MAX_JOBS:-4}"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=10 PIP_RESUME_RETRIES=10
: "${PYPI_URL:?Set the PYPI_URL repository variable}"
: "${TRITON_PYPI_URL:?Set the TRITON_PYPI_URL repository variable}"
export PIP_INDEX_URL="${PYPI_URL%/}/nightly/dtk2604/+simple/"
# pip requires a hostname, without scheme, port or path.
export PIP_TRUSTED_HOST
PIP_TRUSTED_HOST=$(python -c 'import os,urllib.parse; print(" ".join(sorted({urllib.parse.urlparse(os.environ[k]).hostname for k in ["PYPI_URL","TRITON_PYPI_URL"]})))')
export TORCH_INDEX_URL="${PYPI_URL%/}/nightly/dtk${DTK_VER}/+simple/"
case "${DTK_VER}" in
  2604) ;;
  2610) export TORCH_INDEX_URL="${TRITON_PYPI_URL%/}/das_nightly/dtk2610/+simple/" ;;
  *) echo "Unsupported DTK ${DTK_VER}" >&2; exit 1 ;;
esac
mkdir -p "${CI_WORK_DIR}" "${WHEEL_OUTPUT}"
