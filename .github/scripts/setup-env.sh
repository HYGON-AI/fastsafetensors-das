#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

printf 'torch==%s\nnumpy==%s\n' "$TORCH_VERSION" "$NUMPY_VERSION" > "$CI_WORK_DIR/constraints.txt"
python -m pip install --force-reinstall --no-deps \
  --index-url "$TORCH_INDEX_URL" "torch==${TORCH_VERSION}"
python -m pip install --constraint "$CI_WORK_DIR/constraints.txt" \
  "torch==${TORCH_VERSION}" "numpy==${NUMPY_VERSION}"

python - <<'PY'
import os
import sys
from importlib.metadata import version

import torch

python_version = f'{sys.version_info.major}.{sys.version_info.minor}'
if python_version != os.environ['PYTHON_VERSION']:
    raise SystemExit(f'Unexpected Python version: {python_version}')
torch_version = version('torch')
release, _, local_version = torch_version.partition('+')
dtk_tag = f'dtk{os.environ["DTK_VER"]}'
if release != os.environ['TORCH_VERSION'] or dtk_tag not in local_version.split('.') or not torch.version.hip:
    raise SystemExit(f'Unexpected HCU Torch package: {torch_version}; expected {dtk_tag}')
print(f'Python {python_version}; Torch {torch_version}; HIP {torch.version.hip}')
PY

if [[ ${1:-} == --build ]]; then
  python -m pip install build 'setuptools>=78.1.1' wheel pybind11 \
    ciupload 'auditwheel>=6.6,<7' 'patchelf>=0.14.5'
fi
