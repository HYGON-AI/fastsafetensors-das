#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

shopt -s nullglob
wheels=("$WHEEL_OUTPUT"/fastsafetensors-*.whl)
[[ ${#wheels[@]} -eq 1 ]] || { echo 'Expected exactly one built wheel' >&2; exit 1; }
python -m pip install --constraint "$CI_WORK_DIR/constraints.txt" "${wheels[0]}[test]"

python - <<'PY'
import torch

count = torch.cuda.device_count()
if count != 4:
    raise SystemExit(f'Expected four HCU devices from the ARC pool, got {count}')
print(f'Visible HCU devices: {count}')
PY

reports="${CI_PROJECT_DIR}/reports"
mkdir -p "$reports"
test_dir=$(mktemp -d "${CI_WORK_DIR}/tests.XXXXXX")
cp -r tests pyproject.toml "$test_dir/"
cd "$test_dir"
unset PYTHONPATH
export TEST_FASTSAFETENSORS_FRAMEWORK=pytorch
suite=(tests/unit/ --ignore=tests/unit/threefs -q -rs)

python -m pytest "${suite[@]}" --junitxml="$reports/hcu.xml" 2>&1 | tee "$reports/hcu.log"
FASTSAFETENSORS_TEST_REQUIRE_GDS=1 python -m pytest "${suite[@]}" \
  --junitxml="$reports/gds.xml" 2>&1 | tee "$reports/gds.log"
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m pytest tests/unit/test_multi.py -q -rs 2>&1 | tee "$reports/multi-hcu.log"
CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
  python -m pytest "${suite[@]}" --junitxml="$reports/cpu.xml" 2>&1 | tee "$reports/cpu.log"
