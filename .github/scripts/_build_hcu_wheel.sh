#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

staging=$(mktemp -d "${CI_WORK_DIR}/wheel.XXXXXX")
shopt -s nullglob
python -m build --wheel --no-isolation --outdir "$staging/raw"
raw=("$staging"/raw/fastsafetensors-*.whl)
[[ ${#raw[@]} -eq 1 ]] || { echo 'Expected exactly one raw wheel' >&2; exit 1; }

bash .github/scripts/repair-manylinux.sh "${raw[0]}" "$staging/repaired"
repaired=("$staging"/repaired/fastsafetensors-*.whl)
[[ ${#repaired[@]} -eq 1 ]] || { echo 'Expected exactly one repaired wheel' >&2; exit 1; }

CIUpload REPAIR \
  --dtk_version "$DTK_VER" \
  --torch_version "$TORCH_VERSION" \
  --git_commit "$(git rev-parse --short=6 HEAD)" \
  --outputdir "$WHEEL_OUTPUT" \
  -f "${repaired[0]}"
wheel=("$WHEEL_OUTPUT"/fastsafetensors-*+dtk"$DTK_VER".torch"${TORCH_VERSION//./}".*.whl)
[[ ${#wheel[@]} -eq 1 ]] || { echo 'Expected exactly one DTK/Torch-tagged wheel' >&2; exit 1; }
printf 'Built wheel: %s\n' "${wheel[0]}"
