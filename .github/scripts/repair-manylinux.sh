#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
wheel=${1:?Specify the raw wheel}
output=${2:?Specify the repair output directory}
torch_lib=$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')
export LD_LIBRARY_PATH="${torch_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Keep the HCU runtime and Torch libraries supplied by the target environment.
auditwheel repair --plat manylinux_2_28_x86_64 --only-plat --strip \
  --exclude 'libgalaxyhip.so*' \
  --exclude 'libMIOpen.so*' \
  --exclude 'librccl.so*' \
  --exclude 'libhipblas.so*' \
  --exclude 'libhipfft.so*' \
  --exclude 'libhiprand.so*' \
  --exclude 'libhipsolver.so*' \
  --exclude 'libhipsparse.so*' \
  --exclude 'libhipnn.so*' \
  --exclude 'librocblas.so*' \
  --exclude 'librocsolver.so*' \
  --exclude 'librocfft.so*' \
  --exclude 'librocrand.so*' \
  --exclude 'librocsparse.so*' \
  --exclude 'librocm_smi64.so*' \
  --exclude 'librocfft-device-0.so*' \
  --exclude 'librocfft-device-1.so*' \
  --exclude 'librocfft-device-2.so*' \
  --exclude 'librocfft-device-3.so*' \
  --exclude 'libc10.so*' \
  --exclude 'libc10_hip.so*' \
  --exclude 'libtorch_cpu.so*' \
  --exclude 'libtorch_hip.so*' \
  --exclude 'libtorch_python.so*' \
  --exclude 'libtorch.so*' \
  --exclude 'libtorchaudio_ffmpeg4.so*' \
  --exclude 'libtorchaudio_ffmpeg5.so*' \
  --exclude 'libtorchaudio_ffmpeg6.so*' \
  --exclude 'libtorchaudio.so*' \
  --exclude 'libtorchaudio_sox.so*' \
  --exclude '_torchaudio_ffmpeg4.so*' \
  --exclude '_torchaudio_ffmpeg5.so*' \
  --exclude '_torchaudio_ffmpeg6.so*' \
  --exclude '_torchaudio.so*' \
  --exclude '_torchaudio_sox.so*' \
  --exclude 'libavcodec.so*' \
  --exclude 'libavdevice.so*' \
  --exclude 'libavfilter.so*' \
  --exclude 'libavformat.so*' \
  --exclude 'libavutil.so*' \
  --exclude 'libsox.so*' \
  --exclude 'libomp.so*' \
  --exclude 'libhipblaslt.so*' \
  -w "$output" "$wheel"
