---
name: build-hcu-wheel
description: Build or diagnose the fastsafetensors HCU manylinux wheel with the repository's DTK and Torch build scripts, locally on Linux or in ARC CI.
---

Use `.github/scripts/_build_hcu_wheel.sh` from the repository root. The script builds the C++17 extension, runs `auditwheel repair` for `manylinux_2_28_x86_64`, and adds the DTK, Torch and commit information with `CIUpload REPAIR`. A Linux x86_64 build environment needs `/opt/dtk/env.sh`, a C++17 compiler, the matching Python and HCU Torch, and the build and repair tools. The ARC build job uses `hygon-build`.

Read the supported combinations in `.github/workflows/ci.yml`. For a local build, activate the matching Linux DTK image and set the same environment variables as the reusable workflow. For example, to build DTK 26.10 with Python 3.12 and Torch 2.11.0:

```bash
export GITHUB_WORKSPACE="$PWD"
export RUNNER_TEMP="$(mktemp -d)"
export GITHUB_RUN_ID=local GITHUB_RUN_ATTEMPT=1 GITHUB_JOB=build
export DTK_VER=2610 PYTHON_VERSION=3.12 TORCH_VERSION=2.11.0 NUMPY_VERSION=1.26.4
export PYPI_URL=http://12.12.12.201:9929
export TRITON_PYPI_URL="$PYPI_URL"

bash .github/scripts/setup-env.sh --build
bash .github/scripts/_build_hcu_wheel.sh
```

`setup-env.sh --build` installs the selected Torch, NumPy and packaging tools, then validates Python and the Torch DTK tag. When those dependencies are already prepared, run the build script directly with the same environment variables. Use a fresh output directory for the selected combination so that the script can verify it produced exactly one wheel. The output is `output/dtk${DTK_VER}/py${PYTHON_VERSION}/torch${TORCH_VERSION}/`, and the build prints the final wheel path.

For build failures, check that the selected image contains the expected DTK and Python, that the Torch package matches the requested DTK tag, and that the compiler and repair tools are available. The ARC test job installs the wheel artifact produced by this entry point. The publication workflow consumes the same build artifact.
