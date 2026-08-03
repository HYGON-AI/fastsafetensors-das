# SPDX-License-Identifier: Apache-2.0

"""Utilities for platform detection and conditional test execution."""

import os

import pytest


def is_rocm_platform():
    """Detect if running on ROCm/AMD platform.

    Uses the C++ extension's is_hip_found() to avoid framework-specific calls.
    """
    try:
        from fastsafetensors import cpp as fstcpp

        return fstcpp.is_hip_found()
    except:
        return False


def is_cuda_platform():
    """Detect if running on CUDA/NVIDIA platform."""
    return not is_rocm_platform()


def is_gds_platform():
    """True if GDS is usable: cuFile on NVIDIA, or hyfs (libhyfile) on HCU."""
    try:
        from fastsafetensors import cpp as fstcpp

        return fstcpp.is_cufile_found()
    except:
        return False


_gds_probe = None


def gds_can_open(path):
    """True if GDS can actually register a file handle for `path`.

    is_cufile_found() and is_gds_supported() only report that the library loaded
    and the GPU is eligible. Both stay true on a host that never loaded the
    nvidia-fs/hyfs kernel module, or whose storage cannot do the DMA at all
    (LVM, SATA, ...), and the failure only surfaces later at
    cuFileHandleRegister. Registering a handle is the cheapest call that
    exercises library, driver and filesystem together.

    On CPU-only hosts the extension falls back to pread, which registers fine --
    so this stays true there and those tests keep running, as they do upstream.
    Cached: the probe costs a driver open plus a file open.
    """
    global _gds_probe
    if _gds_probe is None:
        try:
            from fastsafetensors import cpp as fstcpp
            from fastsafetensors.common import is_gpu_found

            fstcpp.init_gds()
            fh = fstcpp.gds_file_handle(path, True, is_gpu_found())
            del fh
            _gds_probe = True
        except Exception:
            _gds_probe = False
    return _gds_probe


def skip_if_no_gds(path):
    """Skip a GDS-only test when the host cannot actually do GDS.

    Without this the tests lie in both directions: a missing hyfs makes
    create_copier_constructor() downgrade to nogds and the test passes while
    covering nothing, and a missing nvidia-fs makes it fail on an environment
    issue. Set FASTSAFETENSORS_TEST_REQUIRE_GDS=1 on a CI that is supposed to
    have GDS -- then a host that lost it fails loudly instead of skipping.
    """
    if gds_can_open(path):
        return
    reason = f"GDS cannot open {path} (no kernel driver, or unsupported storage)"
    if os.getenv("FASTSAFETENSORS_TEST_REQUIRE_GDS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


_driver_probe = None


def gds_driver_can_open():
    """True if the GDS driver itself can be opened.

    Narrower than gds_can_open(): no file and no filesystem involved, just
    cuFileDriverOpen/hipFileDriverOpen. hyfs before 3.4.0.0 refuses the open
    outright on a host booted without `iommu=off` (see README), which is a
    property of the installed driver, not of the storage. Cached.
    """
    global _driver_probe
    if _driver_probe is None:
        try:
            from fastsafetensors import cpp as fstcpp

            _driver_probe = fstcpp.init_gds() == 0
        except Exception:
            _driver_probe = False
    return _driver_probe


def skip_if_no_gds_driver():
    """Skip a driver-level GDS test when the driver cannot be opened at all.

    Same contract as skip_if_no_gds(): a silent skip by default, and a loud
    failure under FASTSAFETENSORS_TEST_REQUIRE_GDS=1.
    """
    if gds_driver_can_open():
        return
    reason = "GDS driver cannot be opened (no kernel module, or hyfs refuses the open)"
    if os.getenv("FASTSAFETENSORS_TEST_REQUIRE_GDS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def is_hyfs_gds():
    """True only when GDS is backed by hyfs (HCU), not by NVIDIA cuFile.

    hyfs differs from cuFile in three ways that tests must branch on:
      * it registers buffers at 4KB page granularity and rejects overlapping
        regions, so one buffer cannot hold two separately registered sub-ranges;
      * it only serves reads into registered buffers -- cuFile falls back
        internally for unregistered ones;
      * it does not re-open the driver on demand after cuFileDriverClose.
    """
    return is_rocm_platform() and is_gds_platform()


# List of tests that are expected to fail on ROCm (based on TEST_RESULTS.md)
ROCM_EXPECTED_FAILURES = {
    "test_GdsFileCopier",  # GDS not available on AMD
}


def skip_if_rocm_expected_failure(test_name):
    """Skip test if it's an expected failure on ROCm.

    HCU is a HIP runtime but *does* provide GDS via hyfs, so only skip when
    the cuFile/hyfs library actually failed to load.
    """
    if not is_rocm_platform() or is_gds_platform():
        return
    if test_name in ROCM_EXPECTED_FAILURES:
        pytest.skip(
            f"Test '{test_name}' is expected to fail on ROCm (GDS not supported)"
        )


def get_platform_info():
    """Get platform information for debugging.

    Uses framework's get_cuda_ver() to avoid direct torch calls where possible.
    """
    info = {
        "is_rocm": is_rocm_platform(),
        "is_cuda": is_cuda_platform(),
    }

    try:
        from fastsafetensors import cpp as fstcpp
        from fastsafetensors.common import is_gpu_found

        if is_gpu_found():
            # Get version info from framework
            try:
                from fastsafetensors.frameworks import get_framework_op

                framework = get_framework_op("pytorch")
                gpu_ver = framework.get_cuda_ver()
                info["gpu_version"] = gpu_ver

                # Parse the version to get specific info
                if gpu_ver.startswith("hip-"):
                    info["hip_version"] = gpu_ver[4:]  # Remove 'hip-' prefix
                    info["rocm_version"] = gpu_ver[4:]
                elif gpu_ver.startswith("cuda-"):
                    info["cuda_version"] = gpu_ver[5:]  # Remove 'cuda-' prefix
            except:
                pass

            # Get device count and name (still needs torch for this)
            try:
                import torch

                if torch.cuda.is_available():
                    info["torch_version"] = torch.__version__
                    info["device_count"] = torch.cuda.device_count()
                    info["device_name"] = (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.device_count() > 0
                        else None
                    )
            except:
                pass
    except:
        pass

    return info
