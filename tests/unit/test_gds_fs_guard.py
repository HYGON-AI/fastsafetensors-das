# SPDX-License-Identifier: Apache-2.0
"""hyfs NFS guard: files on NFS must fall back to nogds on the hyfs platform.

hyfs (HCU GDS) reports support=true for NFS but cannot DMA over it. The read
then takes a CPU-staging shadow-buffer path that returns all-zero tensors with
every API call reporting success, or panics the kernel when the device pointer
is not 4K aligned.
"""

import types

import pytest

from fastsafetensors.copier import gds as gds_mod
from fastsafetensors.copier.gds import GdsFileCopier
from fastsafetensors.copier.nogds import NoGdsFileCopier
from fastsafetensors.st_types import Device

CPU = Device.from_str("cpu")
EXT4_SUPER_MAGIC = 0xEF53


@pytest.fixture
def hyfs_on_nfs(monkeypatch, framework, tmp_dir):
    """Fake a hyfs platform with the model file on NFS; yields (ctor, meta)."""
    # Deployment hosts may export this globally (the guard's own warning tells
    # them to); left set, the fallback assertions below would fail.
    monkeypatch.delenv("FASTSAFETENSORS_SKIP_FS_CHECK", raising=False)
    monkeypatch.setattr(gds_mod, "_fs_magic", lambda path: gds_mod.NFS_SUPER_MAGIC)
    monkeypatch.setattr(framework, "get_cuda_ver", lambda: "hip-5.7.0")
    monkeypatch.setattr(gds_mod.fstcpp, "is_cufile_found", lambda: True)
    # Do not touch the real driver: on a live hyfs host cuFileDriverOpen can
    # fail (e.g. 5011) and this guard test is about copier routing only.
    monkeypatch.setattr(gds_mod, "init_gds", lambda: None)
    src = f"{tmp_dir}/fs_guard_dummy.bin"
    with open(src, "wb") as f:
        f.write(b"\0" * 16)
    ctor = gds_mod.new_gds_file_copier(CPU)
    return ctor, types.SimpleNamespace(src=src)


def test_nfs_on_hyfs_falls_back_to_nogds(hyfs_on_nfs, framework):
    ctor, meta = hyfs_on_nfs
    with pytest.warns(UserWarning, match="NFS"):
        copier = ctor(meta, CPU, framework)
    assert isinstance(copier, NoGdsFileCopier)


def test_skip_fs_check_env_forces_gds(hyfs_on_nfs, framework, monkeypatch):
    ctor, meta = hyfs_on_nfs
    monkeypatch.setenv("FASTSAFETENSORS_SKIP_FS_CHECK", "1")
    copier = ctor(meta, CPU, framework)
    assert isinstance(copier, GdsFileCopier)


def test_nfs_on_cuda_platform_keeps_gds(hyfs_on_nfs, framework, monkeypatch):
    # cuFile handles NFS itself (compat mode / NFSoRDMA); do not intercept.
    ctor, meta = hyfs_on_nfs
    monkeypatch.setattr(framework, "get_cuda_ver", lambda: "cuda-12.4")
    copier = ctor(meta, CPU, framework)
    assert isinstance(copier, GdsFileCopier)


def test_local_fs_on_hyfs_keeps_gds(hyfs_on_nfs, framework, monkeypatch):
    ctor, meta = hyfs_on_nfs
    monkeypatch.setattr(gds_mod, "_fs_magic", lambda path: EXT4_SUPER_MAGIC)
    copier = ctor(meta, CPU, framework)
    assert isinstance(copier, GdsFileCopier)


def test_fs_magic_is_total(tmp_dir):
    # Never raises: real path, bogus path, and a non-UTF-8 name such as
    # os.listdir() yields for a GBK-named file all have to return an int.
    assert isinstance(gds_mod._fs_magic(tmp_dir), int)
    assert gds_mod._fs_magic("/nonexistent/fs/guard/path") == 0
    assert gds_mod._fs_magic(f"{tmp_dir}/bad\udcffname") == 0
