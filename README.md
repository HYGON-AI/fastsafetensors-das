fastsafetensors
================

fastsafetensors is an efficient safetensors loader. If you develop your own code that loads large safetensors files, you can try fastsafetensors APIs (see [docs](./docs/overview.md)). For example, vLLM and SGLang have `--load-format fastsafetensors` command-line argument to speed up their initialization.

This library supports Linux/CUDA, ROCm without GDS, Windows, [3FS](https://github.com/deepseek-ai/3fs), unified-memory systems such as DGX Spark, and so on. We welcome more platform/storage-specific optimizations like them by adding new [copier backends](fastsafetensors/copier/). Our CI tests Python 3.10-3.14 with PyTorch 2.11.0.

# Performance Highlights

Performance highlights from the [CLOUD 2025 paper](https://arxiv.org/abs/2505.23072) and benchmark docs:
- Standalone model loading was **4.8x-7.5x faster** than the default `safetensors` deserializer on Llama, Falcon, and Bloom models, and reached **26.4 GB/s** NVMe read throughput for Llama-70B on four GPUs with GDS.
- In the paper's vLLM integration experiment, startup time dropped from **12.39s to 4.74s** for Llama-2-13B on 4x L40S GPUs, and from **16.04s to 6.88s** on 1x A100.
- On AMD ROCm without GDS, the documented `nogds` path reached **6.02 GB/s** for GPT-2 Medium versus **1.28 GB/s** with `mmap` (**4.7x** throughput), and **2.62 GB/s** for GPT-2 versus **1.01 GB/s** with `mmap` (**2.6x** throughput). See the [report](./docs/amd-perf.md) for more details.

# Quick Start

```bash
pip install fastsafetensors
pip install vllm # for quick demo
vllm serve Qwen/Qwen3-0.6B --load-format fastsafetensors
...
Loading safetensors using Fastsafetensor loader:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors using Fastsafetensor loader: 100% Completed | 1/1 [00:00<00:00,  1.23it/s]
```

# Build from Source

To build the wheel directly from this checkout:

```bash
python3 setup.py bdist_wheel
python3 -m pip install dist/fastsafetensors-*.whl
```

In an offline container (no PyPI access), install in place instead and skip
build isolation, so the build runs against the packages already present:

```bash
python3 -m pip install -e . --no-build-isolation
```

The build uses standalone `pybind11` headers when available. If `pybind11` is
not installed, it falls back to the pybind11 headers bundled with `torch`.
That fallback only applies with `--no-build-isolation`: a normal PEP 517 build
installs `pybind11` into its isolated environment first (and `torch` is not
visible there anyway).

Only the pybind11 headers are needed at build time. The extension links nothing
but `libstdc++` -- CUDA, HIP and cuFile/hyfs symbols are all resolved at runtime
via `dlopen`, so no CUDA or ROCm SDK is required to build, and one binary serves
both NVIDIA and HCU.

# Running the Unit Tests

`TEST_FASTSAFETENSORS_FRAMEWORK` is mandatory (`conftest.py` reads it at import
time and fails without it); `torch` and `pytorch` are both accepted.

```bash
pip install -e . --no-build-isolation
export TEST_FASTSAFETENSORS_FRAMEWORK=pytorch

# everything except tests/unit/threefs (see below). GDS coverage is not
# guaranteed here: those tests run where GDS works and skip where it does not
python3 -m pytest tests/unit/ --ignore=tests/unit/threefs -q -rs

# same run on hardware that is supposed to have GDS -- now those skips are failures
FASTSAFETENSORS_TEST_REQUIRE_GDS=1 \
  python3 -m pytest tests/unit/ --ignore=tests/unit/threefs -q -rs

# one file, or pick by name
python3 -m pytest tests/unit/test_fastsafetensors.py -q -k "GdsFileCopier or hyfs"

# multi-GPU collectives
torchrun --nproc_per_node=2 --master_port=29500 -m pytest tests/unit/test_multi.py -q
```

Or via the Makefile: `make unittest`, `make unittest-parallel` (4-rank torchrun),
`make test-vllm`, `make unittest-paddle`, `make htmlcov`.

Test data: the `input_files` fixture looks for `.safetensors` under
`.testdata/transformers_cache/models--gpt2/` and downloads gpt2 from HuggingFace
if that directory is missing. **Offline hosts** must pre-create it. Any model
works for `test_fastsafetensors.py`, but `test_multi.py` hardcodes gpt2 keys
(`h.0..h.11.*`), so a stand-in must keep `n_layer=12`:

```python
import os                                           # ~21 MB, 148 tensors
from transformers import GPT2Config, GPT2Model
from safetensors.torch import save_file

d = ".testdata/transformers_cache/models--gpt2"
os.makedirs(d, exist_ok=True)
m = GPT2Model(GPT2Config(n_layer=12, n_embd=192, n_positions=64, vocab_size=512))
save_file({k: v.contiguous() for k, v in m.state_dict().items()},
          os.path.join(d, "model.safetensors"), {"format": "pt"})
```

`tests/unit/threefs/` needs the separate `fastsafetensor_3fs_reader` package;
without it the fixtures fall back to a `pread`-based mock, so those tests only
cover the integration contract, not real 3FS I/O.

GDS-only tests (`test_GdsFileCopier`, `test_SafeTensorsFileLoader`,
`test_shuffle[False]`) skip themselves when the host cannot actually do GDS.
The probe registers a real file handle rather than trusting `is_cufile_found()`
or `is_gds_supported()`: both stay true on a host that has the cuFile library
but no `nvidia-fs`/`hyfs` kernel module, or whose storage cannot do the DMA at
all (LVM, SATA, ...). Skipping matters in both directions -- without it a
missing hyfs makes the loader downgrade to nogds and the test passes while
covering nothing, and a missing `nvidia-fs` makes it fail on an environment
issue. CPU-only hosts fall back to `pread`, where the probe succeeds and those
tests keep running, as they do upstream.

Two environment variables affect the run:

- `FASTSAFETENSORS_TEST_REQUIRE_GDS=1` -- assert that this host is *supposed*
  to have GDS: the skips above become failures instead. Platform-independent,
  so it applies to NVIDIA and HCU alike; set it on any CI that runs on
  GDS-capable hardware, and losing the driver turns the suite red instead of
  quietly downgrading to nogds. On HCU it additionally requires that
  `libhyfile.so` loads (`test_hyfs_loaded_on_hcu`). Leave it unset on CPU-only
  CI, which never exercises real GDS.
- `FASTSAFETENSORS_FORCE_NOGDS` -- has **no** effect on tests: an autouse
  fixture clears it per test, so a globally exported value cannot silently turn
  the GDS tests into nogds runs.

# GDS/HCU Defaults

This branch keeps `nogds` defaulting to `False`, so the base loader uses GDS by
default when the platform support is available. The adjusted copy defaults are:

- `bbuf_size_kb`: `64 * 1024` KB (64 MiB)
- `max_copy_block_size`: `256 * 1024 * 1024` bytes (256 MiB)

Set `FASTSAFETENSORS_FORCE_NOGDS=1` to force the NoGDS path globally. Accepted
true values are `1`, `true`, `yes`, and `on`.

# Design Details

See [Overview](./docs/overview.md) for features, basic API usage, and configuration.

# Code of Conduct

Please refer to [Foundation Model Stack Community Code of Conduct](https://github.com/foundation-model-stack/foundation-model-stack/blob/main/code-of-conduct.md).

# Development

See [Development](./docs/development.md).

# Publication

Takeshi Yoshimura, Tatsuhiro Chiba, Manish Sethi, Daniel Waddington, Swaminathan Sundararaman. (2025) Speeding up Model Loading with fastsafetensors [arXiv:2505.23072](https://arxiv.org/abs/2505.23072) and IEEE CLOUD 2025.
