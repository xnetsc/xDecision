"""Hardware detection and training device/precision selection.

  python -m xdecision.devices      # print what was detected and what training would use

Nothing about the host is assumed. The CPU architecture (x86, ARM or anything else) and its
instruction sets are read from the operating system; GPUs are identified by the runtime that
drives them: NVIDIA CUDA, AMD ROCm (HIP), Intel XPU, Apple MPS through PyTorch, and MLX
(Metal or CUDA). CPU BF16 is used only when the instruction set reports BF16 (or cannot be
read) and a timing probe confirms that it runs, agrees with FP32 and is faster.
"""
import functools
import importlib.util
import json
import os
import platform
import subprocess
import time
from contextlib import nullcontext

import torch

ARCH = {"x86_64": "x86_64", "amd64": "x86_64", "x64": "x86_64", "i386": "x86", "i686": "x86", "x86": "x86",
        "arm64": "arm64", "aarch64": "arm64", "arm64e": "arm64", "armv7l": "arm", "armv8l": "arm", "arm": "arm"}
FAMILY = {"x86_64": "x86", "x86": "x86", "arm64": "arm", "arm": "arm"}
# Instruction sets worth reporting for this workload, and the ones that execute BF16 natively.
ISA = {"x86": ("sse4_2", "avx", "avx2", "fma", "f16c", "avx512f", "avx512bw", "avx512_vnni",
               "avx512_bf16", "avx512_fp16", "amx_tile", "amx_bf16", "amx_int8"),
       "arm": ("neon", "asimd", "advsimd", "asimdhp", "asimddp", "fhm", "sve", "sve2", "sme",
               "i8mm", "bf16", "ebf16")}
BF16_ISA = {"x86": ("avx512_bf16", "amx_bf16"), "arm": ("bf16", "ebf16")}
# macOS x86 sysctl spellings that differ from the Linux flag names.
SYSCTL_X86 = {"avx1_0": "avx", "avx512bf16": "avx512_bf16", "avx512vnni": "avx512_vnni",
              "avx512fp16": "avx512_fp16", "amxbf16": "amx_bf16", "amxtile": "amx_tile", "amxint8": "amx_int8"}


def cpu_arch(machine=None):
    name = (platform.machine() if machine is None else machine).lower()
    return ARCH.get(name, name or "unknown")


def parse_cpuinfo(text):
    """Linux /proc/cpuinfo: x86 lists `flags`, ARM lists `Features`; other keys are ignored."""
    flags = set()
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() in ("flags", "features"):
            flags.update(value.lower().split())
    return flags


def parse_sysctl(text, family):
    """macOS sysctl: `hw.optional.arm.FEAT_BF16: 1` on ARM, feature word lists on x86."""
    flags = set()
    if family == "arm":
        for line in text.splitlines():
            name, _, value = line.partition(":")
            if value.strip() == "1":
                flags.add(name.strip().rsplit(".", 1)[-1].lower().removeprefix("feat_"))
    else:
        for word in text.lower().split():
            word = word.replace(".", "_")
            flags.add(SYSCTL_X86.get(word, word))
    return flags


def _read_isa(system, family):
    if system == "Linux" and os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            return parse_cpuinfo(f.read()), "/proc/cpuinfo"
    if system == "Darwin":
        names = ["hw.optional"] if family == "arm" else ["-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"]
        out = subprocess.run(["sysctl", *names], capture_output=True, text=True, timeout=5)
        return parse_sysctl(out.stdout, family), "sysctl"
    return set(), None


@functools.lru_cache(maxsize=None)
def cpu_report():
    """Architecture and instruction sets as the operating system reports them."""
    system, arch = platform.system(), cpu_arch()
    family = FAMILY.get(arch, "other")
    try:
        flags, source = _read_isa(system, family)
    except (OSError, subprocess.SubprocessError):
        flags, source = set(), None
    capability = getattr(torch.backends.cpu, "get_cpu_capability", lambda: None)()
    bf16_isa = None
    if flags and family in BF16_ISA:
        bf16_isa = any(name in flags for name in BF16_ISA[family])
    return {"os": system, "arch": arch, "family": family, "processor": platform.processor() or None,
            "threads": torch.get_num_threads(), "isa_source": source if flags else None,
            "isa": sorted(flags & set(ISA.get(family, ()))) if family in ISA else sorted(flags)[:32],
            "bf16_isa": bf16_isa, "torch_cpu_capability": capability}


_CPU_BF16_PROBE = None


def probe_cpu_bf16(rows=1024, width=768, trials=3):
    """Measure whether BF16 autocast runs on this CPU and is faster than FP32.

    The probe times a transformer-sized matmul both ways and accepts BF16 only if it runs,
    agrees with FP32 and takes under 80% of the FP32 time; emulated BF16 is slower than FP32
    and is rejected, and any error means FP32. The result is measured once per process.
    """
    global _CPU_BF16_PROBE
    if _CPU_BF16_PROBE is None:
        result = {"supported": False}
        try:
            generator = torch.Generator().manual_seed(0)
            x = torch.randn(rows, width, generator=generator)
            w = torch.randn(4 * width, width, generator=generator) / width ** 0.5

            def timed(enabled):
                best, out = float("inf"), None
                for _ in range(trials + 1):  # the first run warms up kernels
                    start = time.perf_counter()
                    with torch.inference_mode(), torch.autocast("cpu", torch.bfloat16, enabled=enabled):
                        out = torch.nn.functional.linear(x, w)
                    best = min(best, time.perf_counter() - start)
                return best, out.float()

            fp32_s, reference = timed(False)
            bf16_s, value = timed(True)
            agrees = bool(torch.isfinite(value).all()) and torch.allclose(value, reference, rtol=0.05, atol=0.1)
            result = {"supported": agrees and bf16_s < 0.8 * fp32_s,
                      "fp32_ms": round(fp32_s * 1e3, 2), "bf16_ms": round(bf16_s * 1e3, 2)}
        except Exception as error:  # an unsupported autocast path means FP32
            result["error"] = f"{type(error).__name__}: {error}"
        _CPU_BF16_PROBE = result
    return _CPU_BF16_PROBE


def cpu_bf16():
    """The instruction set rules BF16 out without measuring; otherwise the probe decides."""
    if cpu_report()["bf16_isa"] is False:
        return {"supported": False, "reason": "instruction set reports no BF16"}
    return probe_cpu_bf16()


def _xpu():
    xpu = getattr(torch, "xpu", None)
    return xpu if xpu is not None and xpu.is_available() else None


def cuda_runtime():
    """'rocm' when PyTorch drives an AMD GPU through HIP, 'cuda' for NVIDIA, None otherwise."""
    if not torch.cuda.is_available():
        return None
    return "rocm" if getattr(torch.version, "hip", None) else "cuda"


def mlx_report():
    """MLX availability and the GPU backend it would use (Metal, CUDA or none)."""
    if importlib.util.find_spec("mlx") is None:
        return {"installed": False, "gpu": None}
    try:
        import mlx.core as mx
        gpu = ("metal" if mx.metal.is_available() else
               "cuda" if getattr(getattr(mx, "cuda", None), "is_available", lambda: False)() else None)
        return {"installed": True, "gpu": gpu, "version": getattr(mx, "__version__", None)}
    except Exception as error:
        return {"installed": True, "gpu": None, "error": f"{type(error).__name__}: {error}"}


def accelerator_report():
    found = {}
    runtime = cuda_runtime()
    if runtime:
        found[runtime] = [{"index": i, "name": torch.cuda.get_device_name(i),
                           "bf16": supports_bf16(torch.device("cuda", i))}
                          for i in range(torch.cuda.device_count())]
    xpu = _xpu()
    if xpu is not None:
        found["xpu"] = [{"index": i, "name": xpu.get_device_name(i)} for i in range(xpu.device_count())]
    if torch.backends.mps.is_available():
        found["mps"] = {"bf16": torch.backends.mps.is_macos_or_newer(14, 0)}
    mlx = mlx_report()
    if mlx["installed"]:
        found["mlx"] = mlx
    return found


def pick_backend(requested="auto", device=None):
    """'torch' or 'mlx'. Auto prefers PyTorch CUDA/ROCm/XPU, then an MLX GPU, then PyTorch MPS/CPU."""
    if requested not in ("auto", "torch", "mlx"):
        raise ValueError("backend must be auto, torch or mlx")
    if requested == "mlx":
        mlx = mlx_report()
        if not mlx["installed"]:
            raise ValueError("MLX is not installed; pip install -e '.[apple]' or use --backend torch")
        return "mlx"
    if requested == "torch" or device not in (None, "auto"):
        return "torch"
    if cuda_runtime() or _xpu() is not None:
        return "torch"
    return "mlx" if mlx_report()["gpu"] else "torch"


def pick_device(name=None):
    """Prefer visible CUDA/ROCm devices by free memory, then Intel XPU, then MPS, then CPU.

    This is a capability policy, not a throughput benchmark. Explicit requests
    never silently fall back. CUDA indices respect CUDA_VISIBLE_DEVICES.
    """
    if name is None or str(name) == "auto":
        if torch.cuda.is_available():
            index = max(range(torch.cuda.device_count()),
                        key=lambda i: torch.cuda.mem_get_info(i)[0])
            return torch.device("cuda", index)
        if _xpu() is not None:
            return torch.device("xpu", 0)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable: install a CUDA- or ROCm-enabled PyTorch build and check the GPU driver")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is not visible")
        return torch.device("cuda", index)
    if device.type == "xpu":
        xpu = _xpu()
        if xpu is None:
            raise ValueError("Intel XPU unavailable on this host/PyTorch build")
        index = device.index or 0
        if index >= xpu.device_count():
            raise ValueError(f"XPU device index {index} is not visible")
        return torch.device("xpu", index)
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS unavailable on this host/PyTorch build")
        if device.index not in (None, 0):
            raise ValueError("MPS exposes only device 0")
        return torch.device("mps")
    if device.type != "cpu":
        raise ValueError("Supported devices: auto, cuda[:index] (NVIDIA or ROCm), xpu[:index], mps, cpu")
    return torch.device("cpu")


def supports_bf16(device):
    if device.type == "cuda":
        with torch.cuda.device(device):
            return torch.cuda.is_bf16_supported(including_emulation=False)
    if device.type == "xpu":
        return bool(getattr(torch.xpu, "is_bf16_supported", lambda: False)())
    if device.type == "mps":
        return torch.backends.mps.is_macos_or_newer(14, 0)
    if device.type == "cpu":
        return cpu_bf16()["supported"]
    return False


def pick_precision(device, name="auto"):
    if name == "auto":
        if supports_bf16(device):
            return "bf16"
        return "fp16" if device.type == "cuda" else "fp32"
    if name not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"Unknown precision: {name}")
    if name == "fp16" and device.type != "cuda":
        raise ValueError("FP16 training is supported only on CUDA/ROCm with gradient scaling")
    if name == "bf16" and not supports_bf16(device):
        if device.type == "cpu":
            raise ValueError(f"BF16 is not usable on this CPU ({cpu_bf16()}); use fp32")
        raise ValueError(f"Native BF16 is unavailable on {device}; use fp32")
    return name


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_stats(device):
    """CPU RSS and MPS driver memory are not CUDA allocator peak memory."""
    if device.type == "cuda":
        return {"allocated_gb": torch.cuda.memory_allocated(device) / 2**30,
                "reserved_gb": torch.cuda.memory_reserved(device) / 2**30,
                "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30}
    if device.type == "xpu":
        return {"allocated_gb": torch.xpu.memory_allocated(device) / 2**30,
                "reserved_gb": torch.xpu.memory_reserved(device) / 2**30}
    if device.type == "mps":
        return {"allocated_gb": torch.mps.current_allocated_memory() / 2**30,
                "driver_allocated_gb": torch.mps.driver_allocated_memory() / 2**30}
    return {}


def device_info(device, precision):
    info = {"device": str(device), "precision": precision, "framework": "pytorch",
            "gradient_scaling": precision == "fp16", "torch_version": str(torch.__version__)}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        info.update(runtime=cuda_runtime(), name=torch.cuda.get_device_name(device),
                    free_gb=round(free / 2**30, 2), total_gb=round(total / 2**30, 2))
    elif device.type == "xpu":
        info.update(runtime="xpu", name=torch.xpu.get_device_name(device))
    elif device.type == "cpu":
        cpu = cpu_report()
        info.update(arch=cpu["arch"], isa=cpu["isa"], cpu_bf16=cpu_bf16())
    return info


def main():
    device = pick_device()
    report = {"cpu": cpu_report(), "cpu_bf16": cpu_bf16(), "accelerators": accelerator_report(),
              "training": {"backend": pick_backend(), "torch_device": str(device),
                           "torch_precision": pick_precision(device)}}
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
