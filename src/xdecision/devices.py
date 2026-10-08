"""Hardware-aware PyTorch device and training precision selection."""
import time
from contextlib import nullcontext

import torch


def pick_device(name=None):
    """Prefer visible CUDA devices by free memory, then MPS, then CPU.

    This is a capability policy, not a throughput benchmark. Explicit requests
    never silently fall back. CUDA indices respect CUDA_VISIBLE_DEVICES.
    """
    if name is None or str(name) == "auto":
        if torch.cuda.is_available():
            index = max(range(torch.cuda.device_count()),
                        key=lambda i: torch.cuda.mem_get_info(i)[0])
            return torch.device("cuda", index)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable: install a CUDA-enabled PyTorch build and check the NVIDIA driver")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is not visible")
        return torch.device("cuda", index)
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS unavailable on this host/PyTorch build")
        if device.index not in (None, 0):
            raise ValueError("MPS exposes only device 0")
        return torch.device("mps")
    if device.type != "cpu":
        raise ValueError("Supported devices: auto, cuda[:index], mps, cpu")
    return torch.device("cpu")


_CPU_BF16_PROBE = None


def probe_cpu_bf16(rows=1024, width=768, trials=3):
    """Measure whether BF16 autocast runs on this CPU and is faster than FP32.

    No instruction set is assumed. The probe times a transformer-sized matmul both ways and
    accepts BF16 only if it runs, agrees with FP32 and takes under 80% of the FP32 time;
    emulated BF16 is slower than FP32 and is rejected, and any error means FP32. The result
    is measured once per process.
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


def supports_bf16(device):
    if device.type == "cuda":
        with torch.cuda.device(device):
            return torch.cuda.is_bf16_supported(including_emulation=False)
    if device.type == "mps":
        return torch.backends.mps.is_macos_or_newer(14, 0)
    if device.type == "cpu":
        return probe_cpu_bf16()["supported"]
    return False


def pick_precision(device, name="auto"):
    if name == "auto":
        if supports_bf16(device):
            return "bf16"
        return "fp16" if device.type == "cuda" else "fp32"
    if name not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"Unknown precision: {name}")
    if name == "fp16" and device.type != "cuda":
        raise ValueError("FP16 training is supported only on CUDA with gradient scaling")
    if name == "bf16" and not supports_bf16(device):
        if device.type == "cpu":
            raise ValueError(f"BF16 did not run faster than FP32 on this CPU ({probe_cpu_bf16()}); use fp32")
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
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_stats(device):
    """CPU RSS and MPS driver memory are not CUDA allocator peak memory."""
    if device.type == "cuda":
        return {"allocated_gb": torch.cuda.memory_allocated(device) / 2**30,
                "reserved_gb": torch.cuda.memory_reserved(device) / 2**30,
                "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30}
    if device.type == "mps":
        return {"allocated_gb": torch.mps.current_allocated_memory() / 2**30,
                "driver_allocated_gb": torch.mps.driver_allocated_memory() / 2**30}
    return {}


def device_info(device, precision):
    info = {"device": str(device), "precision": precision, "framework": "pytorch",
            "gradient_scaling": precision == "fp16", "torch_version": str(torch.__version__)}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        info.update(name=torch.cuda.get_device_name(device),
                    free_gb=round(free / 2**30, 2), total_gb=round(total / 2**30, 2))
    if device.type == "cpu":
        info["cpu_bf16_probe"] = probe_cpu_bf16()
    return info
