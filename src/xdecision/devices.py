"""Hardware-aware PyTorch device and training precision selection."""
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


def cpu_has_native_bf16():
    """AMX or AVX512-BF16 instructions, read from /proc/cpuinfo (Linux); False elsewhere.

    Without them oneDNN emulates BF16 and is slower than FP32. With AMX a training step on
    this model is about twice as fast as FP32.
    """
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
    except OSError:
        return False
    return "amx_bf16" in flags or "avx512_bf16" in flags


def supports_bf16(device):
    if device.type == "cuda":
        with torch.cuda.device(device):
            return torch.cuda.is_bf16_supported(including_emulation=False)
    if device.type == "mps":
        return torch.backends.mps.is_macos_or_newer(14, 0)
    if device.type == "cpu":
        return cpu_has_native_bf16()
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
    return info
