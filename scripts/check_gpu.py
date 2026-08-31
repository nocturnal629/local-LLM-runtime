"""Detect the local NVIDIA GPU and confirm it's visible to CUDA."""

import sys

import pynvml


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def main() -> int:
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        print(f"No NVIDIA GPU detected or driver not accessible: {exc}")
        return 1

    try:
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            print("nvmlInit succeeded but no GPUs were found.")
            return 1

        driver = _decode(pynvml.nvmlSystemGetDriverVersion())
        print(f"Driver version: {driver}")

        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = _decode(pynvml.nvmlDeviceGetName(handle))
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = mem.total / (1024**3)
            free_gb = mem.free / (1024**3)
            print(f"GPU {i}: {name}")
            print(f"  VRAM total: {total_gb:.1f} GB, free: {free_gb:.1f} GB")

        return 0
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    sys.exit(main())
