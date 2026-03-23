"""Known GPU hardware specifications and recommendation logic.

Maps VRAM requirements to the smallest viable GPU configuration.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class HardwareSpec:
    name: str
    vram_gb: float
    tflops_fp16: float


# A simple catalog of common training GPUs
HARDWARE_CATALOG: Dict[str, HardwareSpec] = {
    "RTX_3090_24GB": HardwareSpec(
        name="RTX 3090 / 4090", vram_gb=24, tflops_fp16=300
    ),
    "L4_24GB": HardwareSpec(name="L4", vram_gb=24, tflops_fp16=120),
    "A10G_24GB": HardwareSpec(name="A10G", vram_gb=24, tflops_fp16=125),
    "A100_40GB": HardwareSpec(name="A100-40GB", vram_gb=40, tflops_fp16=312),
    "L40S_48GB": HardwareSpec(name="L40S", vram_gb=48, tflops_fp16=733),
    "A100_80GB": HardwareSpec(name="A100-80GB", vram_gb=80, tflops_fp16=312),
    "H100_80GB": HardwareSpec(name="H100-80GB", vram_gb=80, tflops_fp16=989),
}


def recommend_hardware(required_vram_gb: float) -> str:
    """Recommend the smallest viable single GPU, or a multi-GPU strategy."""
    # Add a 10% safety margin for CUDA contexts
    target_vram = required_vram_gb * 1.1

    # Find smallest single GPU
    sorted_gpus = sorted(HARDWARE_CATALOG.values(), key=lambda x: x.vram_gb)
    for gpu in sorted_gpus:
        if gpu.vram_gb >= target_vram:
            return f"1x {gpu.name} ({gpu.vram_gb}GB)"

    # If it exceeds the largest single GPU (80GB), recommend multi-GPU FSDP
    num_80gbs = int((target_vram + 79) // 80)
    # Typically cluster sizes are powers of 2
    if num_80gbs <= 2:
        return "2x A100/H100-80GB (FSDP / ZeRO-3)"
    elif num_80gbs <= 4:
        return "4x A100/H100-80GB (FSDP / ZeRO-3)"
    elif num_80gbs <= 8:
        return "8x A100/H100-80GB (FSDP / ZeRO-3)"
    else:
        return f"{num_80gbs}x 80GB GPUs (FSDP / ZeRO-3) across multiple nodes"
