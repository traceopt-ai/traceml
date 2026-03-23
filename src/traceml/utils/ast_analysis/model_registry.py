"""Known model registry for static VRAM estimation.

Maps model identifiers (HuggingFace hub names, common class names) to
their parameter count and default training dtype.

bytes_per_param:
    4 = float32 (default PyTorch)
    2 = float16 / bfloat16 (common for LLM fine-tuning)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelSpec:
    num_params: int
    bytes_per_param: int = 4  # float32 default
    description: str = ""


# ---------------------------------------------------------------------------
# Registry: hub_name / class_name -> ModelSpec
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, ModelSpec] = {
    # -------------------------------------------------------------- BERT
    "bert-base-uncased": ModelSpec(110_000_000, 4, "BERT Base"),
    "bert-large-uncased": ModelSpec(340_000_000, 4, "BERT Large"),
    "bert-base-cased": ModelSpec(110_000_000, 4, "BERT Base Cased"),
    "bert-large-cased": ModelSpec(340_000_000, 4, "BERT Large Cased"),
    # ------------------------------------------------------------------ RoBERTa
    "roberta-base": ModelSpec(125_000_000, 4, "RoBERTa Base"),
    "roberta-large": ModelSpec(355_000_000, 4, "RoBERTa Large"),
    # ------------------------------------------------------------------ GPT-2
    "gpt2": ModelSpec(117_000_000, 4, "GPT-2 Small"),
    "gpt2-medium": ModelSpec(345_000_000, 4, "GPT-2 Medium"),
    "gpt2-large": ModelSpec(774_000_000, 4, "GPT-2 Large"),
    "gpt2-xl": ModelSpec(1_500_000_000, 4, "GPT-2 XL"),
    # ------------------------------------------------------------------ GPT-J / NeoX
    "EleutherAI/gpt-j-6b": ModelSpec(6_000_000_000, 2, "GPT-J 6B"),
    "EleutherAI/gpt-neox-20b": ModelSpec(20_000_000_000, 2, "GPT-NeoX 20B"),
    # ------------------------------------------------------------------ LLaMA 2
    "meta-llama/Llama-2-7b-hf": ModelSpec(7_000_000_000, 2, "LLaMA-2 7B"),
    "meta-llama/Llama-2-7b-chat-hf": ModelSpec(
        7_000_000_000, 2, "LLaMA-2 7B Chat"
    ),
    "meta-llama/Llama-2-13b-hf": ModelSpec(13_000_000_000, 2, "LLaMA-2 13B"),
    "meta-llama/Llama-2-13b-chat-hf": ModelSpec(
        13_000_000_000, 2, "LLaMA-2 13B Chat"
    ),
    "meta-llama/Llama-2-70b-hf": ModelSpec(70_000_000_000, 2, "LLaMA-2 70B"),
    "meta-llama/Llama-2-70b-chat-hf": ModelSpec(
        70_000_000_000, 2, "LLaMA-2 70B Chat"
    ),
    # ------------------------------------------------------------------ LLaMA 3
    "meta-llama/Meta-Llama-3-8B": ModelSpec(8_000_000_000, 2, "LLaMA-3 8B"),
    "meta-llama/Meta-Llama-3-8B-Instruct": ModelSpec(
        8_000_000_000, 2, "LLaMA-3 8B Instruct"
    ),
    "meta-llama/Meta-Llama-3-70B": ModelSpec(70_000_000_000, 2, "LLaMA-3 70B"),
    "meta-llama/Meta-Llama-3-70B-Instruct": ModelSpec(
        70_000_000_000, 2, "LLaMA-3 70B Instruct"
    ),
    # ------------------------------------------------------------------ Mistral / Mixtral
    "mistralai/Mistral-7B-v0.1": ModelSpec(7_000_000_000, 2, "Mistral 7B"),
    "mistralai/Mistral-7B-Instruct-v0.2": ModelSpec(
        7_000_000_000, 2, "Mistral 7B Instruct"
    ),
    "mistralai/Mixtral-8x7B-v0.1": ModelSpec(
        46_700_000_000, 2, "Mixtral 8x7B (MoE)"
    ),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ModelSpec(
        46_700_000_000, 2, "Mixtral 8x7B Instruct"
    ),
    # ------------------------------------------------------------------ Falcon
    "tiiuae/falcon-7b": ModelSpec(7_000_000_000, 2, "Falcon 7B"),
    "tiiuae/falcon-40b": ModelSpec(40_000_000_000, 2, "Falcon 40B"),
    "tiiuae/falcon-180B": ModelSpec(180_000_000_000, 2, "Falcon 180B"),
    # ------------------------------------------------------------------ Phi
    "microsoft/phi-1": ModelSpec(1_300_000_000, 4, "Phi-1"),
    "microsoft/phi-1_5": ModelSpec(1_300_000_000, 4, "Phi-1.5"),
    "microsoft/phi-2": ModelSpec(2_700_000_000, 4, "Phi-2"),
    "microsoft/Phi-3-mini-4k-instruct": ModelSpec(
        3_800_000_000, 2, "Phi-3 Mini"
    ),
    # ------------------------------------------------------------------ T5
    "t5-small": ModelSpec(60_000_000, 4, "T5 Small"),
    "t5-base": ModelSpec(220_000_000, 4, "T5 Base"),
    "t5-large": ModelSpec(770_000_000, 4, "T5 Large"),
    "t5-3b": ModelSpec(3_000_000_000, 4, "T5 3B"),
    "t5-11b": ModelSpec(11_000_000_000, 4, "T5 11B"),
    "google/flan-t5-base": ModelSpec(250_000_000, 4, "Flan-T5 Base"),
    "google/flan-t5-large": ModelSpec(780_000_000, 4, "Flan-T5 Large"),
    "google/flan-t5-xl": ModelSpec(3_000_000_000, 4, "Flan-T5 XL"),
    "google/flan-t5-xxl": ModelSpec(11_000_000_000, 4, "Flan-T5 XXL"),
    # ------------------------------------------------------------------ DistilBERT
    "distilbert-base-uncased": ModelSpec(66_000_000, 4, "DistilBERT Base"),
    "distilbert-base-cased": ModelSpec(66_000_000, 4, "DistilBERT Base Cased"),
    # ------------------------------------------------------------------ Compact BERT variants (prajjwal1)
    "prajjwal1/bert-tiny": ModelSpec(4_400_000, 4, "BERT Tiny"),
    "prajjwal1/bert-mini": ModelSpec(11_300_000, 4, "BERT Mini"),
    "prajjwal1/bert-small": ModelSpec(29_100_000, 4, "BERT Small"),
    "prajjwal1/bert-medium": ModelSpec(41_700_000, 4, "BERT Medium"),
    # ------------------------------------------------------------------ MiniLM
    "microsoft/MiniLM-L12-H384-uncased": ModelSpec(
        33_000_000, 4, "MiniLM L12"
    ),
    "microsoft/MiniLM-L6-H384-uncased": ModelSpec(22_000_000, 4, "MiniLM L6"),
    "sentence-transformers/all-MiniLM-L6-v2": ModelSpec(
        22_000_000, 4, "all-MiniLM-L6"
    ),
    "sentence-transformers/all-MiniLM-L12-v2": ModelSpec(
        33_000_000, 4, "all-MiniLM-L12"
    ),
    # ------------------------------------------------------------------ BART
    "facebook/bart-base": ModelSpec(139_000_000, 4, "BART Base"),
    "facebook/bart-large": ModelSpec(406_000_000, 4, "BART Large"),
    "facebook/bart-large-cnn": ModelSpec(406_000_000, 4, "BART Large CNN"),
    # ------------------------------------------------------------------ OPT
    "facebook/opt-125m": ModelSpec(125_000_000, 4, "OPT 125M"),
    "facebook/opt-1.3b": ModelSpec(1_300_000_000, 4, "OPT 1.3B"),
    "facebook/opt-6.7b": ModelSpec(6_700_000_000, 2, "OPT 6.7B"),
    "facebook/opt-13b": ModelSpec(13_000_000_000, 2, "OPT 13B"),
    "facebook/opt-30b": ModelSpec(30_000_000_000, 2, "OPT 30B"),
    "facebook/opt-66b": ModelSpec(66_000_000_000, 2, "OPT 66B"),
    # ------------------------------------------------------------------ Bloom
    "bigscience/bloom-560m": ModelSpec(560_000_000, 4, "BLOOM 560M"),
    "bigscience/bloom-1b7": ModelSpec(1_700_000_000, 4, "BLOOM 1.7B"),
    "bigscience/bloom-7b1": ModelSpec(7_100_000_000, 2, "BLOOM 7.1B"),
    "bigscience/bloom": ModelSpec(176_000_000_000, 2, "BLOOM 176B"),
    # ------------------------------------------------------------------ Vision
    "openai/clip-vit-base-patch32": ModelSpec(150_000_000, 4, "CLIP ViT-B/32"),
    "openai/clip-vit-large-patch14": ModelSpec(
        430_000_000, 4, "CLIP ViT-L/14"
    ),
    "google/vit-base-patch16-224": ModelSpec(86_000_000, 4, "ViT Base"),
    "google/vit-large-patch16-224": ModelSpec(307_000_000, 4, "ViT Large"),
    # ------------------------------------------------------------------ Whisper
    "openai/whisper-tiny": ModelSpec(39_000_000, 4, "Whisper Tiny"),
    "openai/whisper-base": ModelSpec(74_000_000, 4, "Whisper Base"),
    "openai/whisper-small": ModelSpec(244_000_000, 4, "Whisper Small"),
    "openai/whisper-medium": ModelSpec(769_000_000, 4, "Whisper Medium"),
    "openai/whisper-large-v2": ModelSpec(1_550_000_000, 4, "Whisper Large v2"),
    # ------------------------------------------------------------------ Qwen
    "Qwen/Qwen1.5-7B": ModelSpec(7_000_000_000, 2, "Qwen 1.5 7B"),
    "Qwen/Qwen1.5-14B": ModelSpec(14_000_000_000, 2, "Qwen 1.5 14B"),
    "Qwen/Qwen1.5-72B": ModelSpec(72_000_000_000, 2, "Qwen 1.5 72B"),
    "Qwen/Qwen2-7B": ModelSpec(7_600_000_000, 2, "Qwen2 7B"),
    "Qwen/Qwen2-72B": ModelSpec(72_000_000_000, 2, "Qwen2 72B"),
    # ------------------------------------------------------------------ Gemma
    "google/gemma-2b": ModelSpec(2_000_000_000, 2, "Gemma 2B"),
    "google/gemma-7b": ModelSpec(7_000_000_000, 2, "Gemma 7B"),
    "google/gemma-2-9b": ModelSpec(9_000_000_000, 2, "Gemma 2 9B"),
    "google/gemma-2-27b": ModelSpec(27_000_000_000, 2, "Gemma 2 27B"),
    # ------------------------------------------------------------------ CodeLlama
    "codellama/CodeLlama-7b-hf": ModelSpec(7_000_000_000, 2, "CodeLlama 7B"),
    "codellama/CodeLlama-13b-hf": ModelSpec(
        13_000_000_000, 2, "CodeLlama 13B"
    ),
    "codellama/CodeLlama-34b-hf": ModelSpec(
        34_000_000_000, 2, "CodeLlama 34B"
    ),
    # ------------------------------------------------------------------ DeepSeek
    "deepseek-ai/deepseek-coder-6.7b-base": ModelSpec(
        6_700_000_000, 2, "DeepSeek Coder 6.7B"
    ),
    "deepseek-ai/deepseek-coder-33b-base": ModelSpec(
        33_000_000_000, 2, "DeepSeek Coder 33B"
    ),
    # ------------------------------------------------------------------ TinyLlama
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": ModelSpec(
        1_100_000_000, 2, "TinyLlama 1.1B"
    ),
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T": ModelSpec(
        1_100_000_000, 2, "TinyLlama 1.1B"
    ),
    # ------------------------------------------------------------------ StableLM
    "stabilityai/stablelm-3b-4e1t": ModelSpec(3_000_000_000, 2, "StableLM 3B"),
    "stabilityai/stablelm-tuned-alpha-7b": ModelSpec(
        7_000_000_000, 2, "StableLM 7B"
    ),
    # ------------------------------------------------------------------ MPT
    "mosaicml/mpt-7b": ModelSpec(7_000_000_000, 2, "MPT 7B"),
    "mosaicml/mpt-30b": ModelSpec(30_000_000_000, 2, "MPT 30B"),
    # ------------------------------------------------------------------ Yi
    "01-ai/Yi-6B": ModelSpec(6_000_000_000, 2, "Yi 6B"),
    "01-ai/Yi-34B": ModelSpec(34_000_000_000, 2, "Yi 34B"),
    # ------------------------------------------------------------------ Zephyr / Mistral derivatives
    "HuggingFaceH4/zephyr-7b-beta": ModelSpec(7_000_000_000, 2, "Zephyr 7B"),
    # ------------------------------------------------------------------ OLMo
    "allenai/OLMo-7B": ModelSpec(7_000_000_000, 2, "OLMo 7B"),
    # ------------------------------------------------------------------ SmolLM
    "HuggingFaceTB/SmolLM-135M": ModelSpec(135_000_000, 4, "SmolLM 135M"),
    "HuggingFaceTB/SmolLM-360M": ModelSpec(360_000_000, 4, "SmolLM 360M"),
    "HuggingFaceTB/SmolLM-1.7B": ModelSpec(1_700_000_000, 2, "SmolLM 1.7B"),
}


def lookup_model(model_name: str) -> Optional[ModelSpec]:
    """Look up a model by its exact hub name or a case-insensitive suffix match.

    Returns None if not found.
    """
    if not model_name:
        return None

    # Exact match first
    spec = _REGISTRY.get(model_name)
    if spec:
        return spec

    # Normalize: try lowercase for case-insensitive matching
    lower = model_name.lower()
    for key, spec in _REGISTRY.items():
        if key.lower() == lower:
            return spec

    # Partial suffix match: "Llama-2-7b" should match "meta-llama/Llama-2-7b-hf"
    for key, spec in _REGISTRY.items():
        if lower in key.lower():
            return spec

    return None


# ---------------------------------------------------------------------------
# Torchvision factory function registry
# ---------------------------------------------------------------------------
# Maps the torchvision.models function name (e.g. "vit_b_16") -> ModelSpec.
# These are called as model = vit_b_16(pretrained=True) or vit_b_16(num_classes=N)
# rather than via from_pretrained(), so they need a separate lookup path.
_TORCHVISION_FACTORIES: Dict[str, ModelSpec] = {
    # Vision Transformers
    "vit_b_16": ModelSpec(86_600_000, 4, "ViT-B/16"),
    "vit_b_32": ModelSpec(88_200_000, 4, "ViT-B/32"),
    "vit_l_16": ModelSpec(307_000_000, 4, "ViT-L/16"),
    "vit_l_32": ModelSpec(307_000_000, 4, "ViT-L/32"),
    "vit_h_14": ModelSpec(632_000_000, 4, "ViT-H/14"),
    # ResNets
    "resnet18": ModelSpec(11_700_000, 4, "ResNet-18"),
    "resnet34": ModelSpec(21_800_000, 4, "ResNet-34"),
    "resnet50": ModelSpec(25_600_000, 4, "ResNet-50"),
    "resnet101": ModelSpec(44_500_000, 4, "ResNet-101"),
    "resnet152": ModelSpec(60_200_000, 4, "ResNet-152"),
    "wide_resnet50_2": ModelSpec(68_900_000, 4, "Wide ResNet-50-2"),
    "wide_resnet101_2": ModelSpec(126_900_000, 4, "Wide ResNet-101-2"),
    # EfficientNet
    "efficientnet_b0": ModelSpec(5_300_000, 4, "EfficientNet-B0"),
    "efficientnet_b1": ModelSpec(7_800_000, 4, "EfficientNet-B1"),
    "efficientnet_b4": ModelSpec(19_300_000, 4, "EfficientNet-B4"),
    "efficientnet_b7": ModelSpec(66_300_000, 4, "EfficientNet-B7"),
    "efficientnet_v2_s": ModelSpec(21_500_000, 4, "EfficientNet-V2-S"),
    "efficientnet_v2_m": ModelSpec(54_100_000, 4, "EfficientNet-V2-M"),
    "efficientnet_v2_l": ModelSpec(118_500_000, 4, "EfficientNet-V2-L"),
    # ConvNeXt
    "convnext_tiny": ModelSpec(28_600_000, 4, "ConvNeXt-Tiny"),
    "convnext_small": ModelSpec(50_200_000, 4, "ConvNeXt-Small"),
    "convnext_base": ModelSpec(88_600_000, 4, "ConvNeXt-Base"),
    "convnext_large": ModelSpec(197_800_000, 4, "ConvNeXt-Large"),
    # DenseNet
    "densenet121": ModelSpec(8_000_000, 4, "DenseNet-121"),
    "densenet169": ModelSpec(14_100_000, 4, "DenseNet-169"),
    "densenet201": ModelSpec(20_000_000, 4, "DenseNet-201"),
    # MobileNet
    "mobilenet_v2": ModelSpec(3_500_000, 4, "MobileNet-V2"),
    "mobilenet_v3_small": ModelSpec(2_500_000, 4, "MobileNet-V3-Small"),
    "mobilenet_v3_large": ModelSpec(5_500_000, 4, "MobileNet-V3-Large"),
    # Swin Transformer
    "swin_t": ModelSpec(28_300_000, 4, "Swin-T"),
    "swin_s": ModelSpec(49_600_000, 4, "Swin-S"),
    "swin_b": ModelSpec(87_800_000, 4, "Swin-B"),
    "swin_v2_t": ModelSpec(28_400_000, 4, "Swin-V2-T"),
    "swin_v2_b": ModelSpec(87_900_000, 4, "Swin-V2-B"),
    # RegNet
    "regnet_y_8gf": ModelSpec(39_200_000, 4, "RegNet-Y-8GF"),
    "regnet_y_16gf": ModelSpec(83_600_000, 4, "RegNet-Y-16GF"),
    "regnet_y_32gf": ModelSpec(145_000_000, 4, "RegNet-Y-32GF"),
    # CLIP (OpenAI)
    "vit_b_16_clip": ModelSpec(149_600_000, 4, "CLIP ViT-B/16"),
}


def lookup_torchvision_factory(func_name: str) -> Optional[ModelSpec]:
    """Look up a torchvision.models factory function by its bare name.

    Handles calls like::

        model = vit_b_16(num_classes=365)
        model = torchvision.models.resnet50(pretrained=True)

    The *func_name* should be the leaf function name (e.g. ``"vit_b_16"``),
    not the fully-qualified path. Returns ``None`` when not found.
    """
    return _TORCHVISION_FACTORIES.get(func_name)


def list_known_models() -> Dict[str, ModelSpec]:
    """Return a copy of the full registry."""
    return dict(_REGISTRY)
