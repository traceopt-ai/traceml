import torch.nn as nn

subtree_param_cache: dict[nn.Module, int] = {}


def subtree_param_bytes(module: nn.Module) -> int:
    """Total parameter memory of this module INCLUDING descendants."""
    if module in subtree_param_cache:
        return subtree_param_cache[module]

    size = sum(
        p.numel() * p.element_size()
        for p in module.parameters(recurse=True)
    )
    subtree_param_cache[module] = size
    return size


def should_hook(
    module: nn.Module,
    min_memory_threshold,  # MB
) -> bool:
    """
    Hook logic:
    - If threshold is None → hook everything
    - If any child subtree >= threshold → descend
    - Otherwise → hook at this level
    """
    if min_memory_threshold is None:
        return True

    threshold_bytes = min_memory_threshold * 1024 * 1024
    children = list(module.children())

    # Leaf module
    if not children:
        return subtree_param_bytes(module) >= threshold_bytes

    # Non-leaf: descend if any child subtree is significant
    for child in children:
        if subtree_param_bytes(child) >= threshold_bytes:
            return False

    return True


def model_is_on_cuda(model: nn.Module) -> bool:
    for p in model.parameters():
        return p.is_cuda
    for b in model.buffers():
        return b.is_cuda
    return False
