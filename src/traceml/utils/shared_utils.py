import torch.nn as nn

def model_is_on_cuda(model: nn.Module) -> bool:
    for p in model.parameters():
        return p.is_cuda
    for b in model.buffers():
        return b.is_cuda
    return False

