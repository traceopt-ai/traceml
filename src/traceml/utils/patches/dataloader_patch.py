from torch.utils.data import DataLoader
from traceml.utils.timing import timed_region


_ORIG_DATALOADER_ITER = DataLoader.__iter__


def _traceml_dataloader_iter(self):
    it = _ORIG_DATALOADER_ITER(self)

    while True:
        try:
            with timed_region(name="_traceml_internal:dataloader_next", scope="step", use_gpu=False):
                batch = next(it)
        except StopIteration:
            break

        yield batch


def patch_dataloader():
    """
    Patch torch.utils.data.DataLoader.__iter__ once.
    Safe to call multiple times.
    """
    if getattr(DataLoader, "_traceml_patched", False):
        return

    DataLoader.__iter__ = _traceml_dataloader_iter
    DataLoader._traceml_patched = True
