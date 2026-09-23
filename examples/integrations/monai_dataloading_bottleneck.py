"""Find where a MONAI training run waits: spleen segmentation.

Trains a 3D UNet on patches from the Medical Segmentation Decathlon spleen
task (Task09_Spleen, CC BY-SA 4.0) under one MONAI ``SupervisedTrainer``.
Each flag changes one setting, so two runs that differ in one flag measure
that setting alone:

    --dataset plain|cache    Dataset, or CacheDataset(cache_rate=1.0)
    --num-workers N          loader worker processes
    --loader torch|thread    DataLoader, or ThreadDataLoader
    --amp                    SupervisedTrainer(amp=True)

Launch through ``traceml run`` (a bare ``python`` run trains untraced):

    traceml run --mode summary --logs-dir logs --run-name spleen_1_baseline \\
        monai_dataloading_bottleneck.py --args --data-dir data
    traceml run --mode summary --logs-dir logs --run-name spleen_2_workers \\
        monai_dataloading_bottleneck.py --args --data-dir data \\
        --num-workers 4
    traceml compare logs/spleen_1_baseline/final_summary.json \\
        logs/spleen_2_workers/final_summary.json

CPU-only check without the dataset (what the notebook smoke job runs):

    traceml run --mode summary --logs-dir logs --run-name monai_smoke \\
        monai_dataloading_bottleneck.py --args --smoke --epochs 2

The spleen archive is downloaded into ``--data-dir`` on first use.
``--smoke`` writes small synthetic volumes to a temporary directory and
sends them through the same transforms and patch sampler. Reading NIfTI
files needs ``nibabel``.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

import ignite
import monai
import numpy as np
import torch
from monai.apps import download_and_extract
from monai.data import CacheDataset, DataLoader, Dataset, ThreadDataLoader
from monai.engines import SupervisedTrainer
from monai.losses import DiceLoss
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.utils import set_determinism

import traceml_ai
from traceml_ai.integrations import monai as traceml_monai

SEED = 42
SPLEEN_URL = (
    "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
)
SPLEEN_MD5 = "410d4a301da4e5b2f6f86ec3ddba524e"
# The task has 41 labelled volumes. MONAI's spleen tutorial keeps the last
# nine for validation; this study trains on the same 32 and runs no
# validation.
LABELLED = 41
HELD_OUT = 9
PATCH = (96, 96, 96)
PATCHES_PER_VOLUME = 4
CACHE_RATE = 1.0
# Fixed, so the cache is built the same way whatever the loader settings.
CACHE_WORKERS = 4
SMOKE_PATCH = (32, 32, 32)
SMOKE_SHAPE = (64, 64, 40)
SMOKE_VOLUMES = 4


def spleen_files(data_dir):
    """Download Task09_Spleen once and return the 32 training pairs."""
    root = os.path.join(data_dir, "Task09_Spleen")
    if not os.path.isdir(root):
        download_and_extract(
            SPLEEN_URL,
            os.path.join(data_dir, "Task09_Spleen.tar"),
            data_dir,
            hash_val=SPLEEN_MD5,
        )
    images = sorted(glob.glob(os.path.join(root, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(root, "labelsTr", "*.nii.gz")))
    # An interrupted extraction leaves a partial directory, which would
    # otherwise train on fewer volumes without saying so.
    if len(images) != LABELLED or len(labels) != LABELLED:
        raise RuntimeError(
            f"Expected {LABELLED} labelled volumes under {root}, found "
            f"{len(images)} images and {len(labels)} labels. Delete the "
            "directory and run again to download a fresh copy."
        )
    pairs = [{"image": i, "label": s} for i, s in zip(images, labels)]
    return pairs[:-HELD_OUT]


def write_synthetic_volumes(directory, count=SMOKE_VOLUMES):
    """Write small CT-like volumes, each with one bright labelled organ."""
    import nibabel as nib

    rng = np.random.default_rng(SEED)
    affine = np.diag([0.8, 0.8, 2.5, 1.0])
    grid = np.indices(SMOKE_SHAPE)
    pairs = []
    for index in range(count):
        centre = [rng.integers(12, size - 12) for size in SMOKE_SHAPE]
        distance = sum((g - c) ** 2 for g, c in zip(grid, centre))
        organ = distance <= 8**2
        image = rng.normal(0.0, 30.0, SMOKE_SHAPE) + 100.0 * organ
        pair = {}
        for key, array in (
            ("image", image.astype(np.float32)),
            ("label", organ.astype(np.uint8)),
        ):
            pair[key] = os.path.join(directory, f"{key}_{index}.nii.gz")
            nib.save(nib.Nifti1Image(array, affine), pair[key])
        pairs.append(pair)
    return pairs


def train_transforms(patch):
    """
    MONAI's spleen tutorial preprocessing, then its random patch sampler.

    Everything before ``RandCropByPosNegLabeld`` is deterministic, so
    ``CacheDataset`` stores its output once and only the crop runs per
    iteration.
    """
    keys = ["image", "label"]
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="label",
                spatial_size=patch,
                pos=1,
                neg=1,
                num_samples=PATCHES_PER_VOLUME,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )


def build_dataset(kind, files, patch):
    transforms = train_transforms(patch)
    if kind == "cache":
        return CacheDataset(
            files,
            transforms,
            cache_rate=CACHE_RATE,
            num_workers=CACHE_WORKERS,
            progress=False,
        )
    return Dataset(files, transforms)


def build_loader(kind, dataset, batch_size, num_workers):
    # MONAI's loaders collate the sampler's patches into one batch, so a
    # batch holds batch_size * PATCHES_PER_VOLUME patches.
    loader = ThreadDataLoader if kind == "thread" else DataLoader
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def build_network():
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )


def _gpu_and_driver(device):
    if device.type != "cuda":
        return None, None
    # The driver is host-wide, so any row answers. nvidia-smi numbers GPUs
    # physically, not the way CUDA_VISIBLE_DEVICES renumbers them.
    try:
        rows = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        rows = []
    return torch.cuda.get_device_name(device), rows[0] if rows else None


def run_record(argv, trainer, device, patch, dataset_s, train_s):
    """
    Everything needed to repeat or compare this run, as one dict.

    The dataset, loader, worker and AMP settings are read back from the
    trainer and its loader, so the record shows what ran rather than what
    was asked for. The rest are the script's fixed settings.
    """
    gpu, driver = _gpu_and_driver(device)
    loader = trainer.data_loader
    cached = isinstance(loader.dataset, CacheDataset)
    return {
        "argv": list(argv),
        "gpu": gpu,
        "driver": driver,
        "torch": torch.__version__,
        "monai": monai.__version__,
        "ignite": ignite.__version__,
        "traceml": traceml_ai.__version__,
        "seed": SEED,
        "dataset": type(loader.dataset).__name__,
        "volumes": len(loader.dataset),
        "cache_rate": CACHE_RATE if cached else None,
        "cached_volumes": loader.dataset.cache_num if cached else None,
        "loader": type(loader).__name__,
        "num_workers": loader.num_workers,
        "amp": trainer.amp,
        "batch_size": loader.batch_size,
        "patches_per_volume": PATCHES_PER_VOLUME,
        "patch_size": list(patch),
        "epochs": trainer.state.max_epochs,
        "steps": trainer.state.iteration,
        "dataset_ready_s": round(dataset_s, 3),
        "train_s": round(train_s, 3),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "3D UNet on MONAI's spleen task under SupervisedTrainer; each "
            "flag changes one setting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["plain", "cache"],
        default="plain",
        help="Dataset, or CacheDataset holding every preprocessed volume.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--loader",
        choices=["torch", "thread"],
        default="torch",
        help="monai.data.DataLoader, or ThreadDataLoader.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Mixed precision in SupervisedTrainer, a compute setting.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Where the spleen archive is downloaded and extracted.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CPU-only synthetic volumes; downloads nothing.",
    )
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    set_determinism(seed=SEED)

    # TraceML line 1: arms H2D timing. Input Wait comes from the engine's
    # own batch-fetch events, so it is right for ThreadDataLoader too.
    traceml_monai.init()

    cuda = torch.cuda.is_available() and not args.smoke
    device = torch.device("cuda" if cuda else "cpu")
    patch = SMOKE_PATCH if args.smoke else PATCH
    print(
        f"[demo] dataset={args.dataset} num_workers={args.num_workers} "
        f"loader={args.loader} amp={args.amp} smoke={args.smoke} "
        f"device={device} batch={args.batch_size} patch={patch} "
        f"epochs={args.epochs}",
        flush=True,
    )

    with contextlib.ExitStack() as stack:
        if args.smoke:
            scratch = stack.enter_context(tempfile.TemporaryDirectory())
            files = write_synthetic_volumes(scratch)
        else:
            files = spleen_files(args.data_dir)

        # CacheDataset preprocesses every volume here, before training, so
        # none of this time is inside a TraceML step.
        start = time.perf_counter()
        dataset = build_dataset(args.dataset, files, patch)
        dataset_s = time.perf_counter() - start
        print(f"[demo] dataset ready in {dataset_s:.1f} s", flush=True)

        network = build_network().to(device)
        trainer = SupervisedTrainer(
            device=device,
            max_epochs=args.epochs,
            train_data_loader=build_loader(
                args.loader, dataset, args.batch_size, args.num_workers
            ),
            network=network,
            optimizer=torch.optim.Adam(network.parameters(), lr=1e-4),
            loss_function=DiceLoss(to_onehot_y=True, softmax=True),
            amp=args.amp,
            train_handlers=[traceml_monai.TraceMLHandler()],  # TraceML line 2
        )
        start = time.perf_counter()
        trainer.run()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_s = time.perf_counter() - start

    record = run_record(argv, trainer, device, patch, dataset_s, train_s)
    print("[record] " + json.dumps(record, sort_keys=True), flush=True)
    return record


if __name__ == "__main__":
    main()
