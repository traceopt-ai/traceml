"""Train a small UNet on synthetic volumes with MONAI and TraceML."""

from __future__ import annotations

import argparse

SEED = 42
VOLUME_SHAPE = (1, 32, 32, 32)
DATASET_SIZE = 32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal MONAI SupervisedTrainer traced by TraceML. Everything "
            "is synthetic, so nothing is downloaded."
        )
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="One TraceML step is one optimizer update, so a value above 1 "
        "makes a step span that many iterations.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Loader worker processes. The volumes here are already in "
        "memory, so raising this adds startup and transfer cost rather "
        "than removing Input Wait.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda. H2D time is only measured on cuda.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Keep --help usable without the optional stack installed.
    import torch
    from monai.data import DataLoader, Dataset
    from monai.engines import SupervisedTrainer
    from monai.networks.nets import UNet

    from traceml_ai.integrations import monai as traceml_monai

    # The only TraceML init in the process. It arms H2D timing and leaves
    # the torch DataLoader patch off, because the engine's own fetch events
    # are what Input Wait is measured from here.
    traceml_monai.init()

    torch.manual_seed(SEED)
    device = torch.device(args.device)

    # Random volumes, so the example runs anywhere and downloads nothing.
    records = [
        {
            "image": torch.randn(*VOLUME_SHAPE),
            "label": torch.randint(0, 2, VOLUME_SHAPE).float(),
        }
        for _ in range(DATASET_SIZE)
    ]
    loader = DataLoader(
        Dataset(records),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    network = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(4, 8, 16),
        strides=(2, 2),
        num_res_units=1,
    ).to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=args.epochs,
        train_data_loader=loader,
        network=network,
        optimizer=optimizer,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        accumulation_steps=args.accumulation_steps,
        train_handlers=[traceml_monai.TraceMLHandler()],
    )

    # Count the real updates, so the line below can be compared with the step
    # count in the summary. They should agree.
    updates = []
    optimizer.register_step_post_hook(lambda *_: updates.append(1))

    trainer.run()

    print(
        f"iterations: {trainer.state.iteration}, "
        f"optimizer updates: {len(updates)}"
    )


if __name__ == "__main__":
    main()
