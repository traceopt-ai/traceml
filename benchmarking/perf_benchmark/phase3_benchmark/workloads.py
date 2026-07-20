"""Synthetic benchmark workloads for publishable TraceML overhead runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class WorkloadSpec:
    model: str
    batch_size: int
    dataloader: str = "synthetic"
    input_dim: int | None = None
    hidden_dim: int | None = None
    layers: int | None = None
    num_classes: int = 1000
    seq_len: int = 128
    vocab_size: int = 50257
    realistic_num_workers: int = 2
    pin_memory: bool = True


class PrebuiltTensorDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class RandomVectorDataset(Dataset):
    def __init__(self, *, rows: int, input_dim: int, num_classes: int) -> None:
        self.rows = int(rows)
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        del index
        return torch.randn(self.input_dim), torch.randint(
            0, self.num_classes, ()
        )


class RandomTokenDataset(Dataset):
    def __init__(self, *, rows: int, seq_len: int, vocab_size: int) -> None:
        self.rows = int(rows)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        del index
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        y = torch.randint(0, self.vocab_size, (self.seq_len,))
        return x, y


class SyntheticBatchIterator:
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __iter__(self) -> "SyntheticBatchIterator":
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x, self.y


class MLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        dim = input_dim
        for _ in range(layers):
            modules.append(nn.Linear(dim, hidden_dim))
            modules.append(nn.GELU())
            dim = hidden_dim
        modules.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        hidden_dim: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.token(x) + self.pos[:, : x.shape[1], :]
        return self.lm_head(self.encoder(hidden))


def model_shape(spec: WorkloadSpec) -> tuple[int, int, int]:
    presets = {
        "tiny_mlp": (
            spec.input_dim or 512,
            spec.hidden_dim or 512,
            spec.layers or 2,
        ),
        "small_mlp": (
            spec.input_dim or 2048,
            spec.hidden_dim or 2048,
            spec.layers or 4,
        ),
        "wide_mlp": (
            spec.input_dim or 4096,
            spec.hidden_dim or 4096,
            spec.layers or 6,
        ),
    }
    return presets[spec.model]


def is_transformer(model_name: str) -> bool:
    return model_name in {"tiny_transformer", "transformer_small"}


def build_model(spec: WorkloadSpec) -> tuple[nn.Module, dict]:
    if is_transformer(spec.model):
        hidden = spec.hidden_dim or (
            384 if spec.model == "tiny_transformer" else 768
        )
        layers = spec.layers or (4 if spec.model == "tiny_transformer" else 12)
        heads = 6 if hidden % 6 == 0 else 8
        model = TinyTransformerLM(
            vocab_size=spec.vocab_size,
            seq_len=spec.seq_len,
            hidden_dim=hidden,
            layers=layers,
            heads=heads,
        )
        return model, {
            "name": spec.model,
            "hidden_dim": hidden,
            "layers": layers,
            "heads": heads,
            "seq_len": spec.seq_len,
            "vocab_size": spec.vocab_size,
            "batch_size": spec.batch_size,
            "dataloader": spec.dataloader,
        }

    input_dim, hidden_dim, layers = model_shape(spec)
    model = MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        layers=layers,
        num_classes=spec.num_classes,
    )
    return model, {
        "name": spec.model,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "layers": layers,
        "num_classes": spec.num_classes,
        "batch_size": spec.batch_size,
        "dataloader": spec.dataloader,
    }


def build_batch_iterator(
    spec: WorkloadSpec,
    *,
    total_batches: int,
    seed: int,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    rows = total_batches * spec.batch_size
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    if is_transformer(spec.model):
        x = torch.randint(
            0,
            spec.vocab_size,
            (spec.batch_size, spec.seq_len),
            generator=gen,
        )
        y = torch.randint(
            0,
            spec.vocab_size,
            (spec.batch_size, spec.seq_len),
            generator=gen,
        )
        if spec.dataloader == "synthetic":
            return SyntheticBatchIterator(x, y)
        dataset: Dataset = (
            PrebuiltTensorDataset(
                x.repeat(total_batches, 1),
                y.repeat(total_batches, 1),
            )
            if spec.dataloader == "torch_synthetic"
            else RandomTokenDataset(
                rows=rows,
                seq_len=spec.seq_len,
                vocab_size=spec.vocab_size,
            )
        )
    else:
        input_dim, _, _ = model_shape(spec)
        x = torch.randn(spec.batch_size, input_dim, generator=gen)
        y = torch.randint(
            0, spec.num_classes, (spec.batch_size,), generator=gen
        )
        if spec.dataloader == "synthetic":
            return SyntheticBatchIterator(x, y)
        dataset = (
            PrebuiltTensorDataset(
                x.repeat(total_batches, 1),
                y.repeat(total_batches),
            )
            if spec.dataloader == "torch_synthetic"
            else RandomVectorDataset(
                rows=rows,
                input_dim=input_dim,
                num_classes=spec.num_classes,
            )
        )

    workers = (
        0
        if spec.dataloader == "torch_synthetic"
        else spec.realistic_num_workers
    )
    return iter(
        DataLoader(
            dataset,
            batch_size=spec.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=spec.pin_memory,
            drop_last=True,
            persistent_workers=workers > 0,
        )
    )
