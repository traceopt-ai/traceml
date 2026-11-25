# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from traceml_ai.utils.ast_analysis import analyze_script, build_code_manifest


def test_lightning_trainer_is_not_reported_as_hf_trainer(tmp_path) -> None:
    script = tmp_path / "train.py"
    script.write_text(
        "import pytorch_lightning as pl\n"
        "from lightning.pytorch import Trainer\n"
        "trainer = pl.Trainer(max_epochs=1)\n"
        "other = Trainer(max_epochs=1)\n"
        "trainer.fit(model)\n",
        encoding="utf-8",
    )

    manifest = build_code_manifest(analyze_script(str(script)))

    assert manifest["distributed"]["lightning"] is True
    assert manifest["distributed"]["hf_trainer"] is False
    assert manifest["framework"]["huggingface"] is False


def test_hf_trainer_is_still_detected(tmp_path) -> None:
    script = tmp_path / "train.py"
    script.write_text(
        "from transformers import Trainer\n"
        "trainer = Trainer(model=model)\n"
        "trainer.train()\n",
        encoding="utf-8",
    )

    manifest = build_code_manifest(analyze_script(str(script)))

    assert manifest["distributed"]["hf_trainer"] is True
    assert manifest["distributed"]["lightning"] is False
    assert manifest["framework"]["huggingface"] is True
