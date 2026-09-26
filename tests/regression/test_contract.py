# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the declared regression measurement contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from traceml_ai.regression.contract import (
    MAX_PARAMETER_COUNT,
    MAX_PARAMETER_KEY_LENGTH,
    MAX_PARAMETER_STRING_LENGTH,
    MAX_SIGNED_INT,
    MAX_WORKLOAD_NAME_LENGTH,
    MIN_SIGNED_INT,
    ContractValidationError,
    parse_guard_contract,
)


def _contract(**overrides):
    raw = {
        "schema_version": 1,
        "workload": {"name": "image-training"},
        "measurement": {"start_step": 10, "completed_steps": 50},
    }
    raw.update(overrides)
    return raw


def test_minimal_contract_normalizes_optional_parameters() -> None:
    contract = parse_guard_contract(_contract())

    assert contract.end_step == 59
    assert contract.to_dict() == {
        "schema_version": 1,
        "workload": {"name": "image-training", "parameters": {}},
        "measurement": {"start_step": 10, "completed_steps": 50},
    }


def test_contract_trims_name_and_sorts_scalar_parameters() -> None:
    contract = parse_guard_contract(
        _contract(
            workload={
                "name": "  image-training  ",
                "parameters": {
                    "precision": "bf16",
                    "enabled": True,
                    "batch_size": 32,
                    "dropout": 0.1,
                },
            }
        )
    )

    assert contract.workload_parameters == (
        ("batch_size", 32),
        ("dropout", 0.1),
        ("enabled", True),
        ("precision", "bf16"),
    )
    assert contract.to_dict()["workload"]["name"] == "image-training"


def test_contract_is_independent_from_source_mapping() -> None:
    raw = _contract(
        workload={"name": "before", "parameters": {"model": "small"}}
    )
    contract = parse_guard_contract(raw)

    raw["workload"]["name"] = "after"
    raw["workload"]["parameters"]["model"] = "large"

    assert contract.to_dict()["workload"] == {
        "name": "before",
        "parameters": {"model": "small"},
    }
    with pytest.raises(FrozenInstanceError):
        contract.start_step = 20  # type: ignore[misc]


def test_contract_accepts_exact_supported_boundaries() -> None:
    parameters = {
        f"parameter_{index}": index for index in range(MAX_PARAMETER_COUNT - 1)
    }
    parameters["parameter_0"] = MIN_SIGNED_INT
    parameters["parameter_1"] = MAX_SIGNED_INT
    parameters["k" * MAX_PARAMETER_KEY_LENGTH] = (
        "v" * MAX_PARAMETER_STRING_LENGTH
    )

    contract = parse_guard_contract(
        _contract(
            workload={
                "name": "w" * MAX_WORKLOAD_NAME_LENGTH,
                "parameters": parameters,
            },
            measurement={"start_step": 1, "completed_steps": MAX_SIGNED_INT},
        )
    )

    assert len(contract.workload_parameters) == MAX_PARAMETER_COUNT
    assert contract.end_step == MAX_SIGNED_INT


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (None, "guard must be a mapping"),
        ({1: "invalid"}, "guard keys must be strings"),
        ({}, "guard.schema_version is required"),
        (_contract(schema_version=True), "schema_version must be an integer"),
        (_contract(schema_version=2), "schema_version is unsupported"),
        (_contract(extra=True), "unknown field.*guard.extra"),
        (
            _contract(workload={"name": "work", "extra": 1}),
            "unknown field.*guard.workload.extra",
        ),
        (
            _contract(
                measurement={
                    "start_step": 1,
                    "completed_steps": 1,
                    "extra": 1,
                }
            ),
            "unknown field.*guard.measurement.extra",
        ),
        (_contract(workload={}), "guard.workload.name is required"),
        (_contract(workload={"name": 42}), "name must be a string"),
        (_contract(workload={"name": "   "}), "name must not be empty"),
        (
            _contract(workload={"name": "x", "parameters": []}),
            "parameters must be a mapping",
        ),
        (
            _contract(workload={"name": "x", "parameters": {"nested": {}}}),
            "parameters.nested must be a string",
        ),
        (
            _contract(workload={"name": "x", "parameters": {"items": []}}),
            "parameters.items must be a string",
        ),
        (
            _contract(workload={"name": "x", "parameters": {"none": None}}),
            "parameters.none must be a string",
        ),
        (
            _contract(workload={"name": "x", "parameters": {"value": ""}}),
            "parameters.value must not be empty",
        ),
        (
            _contract(
                workload={"name": "x", "parameters": {"value": " padded "}}
            ),
            "parameters.value must not have surrounding whitespace",
        ),
        (
            _contract(
                workload={"name": "x", "parameters": {"value": float("inf")}}
            ),
            "parameters.value must be finite",
        ),
        (
            _contract(
                workload={"name": "x", "parameters": {"value": float("-inf")}}
            ),
            "parameters.value must be finite",
        ),
        (
            _contract(
                workload={"name": "x", "parameters": {"value": float("nan")}}
            ),
            "parameters.value must be finite",
        ),
        (
            _contract(measurement={"start_step": True, "completed_steps": 1}),
            "start_step must be an integer",
        ),
        (
            _contract(measurement={"start_step": 0, "completed_steps": 1}),
            "start_step must be >= 1",
        ),
        (
            _contract(measurement={"start_step": -1, "completed_steps": 1}),
            "start_step must be >= 1",
        ),
        (
            _contract(measurement={"start_step": 1, "completed_steps": False}),
            "completed_steps must be an integer",
        ),
        (
            _contract(measurement={"start_step": 1, "completed_steps": 0}),
            "completed_steps must be >= 1",
        ),
        (
            _contract(measurement={"start_step": 1, "completed_steps": -1}),
            "completed_steps must be >= 1",
        ),
        (
            _contract(
                measurement={
                    "start_step": MAX_SIGNED_INT,
                    "completed_steps": 2,
                }
            ),
            "requested step range exceeds",
        ),
    ],
)
def test_contract_rejects_invalid_declarations(raw, match) -> None:
    with pytest.raises(ContractValidationError, match=match):
        parse_guard_contract(raw)


@pytest.mark.parametrize(
    "workload",
    [
        {"name": "x" * (MAX_WORKLOAD_NAME_LENGTH + 1)},
        {
            "name": "x",
            "parameters": {
                f"key_{index}": index
                for index in range(MAX_PARAMETER_COUNT + 1)
            },
        },
        {
            "name": "x",
            "parameters": {"k" * (MAX_PARAMETER_KEY_LENGTH + 1): 1},
        },
        {"name": "x", "parameters": {" padded ": 1}},
        {
            "name": "x",
            "parameters": {"value": "x" * (MAX_PARAMETER_STRING_LENGTH + 1)},
        },
        {
            "name": "x",
            "parameters": {"value": MAX_SIGNED_INT + 1},
        },
    ],
)
def test_contract_enforces_workload_bounds(workload) -> None:
    with pytest.raises(ContractValidationError):
        parse_guard_contract(_contract(workload=workload))
