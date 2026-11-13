# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The wandb exporter streams step time into one W&B run per rank.

The real W&B client is never used: a stand-in ``wandb`` module installed in
``sys.modules`` records every call.
"""

from __future__ import annotations

import builtins
import os
import re
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import traceml_ai
from traceml_ai.aggregator.export import (
    DEFAULT_EXPORT_STREAMS,
    ExporterStartupError,
    ExportWorker,
    StepMemoryRecord,
    StepTimeRecord,
    parse_export_streams,
)
from traceml_ai.aggregator.export import wandb_exporter as module
from traceml_ai.aggregator.export.wandb_exporter import (
    WandbExporter,
    _step_metrics,
)
from traceml_ai.runtime.sender import SenderIdentity
from traceml_ai.telemetry.envelope import build_telemetry_envelope

_SRC_ROOT = str(Path(traceml_ai.__file__).resolve().parents[1])
_KEY = "k" * 40
_IDENTITY_ENV = (
    "WANDB_RUN_ID",
    "WANDB_RESUME",
    "WANDB_RESUME_FROM",
    "WANDB_FORK_FROM",
    "WANDB_SWEEP_ID",
    "WANDB_SWEEP_PARAM_PATH",
    "WANDB_LAUNCH",
    "WANDB_LAUNCH_CONFIG_PATH",
    "WANDB_RUN_GROUP",
    "WANDB_JOB_TYPE",
    "WANDB_TAGS",
    "WANDB_NOTES",
    "WANDB_CONFIG_PATHS",
    "WANDB_SERVICE",
)


# -- the stand-in wandb ------------------------------------------------------


class UsageError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class CommError(Exception):
    pass


class FakeSettings:
    """Records its fields; rejects the names in ``reject``."""

    reject: set = set()

    def __init__(self, **kwargs):
        object.__setattr__(self, "fields", {})
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __setattr__(self, name, value):
        if name in self.reject:
            raise ValueError(f"unknown setting {name}")
        self.fields[name] = value

    def __getattr__(self, name):
        try:
            return self.__dict__["fields"][name]
        except KeyError:
            raise AttributeError(name) from None


class FakeRun:
    def __init__(self, fake, kwargs):
        self._fake = fake
        self.kwargs = kwargs
        self.id = fake.id_override or kwargs["id"]
        self.name = kwargs["name"]
        self.url = f"https://wandb.test/e/p/runs/{self.id}"
        self.project_url = "https://wandb.test/e/p"
        self.summary = {}
        self.defined = []
        self.logged = []
        self.finished = []

    def define_metric(self, name, **kwargs):
        self.defined.append((name, kwargs))

    def log(self, data, **kwargs):
        assert "step" not in kwargs
        if self._fake.log_error is not None:
            raise self._fake.log_error
        self.logged.append((dict(data), kwargs))

    def finish(self, exit_code=None):
        self.finished.append(exit_code)
        if self._fake.finish_block is not None:
            self._fake.finish_block.wait(30)
        if self._fake.finish_error is not None:
            raise self._fake.finish_error


def _login(key=None, relogin=None, prompt=True, verify=True):
    raise AssertionError("replaced per fake")


def _make_fake():
    fake = types.ModuleType("wandb")
    fake.__version__ = "0.31.0"
    fake.calls = []
    fake.runs = []
    fake.login_result = True
    fake.login_error = None
    fake.init_error = None
    fake.init_error_for = None
    fake.init_block = None
    fake.id_override = None
    fake.log_error = None
    fake.finish_error = None
    fake.finish_block = None
    fake.init_delay = {}
    fake.resolved_mode = "online"

    class Settings(FakeSettings):
        reject = set()

    fake.Settings = Settings
    fake.errors = types.SimpleNamespace(
        UsageError=UsageError,
        AuthenticationError=AuthenticationError,
        CommError=CommError,
    )
    fake.sdk = types.SimpleNamespace(
        wandb_setup=types.SimpleNamespace(_singleton=None)
    )

    def generate_id(*args, **kwargs):
        raise AssertionError("TraceML must generate the run id itself")

    fake.util = types.SimpleNamespace(generate_id=generate_id)

    def setup(settings=None):
        fake.calls.append(("setup", settings))
        return types.SimpleNamespace(
            settings=types.SimpleNamespace(mode=fake.resolved_mode)
        )

    def login(key=None, relogin=None, prompt=True, verify=True):
        fake.calls.append(("login", {"prompt": prompt, "verify": verify}))
        if fake.login_error is not None:
            raise fake.login_error
        return fake.login_result

    def init(**kwargs):
        fake.calls.append(("init", kwargs))
        if fake.init_block is not None:
            fake.init_block.wait(30)
        rank = int(kwargs["name"].rsplit(" ", 1)[1])
        time.sleep(fake.init_delay.get(rank, 0.0))
        if fake.init_error is not None:
            failing = fake.init_error_for
            if failing is None or rank in failing:
                raise fake.init_error
        run = FakeRun(fake, kwargs)
        fake.runs.append(run)
        return run

    fake.setup = setup
    fake.login = login
    fake.init = init
    return fake


@pytest.fixture(autouse=True)
def _clean_wandb_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("WANDB_"):
            monkeypatch.delenv(name)
    yield
    # The exporter sets WANDB_ERROR_REPORTING; never leak it.
    for name in [name for name in os.environ if name.startswith("WANDB_")]:
        os.environ.pop(name, None)


@pytest.fixture
def fake(monkeypatch):
    stand_in = _make_fake()
    monkeypatch.setitem(sys.modules, "wandb", stand_in)
    return stand_in


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("WANDB_PROJECT", "proj")


def _exporter(tmp_path, *, world=2, startup=10.0, finish=1.25, **kwargs):
    return WandbExporter(
        tmp_path / "export",
        run_name="my-run",
        expected_world_size=world,
        startup_timeout_sec=startup,
        finish_timeout_sec=finish,
        **kwargs,
    )


def _record(rank=0, step=1, events=None, **overrides) -> StepTimeRecord:
    fields = {
        "observed_timestamp_unix_ns": 1_800_000_000_000_000_000,
        "rank": rank,
        "global_rank": rank,
        "local_rank": rank,
        "world_size": 2,
        "local_world_size": 2,
        "node_rank": 0,
        "hostname": "host-a",
        "timestamp_unix_ns": 1_700_000_000_000_000_000 + (step or 0),
        "seq": step,
        "step": step,
        "events": (
            {
                "_traceml_internal:forward_time": {
                    "cpu": {"cpu_ms": 2.0, "gpu_ms": None, "duration_ms": 2.0}
                }
            }
            if events is None
            else events
        ),
    }
    fields.update(overrides)
    return StepTimeRecord(**fields)


def _runs_by_rank(fake):
    return {int(run.name.rsplit(" ", 1)[1]): run for run in fake.runs}


def _calls(fake, name):
    return [kwargs for call, kwargs in fake.calls if call == name]


# -- construction and import ------------------------------------------------


def test_importing_the_module_does_not_import_wandb():
    code = (
        "import sys; sys.modules['wandb'] = None; "
        "import traceml_ai.aggregator.export.wandb_exporter as m; "
        "import traceml_ai.aggregator.trace_aggregator; "
        "import traceml_ai.launcher.cli; "
        "m.WandbExporter('x', run_name='r', expected_world_size=1, "
        "startup_timeout_sec=None, finish_timeout_sec=1.0); "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": _SRC_ROOT},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.mark.parametrize(
    "streams",
    [
        ("step_time", "step_memory"),
        ("step_memory",),
        ("system", "system_gpu"),
        None,
    ],
)
def test_constructor_rejects_other_streams(tmp_path, streams):
    with pytest.raises(ValueError, match="step_time stream only"):
        _exporter(tmp_path, streams=streams)


def test_constructor_accepts_the_default_stream_selection(tmp_path):
    _exporter(tmp_path, streams=parse_export_streams(DEFAULT_EXPORT_STREAMS))


# -- open ---------------------------------------------------------------------


def test_missing_wandb_names_the_extra(tmp_path, env, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)

    with pytest.raises(ImportError, match=r"traceml-ai\[wandb\]"):
        _exporter(tmp_path).open()


def test_broken_wandb_names_the_broken_install(tmp_path, env, monkeypatch):
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("cannot import name 'x' from 'pydantic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    monkeypatch.setattr(builtins, "__import__", broken_import)

    with pytest.raises(ImportError, match="installed but failed to import"):
        _exporter(tmp_path).open()


def test_open_refuses_when_wandb_was_already_set_up(tmp_path, env, fake):
    fake.sdk.wandb_setup._singleton = object()

    with pytest.raises(RuntimeError, match="already set up"):
        _exporter(tmp_path).open()

    assert fake.calls == []


@pytest.mark.parametrize(
    "present, missing",
    [
        ({"WANDB_PROJECT": "p"}, "WANDB_ENTITY"),
        ({"WANDB_ENTITY": "e"}, "WANDB_PROJECT"),
        ({}, "WANDB_ENTITY and WANDB_PROJECT"),
        ({"WANDB_ENTITY": " ", "WANDB_PROJECT": "p"}, "WANDB_ENTITY"),
    ],
)
def test_entity_and_project_are_required(
    tmp_path, fake, monkeypatch, present, missing
):
    for name, value in present.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=f"needs {missing} set"):
        _exporter(tmp_path).open()

    assert fake.calls == []


@pytest.mark.parametrize(
    "mode", ["offline", "dryrun", "disabled", "shared", "OFFLINE", "bogus"]
)
def test_non_online_wandb_mode_is_rejected_before_setup(
    tmp_path, env, fake, monkeypatch, mode
):
    monkeypatch.setenv("WANDB_MODE", mode)

    with pytest.raises(ValueError, match=f"WANDB_MODE='{mode}'"):
        _exporter(tmp_path).open()

    assert fake.calls == []


@pytest.mark.parametrize("mode", ["online", " Online ", ""])
def test_online_wandb_mode_is_accepted(tmp_path, env, fake, monkeypatch, mode):
    monkeypatch.setenv("WANDB_MODE", mode)

    _exporter(tmp_path).open()

    assert len(fake.runs) == 2


def test_a_settings_file_mode_is_rejected_before_login(tmp_path, env, fake):
    fake.resolved_mode = "offline"

    with pytest.raises(ValueError, match="wandb/settings"):
        _exporter(tmp_path).open()

    assert [call for call, _ in fake.calls] == ["setup"]


def test_setup_is_silent_online_and_before_login(tmp_path, env, fake):
    _exporter(tmp_path).open()

    order = [call for call, _ in fake.calls]
    assert order[:2] == ["setup", "login"]
    (settings,) = _calls(fake, "setup")
    assert settings.fields["mode"] == "online"
    assert settings.fields["silent"] is True
    assert settings.fields["sagemaker_disable"] is True
    assert settings.fields["x_disable_viewer"] is True
    assert 0 < settings.fields["x_service_wait"] <= 9.0


def test_setup_keeps_the_default_service_wait_outside_telemetry(
    tmp_path, env, fake
):
    _exporter(tmp_path, startup=None).open()

    (settings,) = _calls(fake, "setup")
    assert "x_service_wait" not in settings.fields
    for kwargs in _calls(fake, "init"):
        assert "init_timeout" not in kwargs["settings"].fields


def test_open_clears_the_inherited_run_identity(
    tmp_path, env, fake, monkeypatch
):
    for name in _IDENTITY_ENV:
        monkeypatch.setenv(name, "inherited")

    _exporter(tmp_path).open()

    for name in _IDENTITY_ENV:
        assert name not in os.environ
    for kwargs in _calls(fake, "init"):
        assert kwargs["id"] != "inherited"


def test_open_turns_off_wandb_error_reporting_unless_set(
    tmp_path, env, fake, monkeypatch
):
    _exporter(tmp_path).open()
    assert os.environ["WANDB_ERROR_REPORTING"] == "false"

    os.environ.pop("WANDB_ERROR_REPORTING")
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "true")
    _exporter(tmp_path / "other").open()
    assert os.environ["WANDB_ERROR_REPORTING"] == "true"


def test_login_never_prompts_and_never_passes_a_key(
    tmp_path, env, fake, monkeypatch
):
    monkeypatch.setenv("WANDB_API_KEY", _KEY)

    _exporter(tmp_path).open()

    assert _calls(fake, "login") == [{"prompt": False, "verify": False}]


@pytest.mark.parametrize("missing", ["prompt", "verify"])
def test_login_gets_only_the_parameters_it_has(tmp_path, env, fake, missing):
    seen = []

    def login_without_prompt(key=None, relogin=None, verify=True):
        seen.append({"verify": verify})
        return True

    def login_without_verify(key=None, relogin=None, prompt=True):
        seen.append({"prompt": prompt})
        return True

    fake.login = (
        login_without_prompt if missing == "prompt" else login_without_verify
    )

    _exporter(tmp_path).open()

    expected = {"prompt", "verify"} - {missing}
    assert set(seen[0]) == expected
    assert all(value is False for value in seen[0].values())


def test_no_credentials_fails_with_the_setup_hint(tmp_path, env, fake):
    fake.login_result = False

    with pytest.raises(RuntimeError, match="No W&B credentials found"):
        _exporter(tmp_path).open()

    assert _calls(fake, "init") == []


def test_usage_error_maps_to_the_no_credentials_message(tmp_path, env, fake):
    fake.login_error = UsageError("api_key not configured (no-tty)")

    with pytest.raises(RuntimeError) as exc:
        _exporter(tmp_path).open()

    assert str(exc.value).startswith("[TraceML] No W&B credentials found")
    assert "api_key not configured" in str(exc.value)


def test_authentication_error_is_reported_with_the_prefix(
    tmp_path, env, fake, monkeypatch, capsys
):
    monkeypatch.setenv("WANDB_API_KEY", _KEY)
    fake.login_error = AuthenticationError("API key must be 40 characters")

    with pytest.raises(RuntimeError) as exc:
        _exporter(tmp_path).open()

    message = str(exc.value)
    assert message.startswith("[TraceML] wandb.login() failed")
    assert "AuthenticationError" in message
    captured = capsys.readouterr()
    for text in (message, repr(exc.value), captured.out, captured.err):
        assert _KEY not in text
    for _, kwargs in fake.calls:
        assert _KEY not in repr(kwargs)


def test_init_failure_names_entity_and_project(tmp_path, env, fake):
    fake.init_error = CommError("permission denied")

    with pytest.raises(RuntimeError, match="team/proj.*CommError"):
        _exporter(tmp_path).open()


def test_open_creates_one_grouped_run_per_rank(tmp_path, env, fake, capsys):
    exporter = _exporter(tmp_path, world=3)

    exporter.open()

    runs = _runs_by_rank(fake)
    assert sorted(runs) == [0, 1, 2]
    for rank, run in runs.items():
        kwargs = run.kwargs
        assert kwargs["entity"] == "team"
        assert kwargs["project"] == "proj"
        assert kwargs["group"] == "my-run"
        assert kwargs["job_type"] == "traceml"
        assert kwargs["name"] == f"my-run rank {rank}"
        assert kwargs["resume"] == "never"
        assert kwargs["reinit"] == "create_new"
        assert kwargs["mode"] == "online"
        assert kwargs["save_code"] is False
        assert kwargs["dir"] == str(tmp_path / "export")
        assert re.fullmatch(r"[a-z0-9]{8}", kwargs["id"])
        for absent in ("tags", "config", "step", "notes"):
            assert absent not in kwargs
        assert run.defined == [
            ("traceml/step", {"hidden": True}),
            ("traceml/*", {"step_metric": "traceml/step"}),
        ]
    assert len({run.id for run in runs.values()}) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    group_lines = [
        line
        for line in captured.err.splitlines()
        if line.startswith("[TraceML] W&B group:")
    ]
    assert group_lines == [
        "[TraceML] W&B group: https://wandb.test/e/p/groups/my-run"
    ]
    assert (tmp_path / "export").is_dir()


def test_group_line_falls_back_to_the_run_url(tmp_path, env, fake, capsys):
    exporter = _exporter(tmp_path, world=1)
    original = FakeRun.__init__

    def without_project(self, fake_module, kwargs):
        original(self, fake_module, kwargs)
        self.project_url = None

    FakeRun.__init__ = without_project
    try:
        exporter.open()
    finally:
        FakeRun.__init__ = original

    run = fake.runs[0]
    assert f"[TraceML] W&B group: {run.url}" in capsys.readouterr().err


def test_init_settings_are_exactly_as_designed(tmp_path, env, fake):
    _exporter(tmp_path, world=1, finish=1.25).open()

    (kwargs,) = _calls(fake, "init")
    fields = kwargs["settings"].fields
    stop_fn = fields.pop("stop_fn")
    init_timeout = fields.pop("init_timeout")
    assert 0 < init_timeout <= 9.0
    assert fields == {
        "disable_code": True,
        "disable_git": True,
        "console": "off",
        "sagemaker_disable": True,
        "finish_timeout": 1.25,
        "finish_timeout_raises": True,
        "quiet": True,
        "x_disable_stats": True,
        "x_disable_meta": True,
        "x_save_requirements": False,
        "x_update_finish_state": True,
        "x_skip_transaction_log": False,
    }
    assert callable(stop_fn)


def test_stop_fn_prints_one_line_and_does_not_raise(tmp_path, env, fake):
    _exporter(tmp_path, world=1).open()
    (kwargs,) = _calls(fake, "init")
    stop_fn = kwargs["settings"].fields["stop_fn"]

    assert stop_fn() is None


def test_stop_fn_output(capsys):
    module._ignore_stop_request()

    assert capsys.readouterr().err == (
        "[TraceML] W&B stop request ignored; stop the training run "
        "instead.\n"
    )


def test_a_rejected_x_setting_is_one_note_not_a_failure(
    tmp_path, env, fake, capsys
):
    fake.Settings.reject = {"x_disable_stats"}

    _exporter(tmp_path, world=3).open()

    notes = [
        line
        for line in capsys.readouterr().err.splitlines()
        if "x_disable_stats" in line
    ]
    assert len(notes) == 1
    assert notes[0].startswith("[TraceML] W&B:")
    assert len(fake.runs) == 3


def test_the_run_id_never_comes_from_wandb_util(tmp_path, env, fake):
    _exporter(tmp_path, world=2).open()
    _exporter(tmp_path / "again", world=2).open()

    ids = [kwargs["id"] for kwargs in _calls(fake, "init")]
    assert len(set(ids)) == 4


def test_a_mismatched_run_id_raises_without_finishing_that_run(
    tmp_path, env, fake
):
    fake.id_override = "theirs01"

    with pytest.raises(RuntimeError, match="'theirs01'"):
        _exporter(tmp_path, world=1).open()

    assert fake.runs[0].finished == []


def test_a_failed_rank_finishes_the_runs_already_created(tmp_path, env, fake):
    fake.init_error = CommError("bad project")
    fake.init_error_for = {1}

    with pytest.raises(RuntimeError, match="rank 1"):
        _exporter(tmp_path, world=3).open()

    for run in fake.runs:
        assert run.finished == [1]


def test_blocked_init_fails_within_the_budget(tmp_path, env, fake):
    fake.init_block = threading.Event()
    exporter = _exporter(tmp_path, world=2, startup=1.5)

    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="wandb.init"):
            exporter.open()
        assert time.monotonic() - started < 1.5
    finally:
        fake.init_block.set()


def test_a_spent_budget_names_the_phase(tmp_path, env, fake):
    with pytest.raises(TimeoutError, match="before wandb.setup"):
        _exporter(tmp_path, startup=0.5).open()

    assert fake.calls == []


def test_an_unwritable_directory_fails(tmp_path, env, fake, monkeypatch):
    monkeypatch.setattr(module.os, "access", lambda path, mode: False)

    with pytest.raises(PermissionError, match="not writable"):
        _exporter(tmp_path).open()

    assert _calls(fake, "init") == []


def test_open_twice_raises(tmp_path, env, fake):
    exporter = _exporter(tmp_path)
    exporter.open()

    with pytest.raises(RuntimeError):
        exporter.open()


def test_open_on_a_worker_thread_installs_no_signal_handler(
    tmp_path, env, fake, monkeypatch
):
    installed = []
    monkeypatch.setattr(signal, "signal", lambda *args: installed.append(args))
    errors = []

    def run():
        try:
            _exporter(tmp_path).open()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(10)

    assert errors == []
    assert installed == []
    assert len(fake.runs) == 2


# -- mapping ------------------------------------------------------------------


def _events(**phases):
    names = {
        "input_wait": "_traceml_internal:dataloader_next",
        "h2d": "_traceml_internal:h2d_time",
        "forward": "_traceml_internal:forward_time",
        "backward": "_traceml_internal:backward_time",
        "optimizer_step": "_traceml_internal:optimizer_step",
        "traced_step_time": "_traceml_internal:step_time",
    }
    return {names[phase]: devices for phase, devices in phases.items()}


def test_mapping_uses_canonical_names_without_the_rank():
    events = _events(
        forward={"cpu": {"cpu_ms": 2.0, "gpu_ms": None, "duration_ms": 2.0}},
        backward={"cpu": {"cpu_ms": 3.0, "gpu_ms": None, "duration_ms": 3.0}},
    )
    events["custom_region"] = {"cpu": {"cpu_ms": 9.0}}

    rank, metrics = _step_metrics(_record(rank=1, step=5, events=events))

    assert rank == 1
    assert metrics == {
        "traceml/step": 5,
        "traceml/timestamp_unix_ns": 1_700_000_000_000_000_005,
        "traceml/forward_cpu_ms": 2.0,
        "traceml/backward_cpu_ms": 3.0,
    }
    for key in metrics:
        assert "_traceml_internal" not in key
        assert "duration" not in key
        assert "rank" not in key
        assert "." not in key


def test_mapping_keeps_both_clocks_and_sums_devices():
    events = _events(
        forward={
            "cuda:0": {"cpu_ms": 1.0, "gpu_ms": 4.0},
            "cuda:1": {"cpu_ms": 2.0, "gpu_ms": 5.0},
        },
        optimizer_step={"cuda:0": {"cpu_ms": 0.0, "gpu_ms": 0.0}},
    )

    _, metrics = _step_metrics(_record(events=events))

    assert metrics["traceml/forward_cpu_ms"] == 3.0
    assert metrics["traceml/forward_gpu_ms"] == 9.0
    assert metrics["traceml/optimizer_step_cpu_ms"] == 0.0
    assert metrics["traceml/optimizer_step_gpu_ms"] == 0.0


def test_mapping_omits_missing_clocks_and_phases():
    events = _events(
        forward={"cuda:0": {"cpu_ms": 1.0, "gpu_ms": None}},
        backward={"cuda:0": {"cpu_ms": float("nan"), "gpu_ms": float("inf")}},
    )

    _, metrics = _step_metrics(_record(events=events, timestamp_unix_ns=None))

    assert metrics == {"traceml/step": 1, "traceml/forward_cpu_ms": 1.0}


@pytest.mark.parametrize(
    "overrides",
    [
        {"global_rank": None},
        {"global_rank": -1},
        {"step": None},
        {"step": -3},
    ],
)
def test_mapping_skips_records_without_rank_or_step(overrides):
    assert _step_metrics(_record(**overrides)) is None


# -- write and close ----------------------------------------------------------


def test_write_logs_one_committed_row_per_record_into_its_rank(
    tmp_path, env, fake
):
    exporter = _exporter(tmp_path)
    exporter.open()

    exporter.write(
        [
            _record(rank=0, step=1),
            _record(rank=1, step=1),
            _record(rank=0, step=2),
            _record(rank=0, step=2),
        ]
    )
    exporter.flush()

    runs = _runs_by_rank(fake)
    assert [row["traceml/step"] for row, _ in runs[0].logged] == [1, 2, 2]
    assert [row["traceml/step"] for row, _ in runs[1].logged] == [1]
    for run in runs.values():
        assert all(kwargs == {"commit": True} for _, kwargs in run.logged)
        assert len(run.defined) == 2


def test_write_skips_and_counts_records_without_rank_or_step(
    tmp_path, env, fake, capsys
):
    exporter = _exporter(tmp_path, world=1)
    exporter.open()

    exporter.write(
        [_record(global_rank=None), _record(step=None), _record(step=1)]
    )
    exporter.close(dropped_batches=3)

    run = fake.runs[0]
    assert len(run.logged) == 1
    assert run.summary == {
        "traceml/export_written_records": 1,
        "traceml/export_dropped_batches": 3,
        "traceml/export_skipped_records": 2,
    }
    assert run.finished == [0]
    assert "skipped 2 step-time records" in capsys.readouterr().err


def test_a_rank_beyond_the_world_size_gets_one_extra_run(
    tmp_path, env, fake, capsys
):
    exporter = _exporter(tmp_path, world=2, startup=10.0)
    exporter.open()
    init_calls = len(_calls(fake, "init"))

    exporter.write(
        [
            _record(rank=2, step=1, world_size=3),
            _record(rank=2, step=2, world_size=3),
        ]
    )

    assert len(_calls(fake, "init")) == init_calls + 1
    runs = _runs_by_rank(fake)
    assert sorted(runs) == [0, 1, 2]
    assert len(runs[2].logged) == 2
    assert len(runs[2].defined) == 2
    assert runs[2].kwargs["settings"].fields["init_timeout"] == 10.0
    warnings = [
        line
        for line in capsys.readouterr().err.splitlines()
        if "rank 2" in line
    ]
    assert len(warnings) == 1


def test_lazy_run_keeps_the_wandb_default_timeout_outside_telemetry(
    tmp_path, env, fake
):
    exporter = _exporter(tmp_path, world=1, startup=None)
    exporter.open()

    exporter.write([_record(rank=1, step=1, world_size=2)])

    runs = _runs_by_rank(fake)
    assert "init_timeout" not in runs[1].kwargs["settings"].fields


def test_a_huge_rank_is_skipped_and_counted_without_init(
    tmp_path, env, fake, capsys
):
    exporter = _exporter(tmp_path, world=1)
    exporter.open()
    init_calls = len(_calls(fake, "init"))

    exporter.write(
        [
            _record(rank=10**6, step=1),
            _record(rank=10**6, step=2, world_size=None),
            _record(rank=0, step=1),
        ]
    )
    exporter.close()

    assert len(_calls(fake, "init")) == init_calls
    run = fake.runs[0]
    assert len(run.logged) == 1
    assert run.summary["traceml/export_skipped_records"] == 2
    refusals = [
        line
        for line in capsys.readouterr().err.splitlines()
        if "skipping records of rank" in line
    ]
    assert len(refusals) == 1


def test_an_unexpected_rank_beyond_the_run_cap_is_skipped(
    tmp_path, env, fake, monkeypatch
):
    monkeypatch.setattr(module, "_MAX_RUNS", 3)
    exporter = _exporter(tmp_path, world=1)
    exporter.open()

    exporter.write(
        [
            _record(rank=1, step=1, world_size=10),
            _record(rank=2, step=1, world_size=10),
            _record(rank=3, step=1, world_size=10),
        ]
    )
    exporter.close()

    runs = _runs_by_rank(fake)
    assert sorted(runs) == [0, 1, 2]
    assert runs[0].summary["traceml/export_skipped_records"] == 1


def test_a_failed_lazy_run_marks_the_export_failed(tmp_path, env, fake):
    exporter = _exporter(tmp_path, world=1)
    exporter.open()
    exporter.write([_record(rank=0)])
    fake.init_error = CommError("gone")

    with pytest.raises(RuntimeError, match="rank 1"):
        exporter.write([_record(rank=1, world_size=2)])
    exporter.close()

    assert fake.runs[0].finished == [1]


def test_write_rejects_other_records_before_logging(tmp_path, env, fake):
    exporter = _exporter(tmp_path, world=1)
    exporter.open()
    memory = StepMemoryRecord(
        observed_timestamp_unix_ns=1,
        rank=0,
        global_rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        node_rank=0,
        hostname=None,
        timestamp_unix_ns=1,
        seq=1,
        device="cpu",
        step=1,
        peak_alloc_bytes=None,
        peak_reserved_bytes=None,
    )

    with pytest.raises(ValueError, match="step_memory"):
        exporter.write([_record(), memory])

    assert fake.runs[0].logged == []


def test_write_before_open_raises(tmp_path, fake):
    with pytest.raises(RuntimeError):
        _exporter(tmp_path).write([_record()])


def test_a_failed_write_propagates_and_close_reports_failure(
    tmp_path, env, fake
):
    exporter = _exporter(tmp_path)
    exporter.open()
    exporter.write([_record(rank=0), _record(rank=1)])
    fake.log_error = CommError("gone")

    with pytest.raises(CommError):
        exporter.write([_record(rank=0, step=2)])
    exporter.close()

    for run in fake.runs:
        assert run.finished == [1]


def test_close_fails_a_run_that_received_no_row(tmp_path, env, fake):
    exporter = _exporter(tmp_path)
    exporter.open()
    exporter.write([_record(rank=0)])

    exporter.close()

    runs = _runs_by_rank(fake)
    assert runs[0].finished == [0]
    assert runs[1].finished == [1]
    assert runs[1].summary["traceml/export_written_records"] == 0


def test_close_is_one_shot(tmp_path, env, fake):
    exporter = _exporter(tmp_path, world=1)
    exporter.open()
    exporter.close()
    exporter.close()

    assert fake.runs[0].finished == [1]


def test_close_reraises_a_finish_timeout_after_the_summary(
    tmp_path, env, fake
):
    exporter = _exporter(tmp_path)
    exporter.open()
    exporter.write([_record(rank=0), _record(rank=1)])
    fake.finish_error = TimeoutError("finish timed out")

    with pytest.raises(TimeoutError):
        exporter.close()

    for run in fake.runs:
        assert run.finished == [0]
        assert run.summary["traceml/export_written_records"] == 1


def test_close_finishes_runs_in_parallel(tmp_path, env, fake):
    exporter = _exporter(tmp_path, world=4, finish=1.25)
    exporter.open()
    exporter.write([_record(rank=rank) for rank in range(4)])
    original = FakeRun.finish

    def slow_finish(self, exit_code=None):
        time.sleep(0.5)
        original(self, exit_code=exit_code)

    FakeRun.finish = slow_finish
    try:
        started = time.monotonic()
        exporter.close()
        elapsed = time.monotonic() - started
    finally:
        FakeRun.finish = original

    assert elapsed < 1.5
    assert all(run.finished == [0] for run in fake.runs)


def test_close_bounds_a_finish_that_never_returns(tmp_path, env, fake):
    exporter = _exporter(tmp_path, world=1, finish=0.2)
    exporter.open()
    exporter.write([_record()])
    release = threading.Event()
    fake.runs[0].finish = lambda exit_code=None: release.wait(30)

    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="did not return within 0.7"):
            exporter.close()
        assert time.monotonic() - started < 2.0
    finally:
        release.set()


# -- with the real export worker ----------------------------------------------


def _envelope(rank: int, step: int) -> dict:
    return build_telemetry_envelope(
        identity=SenderIdentity(
            global_rank=rank, local_rank=rank, hostname="host-a", pid=42
        ),
        sampler_name="StepTimeSampler",
        tables={
            "StepTimeTable": [
                {
                    "seq": step,
                    "timestamp_unix_ns": 1_700_000_000_000_000_000,
                    "step": step,
                    "events": {
                        "_traceml_internal:forward_time": {
                            "cpu": {
                                "cpu_ms": 1.5,
                                "gpu_ms": None,
                                "duration_ms": 1.5,
                                "is_gpu": False,
                                "n_calls": 1,
                            }
                        }
                    },
                }
            ]
        },
    )


def _worker(exporter):
    return ExportWorker(exporter, interval_sec=0.05, streams=("step_time",))


def test_worker_reports_an_open_failure_as_a_startup_error(
    tmp_path, env, fake
):
    fake.login_result = False
    worker = _worker(_exporter(tmp_path))
    worker.start()
    try:
        with pytest.raises(ExporterStartupError, match="No W&B credentials"):
            worker.wait_until_ready(5.0)
    finally:
        worker.stop(1.0)


def test_worker_times_out_a_blocked_init(tmp_path, env, fake):
    fake.init_block = threading.Event()
    worker = _worker(_exporter(tmp_path, startup=10.0))
    worker.start()
    try:
        with pytest.raises(ExporterStartupError, match="within 0.5 seconds"):
            worker.wait_until_ready(0.5)
    finally:
        fake.init_block.set()
        worker.stop(1.0)


def test_worker_exports_a_real_envelope_and_finishes_the_runs(
    tmp_path, env, fake
):
    worker = _worker(_exporter(tmp_path))
    worker.start()
    worker.wait_until_ready(5.0)
    worker.submit(_envelope(0, 1))
    worker.submit(_envelope(1, 1))
    worker.submit(_envelope(0, 2))

    stats = worker.stop(3.0)

    assert stats.written_records == 3
    assert stats.exporter_failed is False
    runs = _runs_by_rank(fake)
    assert [row for row, _ in runs[0].logged] == [
        {
            "traceml/step": 1,
            "traceml/timestamp_unix_ns": 1_700_000_000_000_000_000,
            "traceml/forward_cpu_ms": 1.5,
        },
        {
            "traceml/step": 2,
            "traceml/timestamp_unix_ns": 1_700_000_000_000_000_000,
            "traceml/forward_cpu_ms": 1.5,
        },
    ]
    assert runs[0].finished == [0]
    assert runs[1].finished == [0]


# -- review follow-ups -------------------------------------------------------


def test_serve_admits_ranks_up_to_the_records_world_size(tmp_path, env, fake):
    # ``traceml serve`` without --nproc-per-node expects one rank.
    exporter = _exporter(tmp_path, world=1)
    exporter.open()

    exporter.write(
        [_record(rank=rank, step=1, world_size=4) for rank in range(4)]
    )
    exporter.write([_record(rank=5000, step=1, world_size=10**6)])
    exporter.close()

    runs = _runs_by_rank(fake)
    assert sorted(runs) == [0, 1, 2, 3]
    assert all(len(run.logged) == 1 for run in runs.values())
    assert runs[0].summary["traceml/export_skipped_records"] == 1


def test_a_rejected_record_marks_the_write_failed(tmp_path, env, fake):
    exporter = _exporter(tmp_path, world=1)
    exporter.open()
    exporter.write([_record(rank=0)])
    memory = StepMemoryRecord(
        observed_timestamp_unix_ns=1,
        rank=0,
        global_rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        node_rank=0,
        hostname=None,
        timestamp_unix_ns=1,
        seq=1,
        device="cpu",
        step=2,
        peak_alloc_bytes=None,
        peak_reserved_bytes=None,
    )

    with pytest.raises(ValueError):
        exporter.write([_record(rank=0, step=2), memory])
    exporter.close()

    assert fake.runs[0].finished == [1]


def test_a_failed_startup_reports_its_cause_within_the_cleanup_bound(
    tmp_path, env, fake
):
    fake.init_error = CommError("bad project")
    fake.init_error_for = {1}
    fake.init_delay = {1: 0.2}
    fake.finish_block = threading.Event()
    exporter = _exporter(tmp_path, world=2, startup=10.0, finish=30.0)

    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="rank 1: CommError"):
            exporter.open()
        elapsed = time.monotonic() - started
    finally:
        fake.finish_block.set()

    assert elapsed < 1.6
    (created,) = fake.runs
    assert created.finished == [1]
