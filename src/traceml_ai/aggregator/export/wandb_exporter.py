# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Exporter that streams step time into Weights & Biases, one run per rank.

The aggregator is TraceML's own process (a launcher child, ``traceml
serve`` or the Ray aggregator actor), a sibling of training. ``open``
creates one new W&B run per expected rank, grouped under the TraceML run
name, and fails before training starts when W&B cannot be reached or no
credentials are configured. ``write`` logs one history row per step-time
record into its rank's run, keyed without the rank and plotted against a
hidden per-run ``traceml/step`` axis. ``close`` records the export counts
in each run's summary and finishes every run within a bounded time.

``wandb`` is imported inside ``open`` only, so importing this module, or
the aggregator, never imports it.
"""

from __future__ import annotations

import inspect
import os
import secrets
import string
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from traceml_ai.aggregator.export.records import (
    Record,
    StepTimeRecord,
    selected_streams,
)
from traceml_ai.step_time.model import STEP_TIME_EVENT_NAMES
from traceml_ai.step_time.sqlite import normalize_step_time_events

_PREFIX = "traceml"
_STEP_KEY = f"{_PREFIX}/step"
_JOB_TYPE = "traceml"

# ``WANDB_MODE`` values that leave the exporter online. An allowlist, so
# an unknown or future mode is rejected rather than guessed at.
_ALLOWED_MODES = frozenset({"", "online"})

# Variables that would attach the aggregator's runs to another run, a
# sweep, a launch job or another process's wandb-core, or set optional run
# metadata. The aggregator is TraceML's own process, a sibling of
# training, so removing them here never changes the training process's
# W&B run. ``WANDB_DIR`` and ``WANDB_NAME`` need no removal: the explicit
# ``dir=`` and ``name=`` arguments win over them.
_INHERITED_RUN_ENV = (
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

# Time kept back from the startup budget so a failure is reported before
# the aggregator's own wait for ``open`` runs out.
_STARTUP_MARGIN_SEC = 1.0
# A rank beyond the expected world size gets a run only while the exporter
# holds fewer runs than this many times the expected world size, or than
# the record's own world size if larger, and never more than _MAX_RUNS.
_LAZY_RUN_CAP_FACTOR = 2
_MAX_RUNS = 1024
# How long a failed startup waits for the runs it created to finish, so the
# real cause is reported before the aggregator's wait for open runs out.
_CLEANUP_FINISH_TIMEOUT_SEC = 1.0
# Extra time ``open`` waits past ``init_timeout`` so that wandb reports its
# own init failure; it stays inside the startup margin.
_INIT_JOIN_GRACE_SEC = 0.5
# Extra time the closing thread waits on each run's bounded ``finish``.
_FINISH_JOIN_GRACE_SEC = 0.5

_RUN_ID_ALPHABET = string.ascii_lowercase + string.digits
_RUN_ID_LENGTH = 8

_INSTALL_HINT = 'pip install "traceml-ai[wandb]"'
_NO_CREDENTIALS = (
    "[TraceML] No W&B credentials found. Run wandb login on the host that "
    "runs the aggregator, or set WANDB_API_KEY or WANDB_IDENTITY_TOKEN_FILE "
    "in the aggregator's environment (the launcher's shell, the serve "
    "shell, or Ray's runtime_env)."
)
_STOP_IGNORED = (
    "[TraceML] W&B stop request ignored; stop the training run instead."
)


def _warn(message: str) -> None:
    try:
        print(f"[TraceML] W&B: {message}", file=sys.stderr)
    except Exception:
        pass


def _ignore_stop_request() -> None:
    """``stop_fn`` for every run: the W&B UI Stop button must not SIGINT
    the aggregator, so it prints one line and does nothing else."""
    try:
        print(_STOP_IGNORED, file=sys.stderr)
    except Exception:
        pass


def _new_run_id() -> str:
    """Eight lowercase base-36 characters, the shape W&B generates.

    TraceML generates the id itself because ``wandb.util.generate_id`` was
    removed in wandb 0.28.2.
    """
    return "".join(
        secrets.choice(_RUN_ID_ALPHABET) for _ in range(_RUN_ID_LENGTH)
    )


def _step_metrics(record: StepTimeRecord) -> Optional[Tuple[int, Dict]]:
    """
    Return ``(rank, metrics)`` for one step-time record, or None.

    The rank is ``global_rank``. A missing or negative rank, or a missing
    or negative step, gives None: the record cannot be placed on a run's
    step axis. Keys carry no rank, because each rank has its own run:
    ``traceml/step``, ``traceml/timestamp_unix_ns`` (the training process's
    clock, when the record has it) and ``traceml/<phase>_cpu_ms`` and
    ``traceml/<phase>_gpu_ms`` for each canonical phase. The phase clocks
    come from ``normalize_step_time_events``, which owns device summation
    and drops invalid durations. A phase the step did not record is left
    out, a clock that was not measured is left out, and a measured zero is
    kept. Custom regions are not exported.
    """
    rank = record.global_rank
    step = record.step
    if rank is None or rank < 0 or step is None or step < 0:
        return None
    metrics: Dict[str, Union[int, float]] = {_STEP_KEY: step}
    if record.timestamp_unix_ns is not None:
        metrics[f"{_PREFIX}/timestamp_unix_ns"] = record.timestamp_unix_ns
    clocks = normalize_step_time_events(record.events) or {}
    for phase in STEP_TIME_EVENT_NAMES:
        values = clocks.get(phase)
        if values is None:
            continue
        if values.cpu_ms is not None:
            metrics[f"{_PREFIX}/{phase}_cpu_ms"] = values.cpu_ms
        if values.gpu_ms is not None:
            metrics[f"{_PREFIX}/{phase}_gpu_ms"] = values.gpu_ms
    return rank, metrics


def _import_wandb() -> Any:
    try:
        import wandb
    except ImportError as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name == "wandb":
            raise ImportError(
                "[TraceML] The wandb exporter requires wandb. Install it "
                f"with {_INSTALL_HINT}."
            ) from exc
        raise ImportError(
            f"[TraceML] wandb is installed but failed to import: {exc}"
        ) from exc
    return wandb


def _already_set_up(wandb: Any) -> bool:
    """True when wandb was set up in this process before ``open``."""
    try:
        setup_module = wandb.sdk.wandb_setup
    except AttributeError:
        return False
    return getattr(setup_module, "_singleton", None) is not None


def _login_parameters(login: Any) -> Set[str]:
    """Names ``wandb.login`` accepts; an unreadable signature gives none."""
    try:
        return set(inspect.signature(login).parameters)
    except (TypeError, ValueError):
        return set()


def _group_url(run: Any, group: str) -> Optional[str]:
    """The group page under the run's project, else the run's own page."""
    try:
        project_url = getattr(run, "project_url", None)
        if project_url is None:
            getter = getattr(run, "get_project_url", None)
            project_url = getter() if callable(getter) else None
        if isinstance(project_url, str) and project_url:
            quoted = urllib.parse.quote(group, safe="")
            return f"{project_url.rstrip('/')}/groups/{quoted}"
    except Exception:
        pass
    try:
        return getattr(run, "url", None)
    except Exception:
        return None


class WandbExporter:
    """
    Stream step-time records into one W&B run per rank.

    ``run_name`` is the TraceML run name: it becomes the W&B group, and
    rank ``r``'s run is named ``"<run_name> rank <r>"``.
    ``expected_world_size`` runs are created in ``open``; a record from a
    higher rank below its own ``world_size`` creates that rank's run on
    first sight, up to twice ``expected_world_size`` runs in total.
    ``streams`` must select ``step_time`` alone. ``startup_timeout_sec``
    bounds ``open`` and each later run creation; None leaves wandb's own
    timeouts in place, for modes where nothing waits on ``open``.
    ``finish_timeout_sec`` bounds each run's ``finish``.

    ``WANDB_ENTITY`` and ``WANDB_PROJECT`` must be set. Credentials come
    from wandb's own configuration (``wandb login``, ``WANDB_API_KEY`` or
    ``WANDB_IDENTITY_TOKEN_FILE``); TraceML never reads or passes a key.
    """

    def __init__(
        self,
        directory: Union[str, os.PathLike],
        *,
        run_name: str,
        expected_world_size: int,
        streams: Optional[Iterable[str]] = ("step_time",),
        startup_timeout_sec: Optional[float],
        finish_timeout_sec: float,
    ) -> None:
        selected = selected_streams(streams)
        if selected != frozenset({StepTimeRecord.STREAM}):
            chosen = "all" if selected is None else sorted(selected)
            raise ValueError(
                "[TraceML] The wandb exporter exports the step_time stream "
                f"only; export_streams selects {chosen}. Leave "
                "export_streams at its default, or use the jsonl exporter "
                "for other streams."
            )
        self._directory = Path(directory)
        self._run_name = str(run_name)
        self._expected_world_size = max(1, int(expected_world_size))
        self._startup_timeout_sec = (
            None if startup_timeout_sec is None else float(startup_timeout_sec)
        )
        self._finish_timeout_sec = float(finish_timeout_sec)

        self._wandb: Any = None
        self._entity: Optional[str] = None
        self._project: Optional[str] = None
        self._runs: Dict[int, Any] = {}
        self._rows: Dict[int, int] = {}
        self._settings_notes: Set[str] = set()
        self._opened = False
        self._closed = False
        self._skipped_records = 0
        self._refused_rank_warned = False
        self._write_failed = False

    # -- open -------------------------------------------------------------

    def open(self) -> None:
        """
        Set wandb up without prompting, then create one run per rank.

        Raises when export cannot start: wandb missing or already set up in
        this process, ``WANDB_ENTITY`` or ``WANDB_PROJECT`` unset, a
        ``WANDB_MODE`` other than online, no usable credentials, an
        unwritable directory, a run that could not be created, or a spent
        startup budget. Runs created before the failure are finished as
        failed. Raises ``RuntimeError`` when called twice or after
        ``close``.
        """
        if self._closed or self._opened:
            raise RuntimeError("WandbExporter cannot be opened twice")
        deadline = (
            None
            if self._startup_timeout_sec is None
            else time.monotonic()
            + self._startup_timeout_sec
            - _STARTUP_MARGIN_SEC
        )
        # wandb snapshots the environment at its first settings load, so
        # the inherited run identity is removed before that point.
        for name in _INHERITED_RUN_ENV:
            os.environ.pop(name, None)
        # wandb reports its own errors to W&B unless told not to; the
        # user's explicit choice wins. The training process is untouched.
        os.environ.setdefault("WANDB_ERROR_REPORTING", "false")

        wandb = _import_wandb()
        if _already_set_up(wandb):
            raise RuntimeError(
                "[TraceML] wandb was already set up in the aggregator "
                "process, so inherited run settings cannot be cleared; "
                "the wandb exporter needs a fresh process."
            )
        entity, project = self._validate_environment()

        self._check_deadline(deadline, "wandb.setup()")
        setup_private: Dict[str, Any] = {"x_disable_viewer": True}
        if deadline is not None:
            setup_private["x_service_wait"] = self._remaining(
                deadline, "wandb.setup()"
            )
        library = wandb.setup(
            settings=self._settings(
                wandb,
                {"mode": "online", "silent": True, "sagemaker_disable": True},
                setup_private,
            )
        )
        self._check_resolved_mode(library)

        self._check_deadline(deadline, "wandb.login()")
        self._login(wandb)

        self._directory.mkdir(parents=True, exist_ok=True)
        if not os.access(self._directory, os.R_OK | os.W_OK):
            raise PermissionError(
                f"[TraceML] W&B run directory {self._directory} is not "
                "writable"
            )

        init_timeout = None
        if deadline is not None:
            init_timeout = self._remaining(deadline, "wandb.init()")
        self._wandb = wandb
        self._entity = entity
        self._project = project
        self._create_runs(range(self._expected_world_size), init_timeout)
        self._opened = True

        url = _group_url(self._runs[min(self._runs)], self._run_name)
        try:
            print(f"[TraceML] W&B group: {url}", file=sys.stderr)
        except Exception:
            pass

    def _validate_environment(self) -> Tuple[str, str]:
        entity = os.environ.get("WANDB_ENTITY", "").strip()
        project = os.environ.get("WANDB_PROJECT", "").strip()
        missing = [
            name
            for name, value in (
                ("WANDB_ENTITY", entity),
                ("WANDB_PROJECT", project),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "[TraceML] The wandb exporter needs "
                f"{' and '.join(missing)} set in the aggregator's "
                "environment."
            )
        mode = os.environ.get("WANDB_MODE", "").strip().lower()
        if mode not in _ALLOWED_MODES:
            raise ValueError(
                f"[TraceML] WANDB_MODE={os.environ['WANDB_MODE']!r} is not "
                "supported by the wandb exporter, which logs online only. "
                "Unset WANDB_MODE, or use the jsonl exporter on hosts "
                "without W&B access."
            )
        return entity, project

    @staticmethod
    def _check_resolved_mode(library: Any) -> None:
        try:
            mode = library.settings.mode
        except Exception:
            return
        if mode != "online":
            raise ValueError(
                f"[TraceML] wandb resolved mode {mode!r} after setup; a "
                "wandb settings file (<cwd>/wandb/settings or "
                "~/.config/wandb/settings) may set it. The wandb exporter "
                "logs online only."
            )

    def _login(self, wandb: Any) -> None:
        parameters = _login_parameters(wandb.login)
        kwargs: Dict[str, Any] = {}
        if "prompt" in parameters:
            kwargs["prompt"] = False
        if "verify" in parameters:
            kwargs["verify"] = False
        try:
            logged_in = wandb.login(**kwargs)
        except Exception as exc:
            if type(exc).__name__ == "UsageError":
                raise RuntimeError(f"{_NO_CREDENTIALS} wandb: {exc}") from exc
            raise RuntimeError(
                f"[TraceML] wandb.login() failed: {type(exc).__name__}: {exc}"
            ) from exc
        if logged_in is False:
            raise RuntimeError(_NO_CREDENTIALS)

    @staticmethod
    def _check_deadline(deadline: Optional[float], phase: str) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                f"[TraceML] W&B startup budget spent before {phase}"
            )

    @staticmethod
    def _remaining(deadline: float, phase: str) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"[TraceML] W&B startup budget spent before {phase}"
            )
        return remaining

    def _settings(
        self,
        wandb: Any,
        public: Mapping[str, Any],
        private: Mapping[str, Any],
    ) -> Any:
        """
        Build ``wandb.Settings`` from public fields, then add each ``x_``
        field on its own: a renamed internal field becomes one stderr note,
        not a startup failure.
        """
        settings = wandb.Settings(**public)
        for name, value in private.items():
            try:
                setattr(settings, name, value)
            except Exception as exc:
                if name not in self._settings_notes:
                    self._settings_notes.add(name)
                    _warn(
                        f"this wandb version does not accept {name} "
                        f"({type(exc).__name__}); continuing without it."
                    )
        return settings

    def _init_settings(self, init_timeout: Optional[float]) -> Any:
        public: Dict[str, Any] = {
            "disable_code": True,
            "disable_git": True,
            "console": "off",
            "sagemaker_disable": True,
            "finish_timeout": self._finish_timeout_sec,
            "finish_timeout_raises": True,
            "quiet": True,
            "stop_fn": _ignore_stop_request,
        }
        if init_timeout is not None:
            public["init_timeout"] = init_timeout
        private = {
            "x_disable_stats": True,
            "x_disable_meta": True,
            "x_save_requirements": False,
            # Pin away WANDB__UPDATE_FINISH_STATE and
            # WANDB__SKIP_TRANSACTION_LOG inherited from the environment.
            "x_update_finish_state": True,
            "x_skip_transaction_log": False,
        }
        return self._settings(self._wandb, public, private)

    def _init_run(self, rank: int, run_id: str, settings: Any) -> Any:
        try:
            return self._wandb.init(
                entity=self._entity,
                project=self._project,
                group=self._run_name,
                job_type=_JOB_TYPE,
                name=f"{self._run_name} rank {rank}",
                id=run_id,
                resume="never",
                reinit="create_new",
                mode="online",
                dir=str(self._directory),
                save_code=False,
                settings=settings,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[TraceML] wandb.init() failed for {self._entity}/"
                f"{self._project} rank {rank}: {type(exc).__name__}: {exc}"
            ) from exc

    def _create_runs(
        self, ranks: Iterable[int], init_timeout: Optional[float]
    ) -> None:
        """
        Create the runs for ``ranks`` in parallel threads.

        Any failure stops the rest: every run created before or after it
        is finished as failed, and the first error is raised. A run whose
        id is not the id TraceML asked for may belong to someone else, so
        it is never finished.
        """
        jobs = [
            (rank, _new_run_id(), self._init_settings(init_timeout))
            for rank in ranks
        ]
        lock = threading.Lock()
        cancel = threading.Event()
        created: Dict[int, Any] = {}
        errors: List[BaseException] = []

        def fail(exc: BaseException) -> None:
            with lock:
                errors.append(exc)
                cancel.set()

        def create(rank: int, run_id: str, settings: Any) -> None:
            if cancel.is_set():
                return
            try:
                run = self._init_run(rank, run_id, settings)
            except BaseException as exc:
                fail(exc)
                return
            if getattr(run, "id", None) != run_id:
                fail(
                    RuntimeError(
                        f"[TraceML] wandb.init() returned run "
                        f"{getattr(run, 'id', None)!r} for rank {rank}, "
                        f"not the new run {run_id!r}; an inherited W&B run "
                        "may be attached. TraceML leaves that run untouched."
                    )
                )
                return
            with lock:
                late = cancel.is_set()
                if not late:
                    created[rank] = run
            if late:
                # ``open`` already gave up, so this run is nobody's.
                try:
                    self._finish_runs(
                        {rank: run},
                        exit_codes={rank: 1},
                        timeout_sec=_CLEANUP_FINISH_TIMEOUT_SEC,
                    )
                except Exception:
                    pass
                return
            try:
                run.define_metric(_STEP_KEY, hidden=True)
                run.define_metric(f"{_PREFIX}/*", step_metric=_STEP_KEY)
            except BaseException as exc:
                fail(exc)

        threads = [
            threading.Thread(
                target=create,
                args=job,
                name=f"TraceML-WandbInit-{job[0]}",
                daemon=True,
            )
            for job in jobs
        ]
        for thread in threads:
            thread.start()
        join_deadline = (
            None
            if init_timeout is None
            else time.monotonic() + init_timeout + _INIT_JOIN_GRACE_SEC
        )
        for thread in threads:
            thread.join(
                timeout=(
                    None
                    if join_deadline is None
                    else max(0.0, join_deadline - time.monotonic())
                )
            )
        with lock:
            if any(thread.is_alive() for thread in threads) and not errors:
                errors.append(
                    TimeoutError(
                        "[TraceML] W&B startup budget spent during "
                        "wandb.init()"
                    )
                )
            if not errors:
                self._runs.update(created)
                for rank in created:
                    self._rows.setdefault(rank, 0)
                return
            # A thread still inside wandb.init sees this and finishes its
            # own run when init returns.
            cancel.set()
            failed = dict(created)
            error = errors[0]
        try:
            self._finish_runs(
                failed,
                exit_codes={rank: 1 for rank in failed},
                timeout_sec=_CLEANUP_FINISH_TIMEOUT_SEC,
            )
        except Exception:
            pass
        raise error

    # -- write ------------------------------------------------------------

    def write(self, records: Sequence[Record]) -> None:
        """
        Log one committed history row per step-time record, in order.

        Every record is mapped before any row is logged; a record that is
        not a ``StepTimeRecord`` raises ``ValueError`` and nothing is
        logged. A record without a usable rank or step is skipped and
        counted. Duplicate steps are logged as they arrive. A rank beyond
        the expected world size gets its own run on first sight, within
        the bounds of ``_may_add_rank``; a rank outside them is skipped and
        counted. A failing call marks the export failed and re-raises.
        """
        if not self._opened or self._closed:
            raise RuntimeError("WandbExporter is not open")
        try:
            mapped: List[Tuple[StepTimeRecord, Optional[Tuple[int, Dict]]]] = (
                []
            )
            for record in records:
                if not isinstance(record, StepTimeRecord):
                    stream = getattr(record, "STREAM", type(record).__name__)
                    raise ValueError(
                        "the wandb exporter accepts step_time records only, "
                        f"not {stream!r}"
                    )
                mapped.append((record, _step_metrics(record)))
            for record, item in mapped:
                if item is None:
                    self._skipped_records += 1
                    continue
                rank, metrics = item
                run = self._runs.get(rank)
                if run is None:
                    if not self._may_add_rank(rank, record.world_size):
                        self._skipped_records += 1
                        continue
                    run = self._add_rank(rank)
                run.log(metrics, commit=True)
                self._rows[rank] += 1
        except BaseException:
            self._write_failed = True
            raise

    def _may_add_rank(self, rank: int, world_size: Optional[int]) -> bool:
        """
        Whether an unseen rank may get a run of its own.

        The rank must be below the record's own ``world_size`` (the
        expected world size when the record has none). The exporter holds
        at most the larger of twice the expected world size and that bound
        in runs, and never more than ``_MAX_RUNS``; a rank at or above that
        cap could never fit in a dense rank range, so it is refused too.
        Under ``traceml serve`` the expected world size is 1, so the
        record's own world size is what admits the other ranks. The first
        refusal prints one stderr line.
        """
        bound = self._expected_world_size if world_size is None else world_size
        cap = min(
            _MAX_RUNS,
            max(self._expected_world_size * _LAZY_RUN_CAP_FACTOR, bound),
        )
        if rank < bound and rank < cap and len(self._runs) < cap:
            return True
        if not self._refused_rank_warned:
            self._refused_rank_warned = True
            _warn(
                f"skipping records of rank {rank}: outside the world size "
                f"{bound} or the limit of {cap} runs."
            )
        return False

    def _add_rank(self, rank: int) -> Any:
        """
        Create the run of a rank beyond the expected world size, with the
        startup budget as its init timeout (wandb's own default outside
        telemetry mode).
        """
        _warn(
            f"rank {rank} is outside the expected world size "
            f"{self._expected_world_size}; creating a run for it."
        )
        self._create_runs([rank], self._startup_timeout_sec)
        return self._runs[rank]

    def flush(self) -> None:
        """
        Do nothing.

        wandb hands logged rows to its service process and has no bounded
        per-call flush, so a written record means one handed to the SDK.
        """

    # -- close ------------------------------------------------------------

    def close(self, *, dropped_batches: int = 0) -> None:
        """
        Record the export counts in every run's summary, then finish every
        run in parallel (one-shot).

        Each summary gets ``traceml/export_written_records`` (rows logged
        to that run), ``traceml/export_dropped_batches`` and
        ``traceml/export_skipped_records`` (worker-wide counts). A run
        finishes with exit code 1 when a write failed or it received no
        row. Each ``finish`` is bounded by ``finish_timeout_sec``; the
        first error, a finish timeout included, is re-raised after every
        run has been tried.
        """
        if self._closed:
            return
        self._closed = True
        if not self._opened:
            return
        error: Optional[BaseException] = None
        for rank, run in self._runs.items():
            try:
                run.summary[f"{_PREFIX}/export_written_records"] = int(
                    self._rows.get(rank, 0)
                )
                run.summary[f"{_PREFIX}/export_dropped_batches"] = int(
                    dropped_batches
                )
                run.summary[f"{_PREFIX}/export_skipped_records"] = int(
                    self._skipped_records
                )
            except Exception as exc:
                error = error or exc
        if self._skipped_records:
            _warn(
                f"skipped {self._skipped_records} step-time records without "
                "a usable rank or step."
            )
        exit_codes = {
            rank: 1 if self._write_failed or not self._rows.get(rank) else 0
            for rank in self._runs
        }
        try:
            self._finish_runs(dict(self._runs), exit_codes=exit_codes)
        except Exception as exc:
            error = error or exc
        if error is not None:
            raise error

    def _finish_runs(
        self,
        runs: Mapping[int, Any],
        *,
        exit_codes: Mapping[int, int],
        timeout_sec: Optional[float] = None,
    ) -> None:
        """
        Finish ``runs`` in parallel; re-raise the first error.

        The wait ends after ``timeout_sec``, or by default after
        ``finish_timeout_sec`` plus a short grace; a run still finishing
        then raises ``TimeoutError`` and keeps finishing on its own thread.
        """
        if not runs:
            return
        bound = (
            self._finish_timeout_sec + _FINISH_JOIN_GRACE_SEC
            if timeout_sec is None
            else float(timeout_sec)
        )
        errors: List[BaseException] = []
        lock = threading.Lock()

        def finish(rank: int, run: Any) -> None:
            try:
                run.finish(exit_code=exit_codes.get(rank, 1))
            except BaseException as exc:
                with lock:
                    errors.append(exc)

        threads = [
            threading.Thread(
                target=finish,
                args=(rank, run),
                name=f"TraceML-WandbFinish-{rank}",
                daemon=True,
            )
            for rank, run in runs.items()
        ]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + bound
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        with lock:
            if errors:
                raise errors[0]
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError(
                f"[TraceML] W&B finish did not return within {bound:g} "
                "seconds"
            )
