# Export telemetry with OTLP

TraceML can send its existing training telemetry from the aggregator to any
OpenTelemetry Protocol (OTLP) Logs endpoint. Export is optional and additive:
SQLite history, the terminal, dashboard, diagnoses, and summaries continue to
work as before.

## Install

Install TraceML with the optional OpenTelemetry dependencies:

```bash
pip install "traceml-ai[otlp]"
```

## Configure

Set a standard OTLP endpoint before starting TraceML. For OTLP/HTTP:

```bash
export OTEL_SERVICE_NAME=my-training-job
export OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=http://localhost:4318/v1/logs
export OTEL_EXPORTER_OTLP_LOGS_PROTOCOL=http/protobuf

traceml run train.py
```

The general OTLP variables work as well:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
```

For OTLP/gRPC, set the protocol to `grpc` and use the collector's gRPC
endpoint, commonly port `4317`. Standard `OTEL_EXPORTER_OTLP_HEADERS`, TLS,
certificate, and timeout variables are handled by the official OpenTelemetry
exporter.

The launcher copies the current environment to the aggregator, so no training
script changes are required. The same variables work when starting
`traceml serve` directly. If no OTLP endpoint is configured, TraceML does not
create an exporter and its behavior is unchanged.

## Delivery behavior

The aggregator maps telemetry from its live ingestion stream, reduces it into
bounded windows, and hands completed windows to OpenTelemetry's official Batch
LogRecord Processor for background export. It does not poll or replay SQLite.
A slow or unavailable collector cannot block rank ingestion; the processor
queue has a fixed memory bound and drops its oldest records if that bound is
reached.

Windowing affects external export only. Rank telemetry, SQLite history, the
dashboard, terminal output, diagnoses, and summaries keep their existing raw
resolution.

| Record | Default window | Grouping |
|---|---:|---|
| Step timing | 10 steps | Per rank, phase, device, and timing clock |
| Step memory | 10 steps | Per rank and device |
| Process | 10 seconds | Per rank and process |
| System | 10 seconds | Per node observer |
| Runtime context | None | Passed through once |

Each changing numeric measurement is exported as `count`, `sum`, `min`, and
`max`. Consumers calculate the sampled mean as `sum / count`. Missing
measurements do not increment `count` and are never converted to zero. Static
capacity fields, such as total RAM and device memory, retain the latest
observed value in the window.

Two TraceML variables tune external window sizes:

| Variable | Default | Meaning |
|---|---:|---|
| `TRACEML_OTLP_STEP_WINDOW` | `10` | Positive number of steps per timing and memory window |
| `TRACEML_OTLP_TIME_WINDOW_SEC` | `10` | Positive seconds per process and system window |

For example, a large job can use wider windows:

```bash
export TRACEML_OTLP_STEP_WINDOW=100
export TRACEML_OTLP_TIME_WINDOW_SEC=30
```

Setting either value to `1` provides finer external resolution. It does not
change TraceML's internal sampler interval. Invalid values produce a warning
and fall back to the documented defaults. Completed windows are emitted when
the next window begins; partial windows are flushed during clean shutdown.

The following standard Batch LogRecord Processor variables tune the queue:

| Variable | Default | Meaning |
|---|---:|---|
| `OTEL_BLRP_MAX_QUEUE_SIZE` | `4096` | Maximum queued records |
| `OTEL_BLRP_MAX_EXPORT_BATCH_SIZE` | `256` | Maximum records per request |
| `OTEL_BLRP_SCHEDULE_DELAY` | `1000` | Maximum batch delay in milliseconds |

`TRACEML_OTLP_SHUTDOWN_TIMEOUT_SEC` controls TraceML's bounded exporter drain
at shutdown and defaults to `2` seconds. Export shutdown may continue on a
daemon thread after that budget rather than delaying training finalization;
records not delivered before process exit can be lost.

TraceML exports structured OTLP LogRecords, not rendered diagnosis text. See
[Telemetry record reference](../reference/telemetry-records.md) for every field,
unit, timestamp, and availability rule.

## OpenTelemetry Collector example

```yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:

exporters:
  debug:
    verbosity: detailed

service:
  pipelines:
    logs:
      receivers: [otlp]
      exporters: [debug]
```

Run the collector with this configuration, configure the endpoint as shown
above, and start training normally.

## Scope and limitations

- Export is live-only; SQLite replay is not implemented.
- OTLP is emitted by the TraceML aggregator, not independently by every rank.
- Step, process, and system OTLP records contain window statistics rather than
  individual source samples.
- Missing measurements are omitted. TraceML does not replace them with zero.
- Exporter failures are reported through OpenTelemetry SDK logging and never
  fail training.
