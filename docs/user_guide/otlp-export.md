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

The aggregator maps telemetry from its live ingestion stream and hands it to
OpenTelemetry's official Batch LogRecord Processor for background export. It
does not poll or replay SQLite. A slow or unavailable collector cannot block
rank ingestion; the processor queue has a fixed memory bound and drops its
oldest records if that bound is reached.

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
- Missing measurements are omitted. TraceML does not replace them with zero.
- Exporter failures are reported through OpenTelemetry SDK logging and never
  fail training.
