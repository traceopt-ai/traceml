# Renderers

Transforms stored telemetry into human-readable output: Rich panels for the CLI, Plotly charts for the web dashboard. Each renderer is read-only against the `RemoteDBStore` and produces either a `Rich` renderable or a NiceGUI component.

::: traceml.renderers
