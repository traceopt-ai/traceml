"""TraceML launcher internals.

The public console entrypoint remains :mod:`traceml.cli`.  This package keeps
the launcher's implementation split by responsibility so process lifecycle,
manifest persistence, command handlers, and argparse wiring can be tested in
isolation.
"""

__all__: list[str] = []
