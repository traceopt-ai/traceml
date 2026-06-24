"""
Fail-loud capability assertion for TraceML integrations.

When an integration becomes active but the runtime's patch configuration will
NOT capture the telemetry streams the integration owes, emit a clear warning
naming the dark streams — so silent absence becomes *loud* absence (the #88
class). Warn, never raise (fail-open: instrumentation must never break training).

`is_initialized()` alone is insufficient: `init(mode="manual")` is initialized
yet installs no patches, so we inspect the config *flags*, not init-ness.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Logical stream -> the init-config flag that must be True to capture it.
# (step_time is emitted directly by trace_step. optimizer is mode-gated:
# since upstream #146 the auto hooks install only when init mode == "auto".
# Neither is flag-gated, so neither is checked here; warning on a dark
# optimizer stream in manual/selective mode is a possible follow-up.)
_PATCH_GATED = {
    "dataloader_fetch": "patch_dataloader",
    "forward": "patch_forward",
    "backward": "patch_backward",
    "h2d": "patch_h2d",
}


def warn_if_missing_streams(integration: str, requires: set[str]) -> None:
    """
    Warn (never raise) if patch-gated streams the integration owes won't be
    captured under the current TraceML init config.
    """
    if os.environ.get("TRACEML_DISABLED") == "1":
        return
    try:
        from traceml_ai.sdk.initial import get_init_config

        cfg = get_init_config()
        missing = []
        for stream in sorted(requires):
            flag = _PATCH_GATED.get(stream)
            if flag is None:
                continue  # not patch-gated; not checked here
            if cfg is None or not getattr(cfg, flag, False):
                missing.append(stream)

        if missing:
            logger.warning(
                "[TraceML] %s is active but these telemetry streams will NOT be "
                "captured: %s. Call traceml_ai.init(mode='auto') before training "
                "to enable them.",
                integration,
                ", ".join(missing),
            )
    except Exception:
        # Capability check is best-effort; it must never break training.
        pass
