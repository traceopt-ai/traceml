from traceml.aggregator.display_drivers.layout import (
    LAYER_COMBINED_MEMORY_LAYOUT,
    LAYER_COMBINED_TIMER_LAYOUT,
    MODEL_COMBINED_LAYOUT,
    PROCESS_LAYOUT,
    STEPTIMER_LAYOUT,
    SYSTEM_LAYOUT,
)

TRACE_ML_PAGE = [
    # Row 1: context
    [
        SYSTEM_LAYOUT,
        PROCESS_LAYOUT,
        STEPTIMER_LAYOUT,
    ],
    # Row 2: model summary
    [
        MODEL_COMBINED_LAYOUT,
    ],
    # Row 3: layer tables
    [
        LAYER_COMBINED_MEMORY_LAYOUT,
        LAYER_COMBINED_TIMER_LAYOUT,
    ],
]
