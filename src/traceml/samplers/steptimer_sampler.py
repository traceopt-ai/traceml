from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, Any, Optional, Deque, List

import numpy as np

from traceml.utils.steptimer import StepTimeEvent, get_steptimer_queue
from .base_sampler import BaseSampler
from traceml.loggers.error_log import setup_error_logger, get_error_logger


