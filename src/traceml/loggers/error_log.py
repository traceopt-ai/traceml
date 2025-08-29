import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_error_logger(log_dir: str = None) -> logging.Logger:
    """
    Configure a global error logger for TraceML.
    Writes WARN+ to stderr, and ERROR+ to a rotating file.
    """
    logger = logging.getLogger("traceml")
    if logger.handlers:
        return logger

    logger.setLevel(logging.ERROR)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(sh)

    if log_dir is None:
        log_dir = Path.cwd()
    else:
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = RotatingFileHandler(
        log_dir / "traceml_errors.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)

    logger.propagate = False
    return logger


def get_error_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"traceml.{name}")
