"""Logging configuration helpers for :mod:`athena`."""

from __future__ import annotations

import json
import logging
import logging.config
from pathlib import Path
from typing import Any, Dict, Optional


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_CONFIG = PACKAGE_ROOT / "config" / "logging.json"


def _prepare_log_handlers(config: Dict[str, Any], log_dir: Path) -> None:
    """Ensure rotating file handler paths point inside ``log_dir``.

    The logging configuration expects handlers to define a ``filename``. To
    make the setup location agnostic we rewrite those paths so that they live
    under the provided ``log_dir``.
    """

    handlers = config.get("handlers", {})
    for handler in handlers.values():
        filename = handler.get("filename")
        if not filename:
            continue
        handler["filename"] = str(log_dir / Path(filename).name)


def setup_logging(
    logger_name: Optional[str] = None,
    *,
    config_path: Optional[Path] = None,
) -> logging.Logger:
    """Configure logging using the packaged JSON configuration.

    Parameters
    ----------
    logger_name:
        Name of the logger to return. ``None`` returns the root logger.
    config_path:
        Optional path to an alternative logging configuration.
    """

    resolved_path = (config_path or DEFAULT_LOG_CONFIG).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Logging config not found: {resolved_path}")

    with resolved_path.open(encoding="utf-8") as fh:
        config = json.load(fh)

    log_dir = resolved_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _prepare_log_handlers(config, log_dir)

    logging.config.dictConfig(config)
    return logging.getLogger(logger_name)


__all__ = ["setup_logging", "DEFAULT_LOG_CONFIG", "PACKAGE_ROOT"]

