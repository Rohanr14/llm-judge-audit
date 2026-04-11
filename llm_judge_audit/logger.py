import logging

from rich.logging import RichHandler

from llm_judge_audit.config import config


def setup_logger(name: str = "llm-judge-audit") -> logging.Logger:
    """Configures and returns a rich-enabled logger."""
    logger = logging.getLogger(name)

    # Only configure if no handlers are present to avoid duplication
    if not logger.handlers:
        log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(log_level)

        handler = RichHandler(
            console=None,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True
        )
        handler.setLevel(log_level)

        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger

logger = setup_logger()
