import logging

from app.config.settings import get_settings


def configure_logging() -> None:
    settings = get_settings()
    level_name = settings.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    # for noisy_logger in ["httpx", "httpcore", "chromadb", "urllib3", "uvicorn.access"]:
    #     logging.getLogger(noisy_logger).setLevel(logging.WARNING)