import os

import uvicorn

from app.api.app import app
from app.logging_config import configure_logging


def run_api() -> None:
    configure_logging()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_api()
