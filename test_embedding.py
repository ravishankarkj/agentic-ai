from app.api.dependencies import get_container
from app.logging_config import configure_logging

configure_logging()

container = get_container()

result = container.ingestion_service.ingest()

print(f"Ingestion Result: {result}")