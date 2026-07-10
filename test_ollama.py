from app.clients.llm_factory import LLMFactory
from app.config.settings import get_settings
from app.logging_config import configure_logging

configure_logging()

settings = get_settings()
llm_factory = LLMFactory(settings)

llm_embedding = llm_factory.create_embeddings()
# llm = llm_factory.create_chat_model()

# response = llm.invoke("Why is earth round?")

# print(f"Response: \n{response}")