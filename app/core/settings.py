from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    # Qdrant
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "legal_precedents"
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"
    REGISTERED_MODEL_NAME: str = "legal-embed-model"
    
    # API
    API_TITLE: str = "Legal Semantic Pipeline"
    API_VERSION: str = "1.0.0"
    
    model_config = ConfigDict(
        env_file=".env",
        extra="allow"
    )

settings = Settings()
