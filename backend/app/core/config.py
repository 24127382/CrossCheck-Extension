import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # API Configuration
    API_TITLE: str = "CrossCheck Fact-Check API"
    API_VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # Model Configuration
    CLIP_MODEL_NAME: str = "ViT-B/32"
    ENTAILMENT_MODEL_NAME: str = "roberta-large-mnli"
    RETRIEVAL_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Service Configuration
    MAX_EVIDENCE_COUNT: int = 5
    RETRIEVAL_TOP_K: int = 10
    CONFIDENCE_THRESHOLD: float = 0.3
    
    # Cache Configuration
    ENABLE_CACHE: bool = True
    CACHE_DIR: str = "./data/cache"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
