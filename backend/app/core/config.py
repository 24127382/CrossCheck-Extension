import os
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    
    # AI Studio Configuration
    PROJECT_NAME: str = "crosscheck-factcheck"
    AI_STUDIO_KEY: str = "" 
    
    # Hugging Face Configuration
    HF_TOKEN: str # <--- THÊM DÒNG NÀY VÀO ĐÂY (Bắt buộc phải có trong file .env)
    
    # Cấu hình tập trung của Pydantic v2
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()