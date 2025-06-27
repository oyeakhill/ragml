"""Configuration management for the Document Q&A Agent."""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Application configuration."""
    
    # OpenAI Configuration
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    max_tokens: int = 2000
    
    # Application Settings
    app_name: str = "Document Q&A Agent"
    app_port: int = 8501
    
    # Vector Store Settings
    vector_store_type: str = "chroma"  # Options: chroma, faiss
    chroma_persist_dir: str = "./chroma_db"
    
    # Chunking Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval Settings
    top_k_results: int = 5
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        return cls(
            openai_api_key=openai_api_key,
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            llm_model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
            app_name=os.getenv("APP_NAME", "Document Q&A Agent"),
            app_port=int(os.getenv("APP_PORT", "8501")),
            vector_store_type=os.getenv("VECTOR_STORE_TYPE", "chroma"),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            top_k_results=int(os.getenv("TOP_K_RESULTS", "5"))
        )
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        if self.top_k_results < 1:
            raise ValueError("top_k_results must be at least 1")
        
        if self.llm_temperature < 0 or self.llm_temperature > 2:
            raise ValueError("LLM temperature must be between 0 and 2")


# Singleton instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the configuration singleton."""
    global _config
    if _config is None:
        _config = Config.from_env()
        _config.validate()
    return _config
