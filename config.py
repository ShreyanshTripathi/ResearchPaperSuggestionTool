from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",           
        env_file_encoding="utf-8",
        extra="ignore",            
    )

    open_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    arxiv_max_results: int = 10
    semantic_scholar_max_results: int = 8
    chroma_persist_dir: Path = Path("./research_vector_store")
    temporal_decay_lambda: float = 0.05
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    broad_topic_expansion_count: int = 3
    agent1_final_papers: int = 5
    agent2_final_papers: int = 6
    agent3_final_papers: int = 4

settings = Settings()
