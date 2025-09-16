"""
Configuration settings for the Research_CODE API.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    """Application settings configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Research_CODE API"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")

    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")

    # Data Paths
    DATA_DIR: Path = Field(default=Path(__file__).parent.parent.parent / "data")
    INPUT_DIR: Optional[Path] = Field(default=None)
    INTERMEDIATE_DIR: Optional[Path] = Field(default=None)
    OUTPUT_DIR: Optional[Path] = Field(default=None)

    # Graph Settings
    DEFAULT_SUBGRAPH_DISTANCE: int = Field(default=1, env="DEFAULT_SUBGRAPH_DISTANCE")
    DEFAULT_GRAPH_PATH: str = Field(default="./data/output/optimized_entity_graph.json", env="DEFAULT_GRAPH_PATH")

    # LLM Settings
    DEFAULT_MODEL: str = Field(default="gpt-4o-mini", env="DEFAULT_MODEL")
    DEFAULT_TEMPERATURE: float = Field(default=0.0, env="DEFAULT_TEMPERATURE")

    # Processing Settings
    MAX_CONCURRENT_REQUESTS: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    REQUEST_TIMEOUT: int = Field(default=300, env="REQUEST_TIMEOUT")  # 5 minutes

    # Evaluation Settings
    EVAL_MODEL: str = Field(default="gpt-4.1", env="EVAL_MODEL")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set derived paths
        if self.INPUT_DIR is None:
            self.INPUT_DIR = self.DATA_DIR / "input"
        if self.INTERMEDIATE_DIR is None:
            self.INTERMEDIATE_DIR = self.DATA_DIR / "intermediate"
        if self.OUTPUT_DIR is None:
            self.OUTPUT_DIR = self.DATA_DIR / "output"

# Global settings instance
settings = Settings()
