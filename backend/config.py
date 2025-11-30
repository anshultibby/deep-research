"""Configuration management for the deep research agent."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings with validation using Pydantic.
    
    Automatically loads from environment variables and .env file.
    """
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / '.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    # API Keys (required)
    openai_api_key: str = Field(
        ...,
        description="OpenAI API key for LLM access"
    )
    serper_api_key: str = Field(
        ...,
        description="Serper API key for web search"
    )
    
    # Model Configuration
    default_model: str = Field(
        default="gpt-5.1",
        description="Default LLM model to use (gpt-5 has fixed temperature=1)"
    )
    max_iterations: int = Field(
        default=15,
        ge=1,
        le=100,
        description="Maximum number of agent iterations"
    )
    
    # Search Configuration
    search_num_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of search results to fetch per search query"
    )
    search_content_max_tokens: int = Field(
        default=5000,
        ge=500,
        le=20000,
        description="Maximum number of characters to extract from each web page before truncating"
    )
    
    @field_validator('openai_api_key', 'serper_api_key')
    @classmethod
    def validate_api_keys(cls, v: str, info) -> str:
        """Validate that API keys are not empty."""
        if not v or not v.strip():
            raise ValueError(
                f"{info.field_name} is required. "
                f"Please add it to backend/.env file"
            )
        return v.strip()


# Create global settings instance
# This will validate on import and raise clear errors if config is missing
settings = Settings()

# Export individual settings for backward compatibility
OPENAI_API_KEY = settings.openai_api_key
SERPER_API_KEY = settings.serper_api_key
DEFAULT_MODEL = settings.default_model
MAX_ITERATIONS = settings.max_iterations
SEARCH_NUM_RESULTS = settings.search_num_results
SEARCH_CONTENT_MAX_TOKENS = settings.search_content_max_tokens

