"""Configuration management for the deep research agent."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Model Configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "15"))

# Note: GPT-5 has fixed temperature=1 (cannot be changed)
# Other models (gpt-4, claude, etc.) use configurable temperature

# Validate required keys
def validate_config():
    """Validate that all required configuration is present."""
    missing = []
    
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not SERPER_API_KEY:
        missing.append("SERPER_API_KEY")
    
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please add them to backend/.env file"
        )

# Validate on import
validate_config()

