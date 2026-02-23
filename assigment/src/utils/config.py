"""Configuration loader utility."""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration loader for the application."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to config YAML file. If None, uses default.
        """
        if config_path is None:
            # Default to config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.
        
        Args:
            key: Dot-notation key (e.g., 'llm.model')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def app_name(self) -> str:
        """Get application name."""
        return self.get('app.name', 'Agentic SQL Generator')
    
    @property
    def app_version(self) -> str:
        """Get application version."""
        return self.get('app.version', '1.0.0')
    
    @property
    def database_path(self) -> str:
        """Get database path."""
        project_root = Path(__file__).parent.parent.parent
        db_path = self.get('database.path', 'data/customer_service.db')
        return str(project_root / db_path)
    
    @property
    def sample_schema_path(self) -> str:
        """Get sample schema path."""
        project_root = Path(__file__).parent.parent.parent
        schema_path = self.get('database.sample_data_path', 'data/sample_schema.json')
        return str(project_root / schema_path)
    
    @property
    def llm_provider(self) -> str:
        """Get LLM provider."""
        return self.get('llm.provider', 'ollama')
    
    @property
    def llm_model(self) -> str:
        """Get LLM model."""
        return self.get('llm.model', 'llama3')
    
    @property
    def llm_temperature(self) -> float:
        """Get LLM temperature."""
        return self.get('llm.temperature', 0.1)
    
    @property
    def llm_base_url(self) -> Optional[str]:
        """Get LLM base URL (for OpenRouter, etc.)."""
        return self.get('llm.base_url')
    
    @property
    def hf_token(self) -> Optional[str]:
        """Get HuggingFace token from environment."""
        import os
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    
    @property
    def embedding_model(self) -> str:
        """Get embedding model."""
        return self.get('embeddings.model', 'sentence-transformers/all-MiniLM-L6-v2')
    
    @property
    def embedding_device(self) -> str:
        """Get embedding device."""
        return self.get('embeddings.device', 'cpu')
    
    @property
    def retrieval_top_k(self) -> int:
        """Get retrieval top-k."""
        return self.get('retrieval.top_k', 5)
    
    @property
    def similarity_threshold(self) -> float:
        """Get similarity threshold."""
        return self.get('retrieval.similarity_threshold', 0.5)
    
    @property
    def max_retries(self) -> int:
        """Get max retries for agent."""
        return self.get('agent.max_retries', 2)
    
    @property
    def confidence_threshold(self) -> float:
        """Get confidence threshold."""
        return self.get('agent.confidence_threshold', 0.7)
    
    @property
    def enable_self_correction(self) -> bool:
        """Get self-correction enabled flag."""
        return self.get('agent.enable_self_correction', True)
    
    @property
    def api_host(self) -> str:
        """Get API host."""
        return self.get('api.host', '0.0.0.0')
    
    @property
    def api_port(self) -> int:
        """Get API port."""
        return self.get('api.port', 8000)
    
    @property
    def log_level(self) -> str:
        """Get log level."""
        return self.get('observability.log_level', 'INFO')


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance.
    
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config
