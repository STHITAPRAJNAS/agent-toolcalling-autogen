from pathlib import Path
from typing import Dict, Any
import yaml
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float
    max_tokens: int
    region: str = None

class EmbeddingConfig(BaseModel):
    store: str
    dimension: int
    connection: Dict[str, Any] = None

class DatabaseConfig(BaseModel):
    type: str
    path: str = None
    connection: Dict[str, Any] = None

class AgentConfig(BaseModel):
    name: str
    description: str
    max_iterations: int
    timeout: int
    connection: Dict[str, Any] = None
    embedding_batch_size: int = None
    schema_cache_ttl: int = None

class MemoryConfig(BaseModel):
    type: str
    host: str
    port: int
    db: int
    ttl: int
    password: str = None

class LoggingConfig(BaseModel):
    level: str
    format: str
    file: str
    max_size: int = None
    backup_count: int = None

class APIConfig(BaseModel):
    host: str
    port: int
    debug: bool
    workers: int
    timeout: int = None
    rate_limit: Dict[str, int] = None

class MonitoringConfig(BaseModel):
    prometheus: Dict[str, Any] = None
    health_check: Dict[str, int] = None

class Config(BaseModel):
    llm: LLMConfig
    embedding: EmbeddingConfig
    database: DatabaseConfig
    agents: Dict[str, AgentConfig]
    memory: MemoryConfig
    logging: LoggingConfig
    api: APIConfig
    monitoring: MonitoringConfig = None

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Replace environment variables
    def replace_env_vars(obj):
        if isinstance(obj, dict):
            return {k: replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]
            return os.getenv(env_var)
        return obj

    config_dict = replace_env_vars(config_dict)
    return Config(**config_dict)

def get_config(env: str = "local") -> Config:
    """Get configuration based on environment."""
    config_dir = Path(__file__).parent.parent.parent / "config"
    config_path = config_dir / f"{env}_config.yaml"
    return load_config(str(config_path)) 