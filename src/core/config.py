from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field

load_dotenv()

@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    region: Optional[str] = None
    api_key: Optional[str] = None

@dataclass
class EmbeddingConfig:
    type: str  # "in_memory" or "pgvector"
    dimension: int = 1536
    connection: Optional[Dict[str, Any]] = None

@dataclass
class DatabaseConfig:
    type: str  # "sqlite" or "postgresql"
    path: Optional[str] = None
    connection: Optional[Dict[str, Any]] = None

@dataclass
class GroupChatConfig:
    max_round: int = 10
    speaker_selection_method: str = "round_robin"
    roles: Dict[str, str] = field(default_factory=dict)

@dataclass
class AgentConfig:
    name: str
    description: str
    max_iterations: int = 5
    connection: Optional[Dict[str, Any]] = None
    group_chat: GroupChatConfig = field(default_factory=GroupChatConfig)

@dataclass
class Config:
    env: str
    llm: LLMConfig
    embedding: EmbeddingConfig
    database: DatabaseConfig
    memory: DatabaseConfig
    agents: Dict[str, AgentConfig]
    logging: Dict[str, Any] = field(default_factory=dict)
    api: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert nested dictionaries to appropriate dataclass instances
    config_dict['llm'] = LLMConfig(**config_dict['llm'])
    config_dict['embedding'] = EmbeddingConfig(**config_dict['embedding'])
    config_dict['database'] = DatabaseConfig(**config_dict['database'])
    config_dict['memory'] = DatabaseConfig(**config_dict['memory'])
    
    # Convert agent configs
    agents = {}
    for name, agent_config in config_dict['agents'].items():
        if 'group_chat' in agent_config:
            agent_config['group_chat'] = GroupChatConfig(**agent_config['group_chat'])
        agents[name] = AgentConfig(**agent_config)
    config_dict['agents'] = agents
    
    return Config(**config_dict)

def get_config() -> Config:
    """Get configuration based on environment."""
    env = os.getenv('ENV', 'development')
    config_path = Path(__file__).parent.parent.parent / 'config' / f'{env}_config.yaml'
    return load_config(str(config_path)) 