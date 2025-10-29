"""
Configuration loader
"""
import yaml
from pathlib import Path
import os
from typing import Dict

def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration from YAML file with environment variable substitution"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Substitute environment variables
    config = _substitute_env_vars(config)
    
    return config

def _substitute_env_vars(config: any) -> any:
    """Recursively substitute environment variables in config"""
    if isinstance(config, dict):
        return {k: _substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_substitute_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
        # Extract variable name and default value
        var_expr = config[2:-1]
        if ':' in var_expr:
            var_name, default_value = var_expr.split(':', 1)
            return os.getenv(var_name, default_value)
        else:
            return os.getenv(var_expr, config)
    else:
        return config
