"""Configuration loader for math2latex.

Loads parameters from configs/base.yaml into Python dictionaries or EasyDict objects.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str = "configs/base.yaml") -> Dict[str, Any]:
    """Load YAML config file from disk.
    
    Args:
        config_path: Relative or absolute path to the YAML config file.
        
    Returns:
        Nested dictionary containing all configuration parameters.
    """
    path = Path(config_path)
    if not path.is_absolute():
        # Look relative to project root (DL folder)
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / config_path
        
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config
