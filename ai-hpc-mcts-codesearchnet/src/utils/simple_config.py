import yaml
import os

def load_config():
    """Load configuration from config.yaml in project root."""
    # Get project root (3 levels up from this file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, 'config.yaml')
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Global config instance
CONFIG = load_config()