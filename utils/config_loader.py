import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_config(config_path=None):
    if config_path is None:
        current_path = Path(__file__)
        project_root = current_path.parent.parent
        config_path = project_root/"config"/"config.yaml"
        
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        logger.info("Config has been load from config/config.yaml")    
    return config

if __name__ == "__main__":
    print(load_config())