from utils.config_loader import load_config
from logger_config import setup_logging

setup_logging()
load_config()

from src.models.iql.actor import Actor

x = Actor()