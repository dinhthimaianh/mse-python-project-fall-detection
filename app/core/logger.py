# app/core/logger.py
import logging
import logging.config
import yaml
import os
from pathlib import Path

def setup_logging(config_path: str = "config/logging.yaml", default_level: int = logging.INFO, enable: bool = True):
    """Setup logging configuration"""
    if not enable:
        logging.disable(logging.CRITICAL)  # Disable all logging if not enabled
        return
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        # Default logging configuration
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/fall_detection.log'),
                logging.StreamHandler()
            ]
        )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")