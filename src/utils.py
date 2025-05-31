import logging
import os
from datetime import datetime

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

def setup_logging(level=logging.INFO):
    """Sets up basic logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()  # To also print to console
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete.")
    return logger

def calculate_years_of_service(start_date_str, current_date_str):
    """Calculates years of service if not provided or needs verification."""
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        return (current_date - start_date).days / 365.25
    except ValueError as e:
        logging.error(f"Error parsing dates: {start_date_str}, {current_date_str} - {e}")
        return 0 # Or raise error

# You can add other common utility functions here