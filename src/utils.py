"""
MIT License

Copyright (c) 2025 Ilker M. Karakas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Core utility functions for the Military Leadership Assessment system.
Handles logging, date calculations, and common operations.
The foundation that keeps everything running smoothly.
"""

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