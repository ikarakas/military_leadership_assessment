import logging
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.synthetic_data_generator import generate_synthetic_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set output file path
output_file = 'data/synthetic_data/generated_realistic.jsonl'

# Generate 5000 synthetic officer profiles
generate_synthetic_data(5000, output_file) 