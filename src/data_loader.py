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
Loads and validates military officer data from JSON and JSONL files.
Ensures data quality and consistency for model training and prediction.
Simple interface, robust error handling.
"""

import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data_from_jsonl(file_path):
    """Loads officer data from a JSON Lines file."""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON on line {line_number} in {file_path}: {e}")
                    logger.error(f"Problematic line content: {line.strip()}")
                    # Optionally, skip the line or raise an error
                    continue # Skip malformed lines
        logger.info(f"Successfully loaded {len(data)} records from {file_path}")
        return pd.DataFrame(data)
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data from {file_path}: {e}")
        raise

def load_officer_data_from_json(file_path):
    """Loads a single officer's data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            officer_data = json.load(f)
        logger.info(f"Successfully loaded officer data from {file_path}")
        return officer_data # Returns a dict
    except FileNotFoundError:
        logger.error(f"Officer data file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading officer data from {file_path}: {e}")
        raise

def validate_data(df):
    """Performs basic validation on the loaded DataFrame."""
    # Example validation: check for essential columns
    required_columns = ['branch', 'rank', 'age', 'competencies', 'psychometric_scores'] # Add more as needed
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"DataFrame is missing required columns: {missing_cols}")
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    # Check if 'leadership_competency_summary' exists for training data (not for prediction input)
    # This check should be context-dependent (i.e., done in the training script)

    logger.info("Data validation passed (basic checks).")
    return True