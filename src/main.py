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
Command-line interface for the Military Leadership Assessment system.
Train models, generate synthetic data, make predictions, and visualize results.
Simple commands, powerful insights.
"""


import argparse
import logging
import os
import sys
import joblib

# Ensure src directory is in Python path if running main.py directly for development
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir)) # Add parent dir (military_leadership_assessment) to path

from src.utils import setup_logging # Assuming running from military_leadership_assessment/ as CWD
from src.data_loader import load_data_from_jsonl, load_officer_data_from_json, validate_data
from src.model_trainer import train_model, DEFAULT_MODEL_PATH as TRAINER_DEFAULT_MODEL_PATH, DEFAULT_PREPROCESSOR_PATH as TRAINER_DEFAULT_PREPROCESSOR_PATH
from src.predictor import predict_competencies, DEFAULT_MODEL_PATH as PREDICTOR_DEFAULT_MODEL_PATH, DEFAULT_PREPROCESSOR_PATH as PREDICTOR_DEFAULT_PREPROCESSOR_PATH
from src.synthetic_data_generator import generate_synthetic_data
from src.feature_processor import TARGET_COLUMNS, get_features_and_targets
from src.visualizations import (
    plot_model_performance,
    plot_feature_importance,
    plot_prediction_radar
)

# Use the predictor's default paths for consistency
DEFAULT_MODEL_PATH = PREDICTOR_DEFAULT_MODEL_PATH
DEFAULT_PREPROCESSOR_PATH = PREDICTOR_DEFAULT_PREPROCESSOR_PATH

logger = None # Will be initialized by setup_logging

def handle_train(args):
    logger.info(f"Executing 'train' command with data: {args.data_path}, model output: {args.model_output_path}, preprocessor output: {args.preprocessor_output_path}")
    try:
        df = load_data_from_jsonl(args.data_path)
        validate_data(df) # Basic validation
        if 'leadership_competency_summary' not in df.columns and not any(isinstance(i, dict) and 'strategic_thinking' in i for i in df['leadership_competency_summary'].dropna()):
             logger.error("Training data must contain 'leadership_competency_summary' field with target values.")
             raise ValueError("Missing 'leadership_competency_summary' in training data.")
        
        trained_model, fitted_preprocessor, results = train_model(df, args.model_output_path, args.preprocessor_output_path)
        logger.info("Training completed successfully.")
        logger.info(f"Model evaluation results (on test split): {results}")

    except FileNotFoundError:
        logger.error(f"Training data file not found: {args.data_path}")
        sys.exit(1)
    except ValueError as ve:
        logger.error(f"Data validation or processing error during training: {ve}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)
        sys.exit(1)

def handle_predict(args):
    logger.info(f"Executing 'predict' command for officer data: {args.officer_data}")
    logger.info(f"Using model: {args.model_path}, preprocessor: {args.preprocessor_path}")
    try:
        officer_input_data = load_officer_data_from_json(args.officer_data)
        if "leadership_competency_summary" in officer_input_data:
            logger.warning("Input officer data contains 'leadership_competency_summary'. It will be ignored for prediction.")
            # No need to remove it from the dict as feature_processor.get_features_and_targets handles it.
        
        predicted_summary = predict_competencies(
            officer_input_data,
            model_path=args.model_path,
            preprocessor_path=args.preprocessor_path
        )
        logger.info("Prediction completed successfully.")
        print("\nPredicted Leadership Competency Summary:")
        for key, value in predicted_summary.items():
            print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
        print("")

    except FileNotFoundError:
        logger.error(f"File not found. Ensure model, preprocessor, and officer data paths are correct.")
        sys.exit(1)
    except ValueError as ve:
        logger.error(f"Data or model error during prediction: {ve}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        sys.exit(1)

def handle_generate_synthetic(args):
    logger.info(f"Executing 'generate-synthetic-data' command. Num officers: {args.num_officers}, Output: {args.output_path}")
    try:
        generate_synthetic_data(args.num_officers, args.output_path)
        logger.info("Synthetic data generation completed successfully.")
    except IOError:
        logger.error(f"Could not write synthetic data to {args.output_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during synthetic data generation: {e}", exc_info=True)
        sys.exit(1)

def handle_visualize(args):
    """Handle the visualize command."""
    logger.info(f"Generating visualizations for data: {args.data_path}")
    try:
        # Load data
        df = load_data_from_jsonl(args.data_path)
        validate_data(df)
        
        # Load model
        model = joblib.load(args.model_path)
        # Load preprocessor
        preprocessor = joblib.load(DEFAULT_PREPROCESSOR_PATH)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate visualizations
        # Plot model performance
        results = {}  # You might want to load actual results from a file
        plot_model_performance(
            results,
            save_path=os.path.join(args.output_dir, 'model_performance.png')
        )
        
        # Plot feature importance for each target
        X, _ = get_features_and_targets(df)
        # Preprocess X to get correct feature names
        from src.feature_processor import preprocess_data
        X_processed = preprocess_data(X, preprocessor)
        processed_feature_names = X_processed.columns
        for i, target in enumerate(TARGET_COLUMNS):
            plot_feature_importance(
                model.estimators_[i],
                processed_feature_names,
                save_path=os.path.join(args.output_dir, f'feature_importance_{target}.png')
            )
        
        # Plot radar for a sample officer
        if len(df) > 0:
            sample_officer = df.iloc[0].to_dict()
            predictions = predict_competencies(
                sample_officer,
                model_path=args.model_path
            )
            plot_prediction_radar(
                predictions,
                save_path=os.path.join(args.output_dir, 'sample_officer_radar.png')
            )
        
        logger.info(f"Visualizations generated and saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred during visualization: {e}", exc_info=True)
        sys.exit(1)

def main():
    global logger # Allow main to assign the global logger

    parser = argparse.ArgumentParser(description="Military Leadership Assessment Application")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new leadership model")
    train_parser.add_argument("--data-path", type=str, required=True, help="Path to the training data file (JSONL format)")
    train_parser.add_argument("--model-output-path", type=str, default=TRAINER_DEFAULT_MODEL_PATH, help=f"Path to save the trained model (default: {TRAINER_DEFAULT_MODEL_PATH})")
    train_parser.add_argument("--preprocessor-output-path", type=str, default=TRAINER_DEFAULT_PREPROCESSOR_PATH, help=f"Path to save the fitted preprocessor (default: {TRAINER_DEFAULT_PREPROCESSOR_PATH})")
    train_parser.set_defaults(func=handle_train)

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict leadership competencies for an officer")
    predict_parser.add_argument("--officer-data", type=str, required=True, help="Path to the JSON file containing the officer's data for prediction")
    predict_parser.add_argument("--model-path", type=str, default=PREDICTOR_DEFAULT_MODEL_PATH, help=f"Path to the trained model file (default: {PREDICTOR_DEFAULT_MODEL_PATH})")
    predict_parser.add_argument("--preprocessor-path", type=str, default=PREDICTOR_DEFAULT_PREPROCESSOR_PATH, help=f"Path to the fitted preprocessor file (default: {PREDICTOR_DEFAULT_PREPROCESSOR_PATH})")
    predict_parser.set_defaults(func=handle_predict)

    # Generate Synthetic Data command
    gsd_parser = subparsers.add_parser("generate-synthetic-data", help="Generate synthetic officer data")
    gsd_parser.add_argument("--output-path", type=str, required=True, help="Path to save the generated synthetic data (JSONL format)")
    gsd_parser.add_argument("--num-officers", type=int, default=100, help="Number of synthetic officer profiles to generate (default: 100)")
    gsd_parser.set_defaults(func=handle_generate_synthetic)

    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations for existing data")
    viz_parser.add_argument("--data-path", type=str, required=True, help="Path to the data file (JSONL format)")
    viz_parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help=f"Path to the trained model file (default: {DEFAULT_MODEL_PATH})")
    viz_parser.add_argument("--output-dir", type=str, default="visualizations", help="Directory to save visualizations (default: visualizations)")
    viz_parser.set_defaults(func=handle_visualize)

    args = parser.parse_args()

    # Setup logging after parsing args to get log_level
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logging(level=log_level_numeric) # Initialize the global logger

    # Call the appropriate handler function based on the command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    # Example of how to run:
    # Create dummy data first if you don't have any.
    # python src/main.py generate-synthetic-data --output-path data/synthetic_data/generated.jsonl --num-officers 500 --log-level DEBUG
    # Then train:
    # python src/main.py train --data-path data/synthetic_data/generated.jsonl --log-level DEBUG
    # Then prepare a single officer JSON (e.g., copy one from generated.jsonl, remove leadership_competency_summary)
    # And predict:
    # python src/main.py predict --officer-data path/to/your/single_officer.json --log-level DEBUG
    main()