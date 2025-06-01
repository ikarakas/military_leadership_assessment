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


import pandas as pd
import joblib
import logging
from .feature_processor import flatten_nested_json_features, preprocess_data, TARGET_COLUMNS, get_features_and_targets
from src.visualizations import plot_prediction_radar

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/leadership_model.joblib"
DEFAULT_PREPROCESSOR_PATH = "models/preprocessor.joblib"

def predict_competencies(officer_data_dict, model_path=DEFAULT_MODEL_PATH, preprocessor_path=DEFAULT_PREPROCESSOR_PATH):
    """
    Predicts leadership competencies for a single officer.
    officer_data_dict: A dictionary representing the officer's data (excluding target summary).
    """
    logger.info("Starting prediction process for a new officer.")

    try:
        # Load the trained model and preprocessor
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found at {model_path}. Please train a model first.")
            raise
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise

        try:
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        except FileNotFoundError:
            logger.error(f"Preprocessor file not found at {preprocessor_path}. This is required for prediction.")
            logger.error("Ensure the preprocessor was saved during training, typically alongside the model.")
            raise
        except Exception as e:
            logger.error(f"Error loading preprocessor from {preprocessor_path}: {e}")
            raise


        # Convert single officer dict to DataFrame for consistency with preprocessing
        officer_df = pd.DataFrame([officer_data_dict])
        logger.debug(f"Officer data as DataFrame: \n{officer_df.head().to_string()}")


        # Flatten features (like competencies, psych_scores) from nested dicts
        officer_df_flat = flatten_nested_json_features(officer_df.copy())
        logger.debug(f"Officer data after flattening: \n{officer_df_flat.head().to_string()}")


        # Ensure officer_df_flat has all columns expected by the preprocessor,
        # filling missing ones with NaN, as the preprocessor's imputer should handle them.
        # Extract feature names the preprocessor was fitted on:
        expected_cols = []
        for name, trans, columns in preprocessor.transformers_:
            if trans != 'drop': # if 'passthrough', columns is a list of original column names
                expected_cols.extend(columns)
        
        # Add any missing expected columns to officer_df_flat with NaN values
        for col in expected_cols:
            if col not in officer_df_flat.columns:
                logger.warning(f"Expected feature '{col}' not found in officer data. Adding as NaN for preprocessing.")
                officer_df_flat[col] = pd.NA # Use pandas NA for better type handling if possible, else np.nan


        # Preprocess the officer's data using the loaded preprocessor
        # The preprocessor was fit on training data (X), not including target columns
        # So, officer_df_flat should only contain feature columns at this stage.
        # If 'leadership_competency_summary' or its flattened parts are present, they should be dropped
        # before this step if the preprocessor wasn't designed to handle them (it usually isn't).
        cols_to_drop_if_present = TARGET_COLUMNS # + any other non-feature identifier cols
        officer_features_df = officer_df_flat.drop(columns=cols_to_drop_if_present, errors='ignore')

        logger.info(f"Officer features DataFrame columns before preprocessing: {officer_features_df.columns.tolist()}")
        X_officer_processed_df = preprocess_data(officer_features_df, preprocessor)
        logger.info(f"Officer data preprocessed. Shape: {X_officer_processed_df.shape}")
        logger.debug(f"Preprocessed officer data: \n{X_officer_processed_df.head().to_string()}")


        # Make predictions
        logger.info("Making predictions...")
        predictions_array = model.predict(X_officer_processed_df)
        logger.info("Predictions made.")

        # Format predictions into the desired dictionary structure
        # The model.estimators_[i].feature_names_in_ might be useful if using MultiOutputRegressor directly
        # but here TARGET_COLUMNS order should be preserved from training.
        predicted_summary = dict(zip(TARGET_COLUMNS, predictions_array[0]))

        logger.info(f"Predicted leadership_competency_summary: {predicted_summary}")

        # Generate radar plot visualization
        logger.info("Generating prediction visualization...")
        # Extract officer's name from the data
        first_name = officer_data_dict.get('first_name', 'unknown')
        last_name = officer_data_dict.get('last_name', 'unknown')
        # Create filename with officer's name
        radar_filename = f'visualizations/officer_radar_{first_name}_{last_name}.png'
        plot_prediction_radar(predicted_summary, save_path=radar_filename)

        return predicted_summary

    except FileNotFoundError:
        # Already logged, re-raise to be handled by CLI
        raise
    except ValueError as ve:
        logger.error(f"ValueError during prediction: {ve}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        raise