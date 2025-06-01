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
Trains machine learning models to predict leadership competencies.
Uses Random Forest for robust predictions across multiple leadership dimensions.
Data-driven insights for better leadership assessment.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import os
from .feature_processor import get_features_and_targets, create_preprocessor, preprocess_data, TARGET_COLUMNS
from src.visualizations import plot_model_performance, plot_feature_importance
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/leadership_model.joblib"
DEFAULT_PREPROCESSOR_PATH = "models/preprocessor.joblib"

def train_model(df_train, model_output_path=DEFAULT_MODEL_PATH, preprocessor_output_path=DEFAULT_PREPROCESSOR_PATH):
    """Trains a multi-output regression model."""
    logger.info("Starting model training process.")

    try:
        X, Y = get_features_and_targets(df_train)
        if Y is None:
            logger.error("Target variables (Y) are missing. Cannot train model.")
            raise ValueError("Target variables (Y) are missing from the training data.")
        if X.empty:
            logger.error("Feature set (X) is empty. Cannot train model.")
            raise ValueError("Feature set (X) is empty.")

        logger.info(f"Shape of X before preprocessing: {X.shape}, Shape of Y: {Y.shape}")

        # Create and fit preprocessor ON THE FULL X data before splitting
        # This ensures encodings/scalings are consistent and learn from all available training data
        preprocessor = create_preprocessor(X) # X here should be the full training feature set
        X_processed = preprocess_data(X, preprocessor)

        logger.info(f"Shape of X after preprocessing: {X_processed.shape}")

        # Split data *after* preprocessing the features
        X_train_proc, X_test_proc, Y_train, Y_test = train_test_split(
            X_processed, Y, test_size=0.2, random_state=42
        )

        logger.info(f"Training data shape: X_train_proc={X_train_proc.shape}, Y_train={Y_train.shape}")
        logger.info(f"Test data shape: X_test_proc={X_test_proc.shape}, Y_test={Y_test.shape}")

        # Define the model - Random Forest is a good start for this kind of data
        # Wrap with MultiOutputRegressor for predicting multiple targets
        base_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        multi_output_model = MultiOutputRegressor(base_regressor)

        logger.info("Training the MultiOutputRegressor model...")
        multi_output_model.fit(X_train_proc, Y_train)
        logger.info("Model training complete.")

        # Evaluate the model
        Y_pred_test = multi_output_model.predict(X_test_proc)
        Y_pred_train = multi_output_model.predict(X_train_proc)

        logger.info("Evaluating model performance...")
        results = {}
        for i, target_name in enumerate(TARGET_COLUMNS):
            train_mse = mean_squared_error(Y_train.iloc[:, i], Y_pred_train[:, i])
            test_mse = mean_squared_error(Y_test.iloc[:, i], Y_pred_test[:, i])
            train_r2 = r2_score(Y_train.iloc[:, i], Y_pred_train[:, i])
            test_r2 = r2_score(Y_test.iloc[:, i], Y_pred_test[:, i])
            logger.info(f"Target: {target_name}")
            logger.info(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
            logger.info(f"  Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")
            results[target_name] = {'test_mse': test_mse, 'test_r2': test_r2}

        # Generate visualizations
        logger.info("Generating model performance visualizations...")
        plot_model_performance(results, save_path='visualizations/model_performance.png')
        
        # Plot feature importance for each target
        for i, target in enumerate(TARGET_COLUMNS):
            plot_feature_importance(
                multi_output_model.estimators_[i],
                X_processed.columns,
                save_path=f'visualizations/feature_importance_{target}.png'
            )

        # Save the trained model and the preprocessor
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(multi_output_model, model_output_path)
        logger.info(f"Trained model saved to {model_output_path}")

        os.makedirs(os.path.dirname(preprocessor_output_path), exist_ok=True)
        joblib.dump(preprocessor, preprocessor_output_path)
        logger.info(f"Fitted preprocessor saved to {preprocessor_output_path}")

        return multi_output_model, preprocessor, results

    except ValueError as ve:
        logger.error(f"ValueError during model training: {ve}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during model training: {e}", exc_info=True)
        raise