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
Processes and transforms military officer data for machine learning.
Handles complex nested data structures, missing values, and feature engineering.
Turns raw data into model-ready features.
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import joblib # For saving/loading preprocessor
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# Define feature groups
# These might need adjustment based on final feature selection strategy
NUMERICAL_FEATURES = [
    'rank_index', 'age', 'years_of_service',
    'combat_deployments', 'medals_and_commendations', 'unit_readiness_score',
    'promotion_potential_score'
]

# Flattened competency scores
COMPETENCY_FEATURES = [
    "thinks_strategically", "possesses_english_language_skills", "engages_in_ethical_reasoning",
    "builds_trust", "facilitates_collaboration_communication", "builds_consensus",
    "integrates_technology", "understands_effects_of_leveraging_technology",
    "understands_capabilities", "instills_need_for_change", "anticipates_change_requirements",
    "provides_support_for_change", "enables_empowers_others", "upholds_principles",
    "relationship_oriented", "thrives_in_ambiguity", "demonstrates_resilience",
    "learning_oriented", "operates_in_nato_context", "operates_in_military_context",
    "operates_in_cross_cultural_context"
]

COMPETENCY_DOMAIN_FEATURES = [
    "cognitive", "social", "technological", "transformative", "personal", "professional"
]

PSYCHOMETRIC_FEATURES = [
    "conscientiousness", "extraversion", "agreeableness", "neuroticism", "openness"
]

CATEGORICAL_FEATURES = [
    'branch', 'rank', 'specialty', 'education', 'leadership_style'
]

# Text features (to be excluded for now for simplicity, requires NLP techniques if used)
# TEXT_FEATURES = ['performance_review', '360_feedback', 'interaction_transcript']

TARGET_COLUMNS = [
    "strategic_thinking", "communication", "team_leadership", "execution", "adaptability"
]

def flatten_nested_json_features(df):
    """Flattens nested JSON structures (competencies, etc.) into individual columns."""
    logger.info("Flattening nested JSON features.")
    # Competencies
    if 'competencies' in df.columns:
        comp_df = pd.json_normalize(df['competencies'])
        comp_df.columns = [f"{col}" for col in comp_df.columns] # Keep original names if they are unique
        df = pd.concat([df.drop('competencies', axis=1), comp_df], axis=1)

    # Competency Domains
    if 'competency_domains' in df.columns:
        comp_domain_df = pd.json_normalize(df['competency_domains'])
        comp_domain_df.columns = [f"{col}" for col in comp_domain_df.columns]
        df = pd.concat([df.drop('competency_domains', axis=1), comp_domain_df], axis=1)

    # Psychometric Scores
    if 'psychometric_scores' in df.columns:
        psych_df = pd.json_normalize(df['psychometric_scores'])
        psych_df.columns = [f"{col}" for col in psych_df.columns]
        df = pd.concat([df.drop('psychometric_scores', axis=1), psych_df], axis=1)

    # Leadership Competency Summary (Targets)
    if 'leadership_competency_summary' in df.columns and \
       isinstance(df['leadership_competency_summary'].iloc[0], dict): # check if it's not already flattened
        target_df = pd.json_normalize(df['leadership_competency_summary'])
        df = pd.concat([df.drop('leadership_competency_summary', axis=1), target_df], axis=1)

    # Ensure top-level columns are preserved
    top_level_columns = ['branch', 'rank', 'specialty', 'education', 'leadership_style']
    for col in top_level_columns:
        if col not in df.columns:
            df[col] = np.nan

    logger.info(f"DataFrame shape after flattening: {df.shape}")
    return df

def create_preprocessor(df_for_fitting):
    """
    Creates a ColumnTransformer preprocessor.
    This preprocessor should be fit on the training data.
    """
    logger.info("Creating feature preprocessor.")

    # Ensure all expected features exist, fill with NaN if not, then impute
    all_defined_features = NUMERICAL_FEATURES + COMPETENCY_FEATURES + COMPETENCY_DOMAIN_FEATURES + PSYCHOMETRIC_FEATURES + CATEGORICAL_FEATURES
    for col in all_defined_features:
        if col not in df_for_fitting.columns:
            logger.warning(f"Column '{col}' not found in DataFrame for fitting preprocessor. Adding it as NaN.")
            df_for_fitting[col] = np.nan

    # Convert categorical columns to string type
    for col in CATEGORICAL_FEATURES:
        if col in df_for_fitting.columns:
            df_for_fitting[col] = df_for_fitting[col].astype(str)

    # Identify features present in the DataFrame for fitting
    current_numerical_features = [f for f in NUMERICAL_FEATURES + COMPETENCY_FEATURES + COMPETENCY_DOMAIN_FEATURES + PSYCHOMETRIC_FEATURES if f in df_for_fitting.columns]
    current_categorical_features = [f for f in CATEGORICAL_FEATURES if f in df_for_fitting.columns]

    if not current_numerical_features and not current_categorical_features:
        logger.error("No numerical or categorical features found in the dataframe to build preprocessor.")
        raise ValueError("No features to preprocess.")

    transformers_list = []
    if current_numerical_features:
        # Use MinMaxScaler instead of StandardScaler
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler(feature_range=(0, 1)))  # Scale to [0,1] range
        ])
        transformers_list.append(('num', num_pipeline, current_numerical_features))

    if current_categorical_features:
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers_list.append(('cat', cat_pipeline, current_categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder='drop'  # Drop other columns (like IDs, text descriptions for now)
    )
    logger.info("Fitting preprocessor...")
    preprocessor.fit(df_for_fitting) # Fit it here
    logger.info("Preprocessor created and fitted.")
    return preprocessor

def preprocess_data(df, preprocessor):
    """
    Applies the fitted preprocessor to the data.
    """
    logger.info(f"Preprocessing data with shape: {df.shape}")

    # Convert categorical columns to string type
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Ensure all columns the preprocessor was trained on are present
    processed_feature_names = []
    try:
        for name, trans, cols in preprocessor.transformers_:
            if trans == 'drop' or trans == 'passthrough':
                continue
            if name == 'cat' and hasattr(trans.named_steps['onehot'], 'get_feature_names_out'):
                processed_feature_names.extend(trans.named_steps['onehot'].get_feature_names_out(cols))
            elif name == 'num':
                processed_feature_names.extend(cols)
    except Exception as e:
        logger.warning(f"Could not extract feature names from preprocessor: {e}")
        # If we can't get the names, use generic ones
        processed_feature_names = [f"feature_{i}" for i in range(preprocessor.transform(df).shape[1])]

    # Transform the data
    processed_data = preprocessor.transform(df)
    
    # Convert to DataFrame with proper column names
    processed_df = pd.DataFrame(processed_data, columns=processed_feature_names)
    logger.info(f"Data processed. New shape: {processed_df.shape}")
    return processed_df

def get_features_and_targets(df):
    """
    Extracts features and target variables from the DataFrame.
    """
    # First flatten any nested JSON structures
    df_flat = flatten_nested_json_features(df.copy())
    
    # Create preprocessor
    preprocessor = create_preprocessor(df_flat)
    
    # Preprocess features
    X = preprocess_data(df_flat, preprocessor)
    
    # Extract targets if they exist
    y = None
    if all(col in df_flat.columns for col in TARGET_COLUMNS):
        y = df_flat[TARGET_COLUMNS]
    
    return X, y