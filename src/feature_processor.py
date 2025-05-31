import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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


    # Impute missing numerical values with median
    # Impute missing categorical values with a constant string 'missing'
    # (More sophisticated imputation could be added here if needed)
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])


    # Identify features present in the DataFrame for fitting
    # This is important because synthetic data or prediction data might miss some due to generation process
    # or because they are not available for a new officer.
    # The preprocessor must handle this gracefully.

    current_numerical_features = [f for f in NUMERICAL_FEATURES + COMPETENCY_FEATURES + COMPETENCY_DOMAIN_FEATURES + PSYCHOMETRIC_FEATURES if f in df_for_fitting.columns]
    current_categorical_features = [f for f in CATEGORICAL_FEATURES if f in df_for_fitting.columns]

    if not current_numerical_features and not current_categorical_features:
        logger.error("No numerical or categorical features found in the dataframe to build preprocessor.")
        raise ValueError("No features to preprocess.")

    transformers_list = []
    if current_numerical_features:
        # Use scikit-learn's SimpleImputer
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
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

    # Ensure all columns the preprocessor was trained on are present
    # This is tricky. The preprocessor's 'transformers' list stores the original columns.
    # We need to make sure df has them, or handle it.
    # For OneHotEncoder with handle_unknown='ignore', it's more robust.
    # For StandardScaler, columns must match.

    # Extract feature names from the preprocessor
    # This is a bit involved to get from a fitted ColumnTransformer
    processed_feature_names = []
    try:
        for name, trans, cols in preprocessor.transformers_:
            if trans == 'drop' or trans == 'passthrough':
                continue
            if name == 'cat' and hasattr(trans.named_steps['onehot'], 'get_feature_names_out'):
                processed_feature_names.extend(trans.named_steps['onehot'].get_feature_names_out(cols))
            elif name == 'num':
                processed_feature_names.extend(cols)
            # Add other transformer types if used
    except Exception as e:
        logger.warning(f"Could not get feature names from preprocessor: {e}. Relying on transform output.")
        # Fallback, feature names might not be accurate if this fails

    # Align columns of df to what preprocessor expects, adding missing ones as NaN
    # The preprocessor (especially SimpleImputer) should handle these NaNs.
    all_expected_features = []
    for _, _, columns in preprocessor.transformers_:
        all_expected_features.extend(columns)
    all_expected_features = list(set(all_expected_features)) # unique

    for col in all_expected_features:
        if col not in df.columns:
            logger.warning(f"Expected column '{col}' not in DataFrame for preprocessing. Adding as NaN.")
            df[col] = np.nan


    X_processed = preprocessor.transform(df)
    logger.info(f"Data processed. New shape: {X_processed.shape}")

    # If feature names couldn't be generated properly, create generic ones
    if not processed_feature_names or len(processed_feature_names) != X_processed.shape[1]:
        processed_feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
        logger.warning("Used generic feature names as specific names could not be derived from preprocessor.")


    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names, index=df.index)
    return X_processed_df


def get_features_and_targets(df):
    """Separates features (X) and targets (Y) from the DataFrame."""
    logger.info("Separating features and targets.")
    df_flat = flatten_nested_json_features(df.copy())

    # Check if target columns exist
    missing_targets = [tc for tc in TARGET_COLUMNS if tc not in df_flat.columns]
    if missing_targets:
        logger.warning(f"Target columns missing, cannot extract Y: {missing_targets}. Assuming prediction mode for this data.")
        Y = None
    else:
        Y = df_flat[TARGET_COLUMNS]
        logger.info(f"Targets (Y) extracted. Shape: {Y.shape}")


    # Define X_cols: all columns that are not targets and not identifiers/raw text
    potential_feature_cols = NUMERICAL_FEATURES + COMPETENCY_FEATURES + COMPETENCY_DOMAIN_FEATURES + PSYCHOMETRIC_FEATURES + CATEGORICAL_FEATURES
    
    # Select only columns that exist in df_flat to avoid KeyErrors
    X_cols = [col for col in potential_feature_cols if col in df_flat.columns]
    
    if not X_cols:
        logger.error("No feature columns found after flattening and selection.")
        raise ValueError("No feature columns identified. Check feature definitions and data.")

    X = df_flat[X_cols]
    logger.info(f"Features (X) extracted. Shape: {X.shape}")

    return X, Y