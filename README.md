# Military Leadership Assessment Application v2.0

This Python-based command-line application uses machine learning to assess and predict leadership competencies for military officers.

## Features

* Locally hosted and run.
* Utilizes open-source AI/ML libraries (scikit-learn, pandas, numpy).
* Trains a model on a dataset of military officers (OF-2 to OF-6).
* Predicts `leadership_competency_summary` (strategic_thinking, communication, team_leadership, execution, adaptability) for a given officer.
* Supports generation of synthetic training data.
* Verbose logging for operational transparency.

## Machine Learning Architecture

### Model Architecture
* **Base Model**: Random Forest Regressor (100 trees)
* **Multi-Output Regression**: Uses scikit-learn's MultiOutputRegressor to predict multiple leadership competencies simultaneously
* **Feature Engineering Pipeline**:
  * Numerical Features: Standardization using StandardScaler
  * Categorical Features: One-Hot Encoding with unknown category handling
  * Missing Value Handling: Median imputation for numerical features, constant imputation for categorical features

### Feature Groups
1. **Numerical Features**:
   * Rank index, age, years of service
   * Combat deployments, medals and commendations
   * Unit readiness score, promotion potential score

2. **Competency Features** (21 dimensions):
   * Strategic thinking, language skills, ethical reasoning
   * Trust building, collaboration, consensus building
   * Technology integration and understanding
   * Change management capabilities
   * Leadership and relationship skills
   * Adaptability and resilience metrics

3. **Competency Domain Features**:
   * Cognitive, social, technological
   * Transformative, personal, professional

4. **Psychometric Features**:
   * Big Five personality traits (conscientiousness, extraversion, agreeableness, neuroticism, openness)

5. **Categorical Features**:
   * Branch, rank, specialty
   * Education, leadership style

### Training Process
1. **Data Preprocessing**:
   * Feature extraction and flattening of nested JSON structures
   * Train-test split (80-20) after preprocessing
   * Comprehensive feature scaling and encoding

2. **Model Training**:
   * Multi-output regression approach
   * Parallel training of multiple Random Forest models
   * Automatic feature importance analysis
   * Performance visualization generation

3. **Evaluation Metrics**:
   * Mean Squared Error (MSE)
   * R-squared (R²) score
   * Per-competency performance tracking

### Machine Learning Pipeline - Design Details
```

1. Input Data:

The system primarily uses JSON Lines (.jsonl) files for training data, where each line is a JSON object representing an officer's profile.
For individual predictions, it uses a single JSON file representing one officer.
The data contains nested JSON structures for fields like competencies, competency_domains, and psychometric_scores.

2. Feature Processing (feature_processor.py):

Flattening Nested Data: Nested JSON objects within the officer data (e.g., competencies, psychometric_scores, leadership_competency_summary) are flattened into individual columns using pd.json_normalize. This makes them suitable for direct use as features or targets.

Feature Identification: The script defines specific lists for different types of features:
* NUMERICAL_FEATURES: e.g., 'age', 'years_of_service'.
* COMPETENCY_FEATURES: Flattened individual competency scores.
* COMPETENCY_DOMAIN_FEATURES: Flattened competency domain scores.
* PSYCHOMETRIC_FEATURES: Flattened psychometric scores.
* CATEGORICAL_FEATURES: e.g., 'branch', 'rank', 'specialty'.

Preprocessing Pipelines: Scikit-learn's Pipeline and ColumnTransformer are used to create a preprocessing workflow.
* Numerical Features:
Missing values are imputed using the median strategy (SimpleImputer(strategy='median')).
Features are then scaled using StandardScaler().
* Categorical Features:
Missing values are imputed with a constant string 'missing' (SimpleImputer(strategy='constant', fill_value='missing')).
Features are then one-hot encoded using OneHotEncoder(handle_unknown='ignore', sparse_output=False). handle_unknown='ignore' is important for dealing with unseen categories during prediction.

Preprocessor Fitting: The ColumnTransformer (preprocessor) is fitted exclusively on the training dataset to learn imputation values, scaling parameters, and one-hot encoding mappings.

Data Transformation: The fitted preprocessor is used to transform both training data and new data for prediction, ensuring consistency.

3. Target Variables (TARGET_COLUMNS):

The model aims to predict multiple leadership competency scores.
These target columns are defined in TARGET_COLUMNS (e.g., "strategic_thinking", "communication", "team_leadership", "execution", "adaptability").

4. Model Training (model_trainer.py):

Feature and Target Separation: The get_features_and_targets function is used to separate the dataset into features (X) and target variables (Y).
Data Splitting: The preprocessed feature set and target variables are split into training and testing sets using train_test_split (defaulting to a test size of 20% and random_state=42 for reproducibility).
Model Algorithm:
A RandomForestRegressor (with n_estimators=100, random_state=42, n_jobs=-1) is used as the base estimator.
This is wrapped in a MultiOutputRegressor from scikit-learn, enabling the Random Forest to predict all target leadership competencies simultaneously.
Model Fitting: The MultiOutputRegressor is trained using the preprocessed training features (X_train_proc) and corresponding training targets (Y_train).
Saving Artifacts:
The trained MultiOutputRegressor model is saved to a .joblib file (default: models/leadership_model.joblib).
The fitted ColumnTransformer preprocessor is also saved to a .joblib file (default: models/preprocessor.joblib).

5. Model Evaluation (model_trainer.py):

Predictions are made on both the training and test sets.
Performance is evaluated for each target variable independently.
Metrics Used:
Mean Squared Error (MSE): mean_squared_error is calculated to measure the average squared difference between predicted and actual values.
R-squared (R² score): r2_score is calculated to determine the proportion of the variance in the dependent variable that is predictable from the independent variables.
Results are logged and can be used for visualization (e.g., plot_model_performance).
6. Prediction (predictor.py):

Loading Artifacts: For making predictions on new officer data, the saved model (.joblib) and preprocessor (.joblib) are loaded.
Data Preparation:
The input officer data (a dictionary) is converted into a Pandas DataFrame.
Nested JSON features are flattened using flatten_nested_json_features.
The loaded preprocessor is used to transform the officer's features, ensuring all necessary columns are present (adding missing ones as NaN if needed, which the imputers in the preprocessor handle).
Making Predictions: The preprocessed features of the officer are fed into the loaded model's predict method.
Output: The predictions (an array of scores for the target competencies) are formatted into a dictionary, mapping target names to their predicted scores.

```

## Security Considerations

* **Data Privacy**:
  * All processing is done locally
  * No data transmission to external servers
  * Synthetic data generation for training purposes

* **Model Security**:
  * Model and preprocessor serialization using joblib
  * Secure storage of trained models
  * Input validation and sanitization

* **Access Control**:
  * Command-line interface with controlled access
  * Logging of all operations for audit trails
  * Error handling and validation at all stages

## Deployment and Requirements

### Standalone Operation
The application is designed to run completely offline and standalone:
* No internet connection required for predictions
* All dependencies are local Python packages
* Models are saved locally after training

### System Requirements
* Python 3.8 or higher
* 4GB RAM minimum (8GB recommended)
* 1GB disk space for models and data

### Dependencies
```
Package         Version
--------------- -----------
flask           >=2.0.0
joblib          >=1.0.0
matplotlib      >=3.4.0
numpy           >=1.20.0
pandas          >=1.3.0
scikit-learn    >=1.0.0
seaborn         >=0.11.0
click           >=8.0.0
names           >=0.3.0  # For synthetic data generation
```

Note: The versions shown are minimum required versions. The application will work with newer versions of these packages.

## Command Line Options

The application provides three main commands: `train`, `predict`, and `generate-synthetic-data`. Each command has its own set of options.

### Global Options

All commands support the following global option:
- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Set the logging level (default: INFO)

### Train Command

Trains a new leadership assessment model using provided training data.

```bash
python src/main.py train [options]
```

Options:
- `--data-path PATH`: Path to the training data file in JSONL format (required)
- `--model-output-path PATH`: Path to save the trained model (default: models/leadership_model.joblib)
- `--preprocessor-output-path PATH`: Path to save the fitted preprocessor (default: models/preprocessor.joblib)

Example:
```bash
# Train with default model and preprocessor paths
python src/main.py --log-level DEBUG train --data-path data/synthetic_data/generated_1000.jsonl

# Train with custom output paths
python src/main.py --log-level DEBUG train \
    --data-path data/synthetic_data/generated_1000.jsonl \
    --model-output-path models/custom_model.joblib \
    --preprocessor-output-path models/custom_preprocessor.joblib
```

### Predict Command

Predicts leadership competencies for a new officer using a trained model.

```bash
python src/main.py predict [options]
```

Options:
- `--officer-data PATH`: Path to the JSON file containing the officer's data for prediction (required)
- `--model-path PATH`: Path to the trained model file (default: models/leadership_model.joblib)
- `--preprocessor-path PATH`: Path to the fitted preprocessor file (default: models/preprocessor.joblib)

Example:
```bash
# Predict using default model and preprocessor
python src/main.py --log-level DEBUG predict --officer-data data/new_officer.json

# Predict using custom model and preprocessor
python src/main.py --log-level DEBUG predict \
    --officer-data data/new_officer.json \
    --model-path models/custom_model.joblib \
    --preprocessor-path models/custom_preprocessor.joblib
```

### Generate Synthetic Data Command

Generates synthetic officer data for training purposes.

```bash
python src/main.py generate-synthetic-data [options]
```

Options:
- `--output-path PATH`: Path to save the generated synthetic data in JSONL format (required)
- `--num-officers N`: Number of synthetic officer profiles to generate (default: 100)

Example:
```bash
# Generate 100 synthetic officers (default)
python src/main.py --log-level DEBUG generate-synthetic-data --output-path data/synthetic_data/generated.jsonl

# Generate 500 synthetic officers
python src/main.py --log-level DEBUG generate-synthetic-data \
    --output-path data/synthetic_data/generated.jsonl \
    --num-officers 500
```

## Summary - Recommended Running Sequence for the new starters

1.  Generate data for model training/learning. 
* You can **choose to** generate synthetic data first.  (The app ships with few generated_ named synthetic leadership data which is used for learning, though)  

*  If you want to generate data, here's an example that generates 2,500 (two thousand five hundred) of leader data: 

   ```python
   # generate data 
   python src/main.py --log-level DEBUG generate-synthetic-data  --output-path data/synthetic_data/generated_2500.jsonl --num-officers 2500 
   ``` 

2. Train the model 

* Here's how you train the model : 

   ```python
   #train 
   python src/main.py train --data-path data/synthetic_data/generated_2500.jsonl --model-output-path models/leadership_model.joblib
   ```

* In addition to model training, this step also generates a bunch of ".png" files under "visualizations" folder 

3. Predict a (new) leader's 5 leadership competencies 

* Here's an example: 

   ```python
   #predict 
   python src/main.py predict --officer-data data/new_officer.json --model-path models/leadership_model.joblib
   ```

* The model-path is the model-output-path from the previous step
* In addition to prediction, this step also generates a single ".png" file belonging to the leader being predicted - where you can visually see the predicted values generated by the application.

