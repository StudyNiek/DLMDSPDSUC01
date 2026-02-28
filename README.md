# Heart failure Prediction - Machine Learning Pipeline

This notebook uses machine learning to predict a deatch event of Indian coronary heart disease patients. Clinical data from the UCI Machine Learning Repository is used for these predictions. This notebook was created for the course DLMDSPDSUC01, as part of an portfolio assignement. AI in the form of LLM has been used for creating the notebook and this README for debugging and readibility. 

My initial idea was to use just a random forest model to make the prediction, in the feedback of phase two I received the suggestion to include a logistic regression model as a baseline to compare the random forest model to. An other suggestion was to include SHAP values, to determined which features contribute the most to the prediciton of the model.

From the beeswarm plot, it becomes evident that Time is the most influential feature in the model. This variable represents the duration between two appointments until either a death event occurs or not.

Lower values of Time, indicated by blue dots in the plot, are associated with a higher impact on predicting death. In other words, shorter intervals between appointments contribute more strongly to a death prediction.

This finding is consistent with clinical expectations. Patients with more severe or rapidly progressing symptoms are likely to require more frequent specialist visits, resulting in shorter time intervals between appointments. In contrast, patients with milder symptoms generally require less frequent monitoring, leading to longer time intervals and a lower predicted risk of death.


##  Dataset

**Source:** UCI ML Repository - Heart Failure Clinical Records (ID: 519)
- **Samples:** ~299 patients
- **Features:** 12 clinical measurements
- **Target:** Death event (binary: 0 = Survival, 1 = Death)

### Features

| Feature | Type | Description |
|---------|------|-------------|
| age | int | Patient age (years) |
| anaemia | binary | Decrease of red blood cells or hemoglobin |
| creatinine_phosphokinase | int | Level of CPK enzyme in blood (mcg/L) |
| diabetes | binary | Patient has diabetes |
| ejection_fraction | int | Percentage of blood leaving heart each contraction (%) |
| high_blood_pressure | binary | Patient has hypertension |
| platelets | float | Platelets in blood (kiloplatelets/mL) |
| serum_creatinine | float | Level of serum creatinine in blood (mg/dL) |
| serum_sodium | int | Level of serum sodium in blood (mEq/L) |
| sex | binary | Woman (0) or man (1) |
| smoking | binary | Patient smokes |
| time | int | Follow-up period (days) |

## Objective

Build a production-ready machine learning pipeline to predict patient mortality from heart failure using:
- **Preprocessing:** MinMaxScaler normalization
- **Baseline Model:** Logistic Regression
- **Main Model:** Random Forest Classifier
- **Optimization:** Two-stage hyperparameter tuning (RandomizedSearchCV → GridSearchCV)
- **Evaluation:** ROC-AUC, Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## Notebook Structure

### 1. Import Libraries and Dataset
- Load required packages (sklearn, pandas, numpy, matplotlib, seaborn, shap)
- Fetch dataset from UCI ML Repository
- Prepare features and target

### 2. Split Data
- Stratified train-test split (80/20)
- Maintains class distribution
- Random state for reproducibility

### 3. Create, Train & Evaluate Pipelines
- **Logistic Regression Pipeline:** Baseline model
- **Random Forest Pipeline:** Primary model
- Integrated MinMaxScaler for feature normalization [0, 1]
- Initial performance comparison

### 4. Hyperparameter Optimization

#### 4.1 Random Parameter Optimization (RandomizedSearchCV)
- Explores wide parameter space efficiently
- 100 iterations with 5-fold cross-validation
- Scoring metric: ROC-AUC
- Parameters tuned:
  - `n_estimators`: [50-500]
  - `max_depth`: [None, 5, 10, 15, 20, 25, 30]
  - `min_samples_split`: [2-20]
  - `min_samples_leaf`: [1-10]
  - `max_features`: ['sqrt', 'log2', None]
  - `bootstrap`: [True, False]
  - `class_weight`: ['balanced', 'balanced_subsample', None]

#### 4.2 Exhaustive Grid Search (GridSearchCV)
- Fine-tunes around RandomizedSearchCV results
- ±10% parameter range refinement
- 5-fold cross-validation
- Maximizes ROC-AUC

#### 4.3 Save Best Pipeline
- Serializes optimized pipeline with `joblib`
- Saves as `heart_failure_pipeline.pkl`
- Includes both scaler and trained model

### 5. Load Pipeline & Test
- Demonstrates loading saved model
- Makes predictions on test samples
- Shows probability scores

### 6. Visualize Model Comparison
- ROC curves for all models (Logistic Regression, RF Baseline, RF Optimized)
- AUC comparison
- Performance visualization

## For running this code:

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap joblib ucimlrepo scipy
```
### Run the Notebook

1. Open `DLMDSPDSUC01.ipynb` in Jupyter or VS Code
2. Run all cells sequentially
3. Wait for hyperparameter optimization (may take several minutes)
4. Review results and saved model

### Expected Outputs

- **Model Performance Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC
- **Saved Model:** `heart_failure_pipeline.pkl`
- **Visualizations:** ROC curves comparing all models
- **Best Parameters:** Optimized hyperparameters printed to console

## Model Performance

The notebook trains and evaluates three models:

1. **Logistic Regression** (Baseline)
   - Fast training
   - Interpretable coefficients
   - Linear decision boundary

2. **Random Forest - Base** (Default parameters)
   - Ensemble of decision trees
   - Non-linear patterns
   - No hyperparameter tuning

3. **Random Forest - Optimized** (Tuned)
   - Two-stage optimization
   - Best ROC-AUC performance
   - Production-ready

### Evaluation Metrics

- **Accuracy:** Overall correctness
- **Precision:** Positive predictive value
- **Recall (Sensitivity):** True positive rate
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under ROC curve (primary metric)

## Model Deployment

The saved pipeline (`heart_failure_pipeline.pkl`) can be:
- Loaded in production environments
- Integrated into web APIs (see `app.py`)
- Used for batch predictions
- Deployed in Docker containers

### Loading the Model

```python
import joblib

# Load pipeline
pipeline = joblib.load('heart_failure_pipeline.pkl')

# Make prediction
prediction = pipeline.predict(new_data)
probability = pipeline.predict_proba(new_data)[:, 1]
```

