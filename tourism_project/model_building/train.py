# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline

# for model training, tuning, and evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# for model serialization
import joblib

# for creating a folder
import os

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
# import mlflow

# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

# ================================================================
# download data files from HF
# ================================================================
X_train_path = "hf://datasets/lcsekar/tourism-project-data/X_train.csv"
X_test_path = "hf://datasets/lcsekar/tourism-project-data/X_test.csv"
y_train_path = "hf://datasets/lcsekar/tourism-project-data/y_train.csv"
y_test_path = "hf://datasets/lcsekar/tourism-project-data/y_test.csv"

X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path)
y_test = pd.read_csv(y_test_path)
print("Data files loaded from HF successfully.")

# ================================================================
# catergorizing columns
# ================================================================
# target
target = "ProdTaken"

# columns to be dropped
drop_cols = ["Unnamed: 0", "CustomerID"]

# numerical columns
numeric_cols = [
    "Age", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
    "NumberOfTrips", "NumberOfChildrenVisiting", "MonthlyIncome"
]

# ordinal columns
ordinal_cols = {
    "CityTier": [1, 2, 3],
    "PreferredPropertyStar": [3., 4., 5.],
    "PitchSatisfactionScore": [1, 2, 3, 4, 5],
    "Designation": ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
}

ordinal_feature_names = list(ordinal_cols.keys())
ordinal_categories = list(ordinal_cols.values())

# nominal columns
nominal_cols = [
    "TypeofContact", "Occupation",
    "ProductPitched", "MaritalStatus"
]

# binary columns
binary_cols = ["ProdTaken", "Passport", "OwnCar"]

# gender is also a binary column, but it needs special treatment
gender_col = ["Gender"]

# ================================================================
# define column transformers
# ================================================================
# numeric transformer
numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

# ordinal transformer
ordinal_transformer = Pipeline([
    ("ordinal", OrdinalEncoder(categories=ordinal_categories))
])

# nominal transformer
nominal_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# binary transformer
binary_transformer = "passthrough"

# gender column transformer
def gender_cleaner(X):
    return X.iloc[:, 0].str.replace("Fe Male", "Female", regex=False).to_frame()

gender_transformer = Pipeline([
    ("clean", FunctionTransformer(gender_cleaner, validate=False)),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# assemble column preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("ord", ordinal_transformer, ordinal_feature_names),
        ("nom", nominal_transformer, nominal_cols),
        ("gender", gender_transformer, gender_col),
        ("bin", binary_transformer, binary_cols)
    ],
    remainder="drop"
)

# Creating a Random Forest Regressor model
model = RandomForestClassifier(random_state=42)

# hyper-parameter grid
param_grid = {
    "randomforestclassifier__n_estimators": [100, 200, 300],
    "randomforestclassifier__max_depth": [None, 5, 10, 15, 20],
    "randomforestclassifier__max_features": ["sqrt", "log2", None],
    "randomforestclassifier__min_samples_split": [2, 5, 10],
    "randomforestclassifier__min_samples_leaf": [1, 2, 4]
}

# Creating a pipeline with preprocessor and model
model_pipeline = make_pipeline(preprocessor, model)

# Fitting the model on the training data
print("Starting model training...")
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Model training completed.")

# best model
best_model = grid_search.best_estimator_

# with mlflow.start_run():
#     # Grid Search
#     grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
#     grid_search.fit(Xtrain, ytrain)

#     # Log parameter sets
#     results = grid_search.cv_results_
#     for i in range(len(results['params'])):
#         param_set = results['params'][i]
#         mean_score = results['mean_test_score'][i]

#         with mlflow.start_run(nested=True):
#             mlflow.log_params(param_set)
#             mlflow.log_metric("mean_neg_mse", mean_score)

#     # Best model
#     mlflow.log_params(grid_search.best_params_)
#     best_model = grid_search.best_estimator_

#     # Predictions
#     y_pred_train = best_model.predict(Xtrain)
#     y_pred_test = best_model.predict(Xtest)

#     # Metrics
#     train_rmse = mean_squared_error(ytrain, y_pred_train, squared=False)
#     test_rmse = mean_squared_error(ytest, y_pred_test, squared=False)

#     train_mae = mean_absolute_error(ytrain, y_pred_train)
#     test_mae = mean_absolute_error(ytest, y_pred_test)

#     train_r2 = r2_score(ytrain, y_pred_train)
#     test_r2 = r2_score(ytest, y_pred_test)

#     # Log metrics
#     mlflow.log_metrics({
#         "train_RMSE": train_rmse,
#         "test_RMSE": test_rmse,
#         "train_MAE": train_mae,
#         "test_MAE": test_mae,
#         "train_R2": train_r2,
#         "test_R2": test_r2
#     })

# Save the model locally
model_path = "best_model_v1.joblib"
joblib.dump(best_model, model_path)

# Log the model artifact
# mlflow.log_artifact(model_path, artifact_path="model")
print(f"Model saved as artifact at: {model_path}")

# Upload to Hugging Face
repo_id = "lcsekar/tourism-project-model"
repo_type = "model"

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# create_repo("churn-model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj="best_model_v1.joblib",
    path_in_repo="best_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Model uploaded to Hugging Face Hub at repo: {repo_id}")
