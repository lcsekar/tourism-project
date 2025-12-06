# for data manipulation
import pandas as pd
import sklearn

# for creating a folder
import os

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for converting text data in to numerical representation
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/lcsekar/tourism-project-data/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

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
# train test data split
# ================================================================
# X contains the features and y contains the target variable
X = df[numeric_cols + ordinal_feature_names + nominal_cols + binary_cols + gender_col]
y = df[target]

# 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train test split completed successfully.")

# ================================================================
# save to csv after split
# ================================================================
X_train.to_csv("X_train.csv",index=False)
X_test.to_csv("X_test.csv",index=False)
y_train.to_csv("y_train.csv",index=False)
y_test.to_csv("y_test.csv",index=False)
print("Data preprocessing completed successfully.")

# ================================================================
# upload to HF
# ================================================================
files = ["X_train.csv","X_test.csv","y_train.csv","y_test.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="lcsekar/tourism-project-data",
        repo_type="dataset",
    )
print("Files uploaded to HF successfully.")
