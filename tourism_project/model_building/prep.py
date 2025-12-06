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
# apply preprocessor
# ================================================================
preprocessor.set_output(transform="pandas")
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

X_train_transformed.to_csv("X_train.csv",index=False)
X_test_transformed.to_csv("X_test.csv",index=False)
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
