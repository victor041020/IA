import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def preprocess_data(X, handle_missing=True, scale_features=True, encode_cat=True):
    """
    Preprocess the input data by handling missing values, scaling numerical features,
    and encoding categorical variables.
    """
    try:
        # Make a copy to avoid modifying the original dataframe
        X_processed = X.copy()

        # Separate numerical and categorical columns
        numerical_cols = X_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_processed.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

        # Create preprocessing pipelines
        preprocessors = []

        # Numerical features pipeline
        if numerical_cols:
            num_pipeline_steps = []

            if handle_missing:
                num_pipeline_steps.append(('imputer', SimpleImputer(strategy='mean')))

            if scale_features:
                num_pipeline_steps.append(('scaler', StandardScaler()))

            if num_pipeline_steps:
                num_pipeline = Pipeline(steps=num_pipeline_steps)
                preprocessors.append(('num', num_pipeline, numerical_cols))

        # Categorical features pipeline
        if categorical_cols and encode_cat:
            cat_pipeline_steps = []

            if handle_missing:
                cat_pipeline_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))

            cat_pipeline_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))

            cat_pipeline = Pipeline(steps=cat_pipeline_steps)
            preprocessors.append(('cat', cat_pipeline, categorical_cols))

        # Apply preprocessing
        if preprocessors:
            ct = ColumnTransformer(
                transformers=preprocessors,
                remainder='passthrough'
            )

            X_processed = pd.DataFrame(ct.fit_transform(X_processed))

        # Handle categorical columns if not one-hot encoded
        if categorical_cols and not encode_cat:
            for col in categorical_cols:
                if col in X_processed.columns:
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))

        return X_processed

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise e

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Make sure these functions are available for import
__all__ = ['preprocess_data', 'split_data']