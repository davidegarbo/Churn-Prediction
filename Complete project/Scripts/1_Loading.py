"""
Churn Prediction Analysis
=========================

A comprehensive machine learning project for predicting customer churn using multiple algorithms,
feature importance analysis, and robust preprocessing.

Author: [Davide Garbo]
Date: September 2025
"""

# ---- 0) Libraries -------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
)

# Set style for better plots
sns.set_style("whitegrid")
plt.style.use('default')

# ---- 1) Load Dataset ----------------------------------------------------
def load_dataset(filename="Dataset.csv"):
    """Load the churn dataset with error handling."""
    try:
        df = pd.read_csv(filename)
        print(f"✅ Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: {filename} not found. Please ensure the file is in the current directory.")
        exit()
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        exit()

# ---- 2) Data Exploration -----------------------------------------------
def explore_dataset(df, target_col):
    """Perform initial data exploration."""
    print("\n" + "="*70)
    print("📊 DATASET OVERVIEW")
    print("="*70)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nMissing values per column:")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if len(missing_data) > 0:
        print(missing_data)
    else:
        print("No missing values found!")
    
    print(f"\nData types:")
    print(df.dtypes.value_counts())

# ---- 3) Target Processing ----------------------------------------------
def process_target(df, target_col):
    """Process and validate target column."""
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found in dataset.")
        print("Available columns:", list(df.columns))
        exit()

    print(f"\n🎯 TARGET VARIABLE: {target_col}")
    print(f"Original target values: {df[target_col].value_counts().to_dict()}")

    # Map target to 0/1
    if df[target_col].dtype == 'object':
        unique_vals = df[target_col].unique()
        if set(unique_vals).issubset({"No", "Yes"}):
            df[target_col] = df[target_col].map({"No": 0, "Yes": 1})
        elif set(unique_vals).issubset({"0", "1"}):
            df[target_col] = df[target_col].astype(int)
        else:
            # Assume first unique value is 0, second is 1
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            df[target_col] = df[target_col].map(mapping)
            print(f"Mapped {unique_vals[0]}->0, {unique_vals[1]}->1")

    print(f"Target after mapping: {df[target_col].value_counts().to_dict()}")
    
    # Drop rows with missing target
    initial_shape = df.shape[0]
    df = df.dropna(subset=[target_col]).copy()
    dropped_rows = initial_shape - df.shape[0]
    if dropped_rows > 0:
        print(f"⚠️  Rows dropped due to missing target: {dropped_rows}")
    
    return df
