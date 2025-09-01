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

# ---- 4) Exploratory Data Analysis --------------------------------------
def perform_eda(df, target_col):
    """Perform comprehensive exploratory data analysis."""
    print("\n" + "="*70)
    print("🔍 EXPLORATORY DATA ANALYSIS")
    print("="*70)

    # Target distribution
    plt.figure(figsize=(15,5))
    
    # Count plot
    plt.subplot(1,3,1)
    target_counts = df[target_col].value_counts()
    sns.countplot(data=df, x=target_col, palette='viridis')
    plt.title("Churn Distribution")
    plt.xlabel("Churn (0=No, 1=Yes)")
    
    # Pie chart
    plt.subplot(1,3,2)
    plt.pie(target_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%', 
            startangle=90, colors=['lightblue', 'salmon'])
    plt.title("Churn Percentage")
    
    # Churn rate
    plt.subplot(1,3,3)
    churn_rate = df[target_col].mean()
    plt.bar(['Churn Rate'], [churn_rate], color='coral', alpha=0.7)
    plt.title(f"Churn Rate: {churn_rate:.2%}")
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    
    plt.tight_layout()
    plt.show()

    print(f"📈 Churn rate: {df[target_col].mean():.2%}")
    print(f"📊 Total customers: {len(df):,}")
    print(f"🔴 Churned customers: {df[target_col].sum():,}")
    print(f"🟢 Retained customers: {(len(df) - df[target_col].sum()):,}")

    # Churn categories analysis
    if "Churn Category" in df.columns:
        plt.figure(figsize=(10,6))
        churn_cat_counts = df["Churn Category"].value_counts()
        if len(churn_cat_counts) > 0:
            sns.countplot(data=df, y="Churn Category", order=churn_cat_counts.index, palette='viridis')
            plt.title("📋 Churn Categories Distribution")
            plt.tight_layout()
            plt.show()

    # Top churn reasons
    if "Churn Reason" in df.columns:
        plt.figure(figsize=(12,8))
        churn_reason_counts = df["Churn Reason"].value_counts().head(15)
        if len(churn_reason_counts) > 0:
            sns.countplot(data=df, y="Churn Reason", order=churn_reason_counts.index, palette='viridis')
            plt.title("🎯 Top 15 Churn Reasons")
            plt.tight_layout()
            plt.show()

    # Correlation heatmap for numeric features
    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols_all) > 1:
        plt.figure(figsize=(14,10))
        correlation_matrix = df[num_cols_all].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, cmap="RdYlBu_r", center=0, 
                    annot=True, fmt='.2f', square=True, linewidths=0.5)
        plt.title("🔥 Correlation Heatmap (Numeric Features)")
        plt.tight_layout()
        plt.show()

# ---- 5) Feature Engineering --------------------------------------------
def prepare_features(df, target_col):
    """Remove leakage columns and prepare features."""
    print("\n" + "="*70)
    print("🔧 FEATURE ENGINEERING")
    print("="*70)
    
    # Define columns to remove (data leakage and IDs)
    leak_cols = ["Churn", "Churn Value", "Churn Category", "Churn Reason",
                 "Churn Score", "Customer Status"]
    id_cols = ["Customer ID", "customerID", "CustomerID", "ID", "Id"]
    
    cols_to_drop = [c for c in leak_cols + id_cols if c in df.columns]
    
    if cols_to_drop:
        print(f"🗑️  Columns being dropped (leakage/ID): {cols_to_drop}")
    else:
        print("✅ No leakage or ID columns found to drop")
    
    # Prepare features and target
    X = df.drop(columns=cols_to_drop + [target_col])
    y = df[target_col]
    
    print(f"📊 Final feature set shape: {X.shape}")
    print(f"🏷️  Features: {list(X.columns)}")
    
    return X, y

# ---- 6) Data Preprocessing ---------------------------------------------
def create_preprocessor(X_train):
    """Create preprocessing pipeline."""
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"\n🔢 Numeric columns ({len(num_cols)}): {num_cols}")
    print(f"📝 Categorical columns ({len(cat_cols)}): {cat_cols}")
    
    transformers = []
    
    if num_cols:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_transformer, num_cols))
    
    if cat_cols:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop='first'))
        ])
        transformers.append(("cat", categorical_transformer, cat_cols))
    
    if not transformers:
        print("❌ Error: No valid columns found for preprocessing!")
        exit()
    
    return ColumnTransformer(transformers=transformers, remainder='drop')

# ---- 7) Model Evaluation -----------------------------------------------
def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    """Evaluate a model pipeline and display comprehensive metrics."""
    try:
        print(f"\n🚀 Training {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Get probabilities if available
        y_prob = None
        try:
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)[:, 1]
            elif hasattr(pipeline.named_steps.get("model", pipeline), "predict_proba"):
                y_prob = pipeline.named_steps["model"].predict_proba(
                    pipeline.named_steps["preprocess"].transform(X_test)
                )[:, 1]
        except:
            pass

        print("\n" + "="*70)
        print(f"📊 {name} - RESULTS")
        print("-"*70)
        print(f"🎯 Accuracy : {accuracy_score(y_test, y_pred):.4f}")
        print(f"🔍 Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"📈 Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"⚖️  F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
        
        if y_prob is not None:
            print(f"🌟 ROC AUC  : {roc_auc_score(y_test, y_prob):.4f}")
        
        print(f"\n📋 Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=['No Churn', 'Churn'], 
                   yticklabels=['No Churn', 'Churn'],
                   ax=axes[0])
        axes[0].set_title(f"Confusion Matrix – {name}")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

        # ROC curve
        if y_prob is not None:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_ylim([0.0, 1.05])
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title(f'ROC Curve – {name}')
            axes[1].legend(loc="lower right")
        else:
            axes[1].text(0.5, 0.5, 'No probability\npredictions available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[1].transAxes, fontsize=14)
            axes[1].set_title('ROC Curve – Not Available')

        plt.tight_layout()
        plt.show()
        
        return pipeline
        
    except Exception as e:
        print(f"❌ Error evaluating {name}: {str(e)}")
        return None

# ---- 8) Feature Importance Analysis ------------------------------------
def analyze_feature_importance(pipeline, X_train, feature_names):
    """Analyze and visualize feature importance for XGBoost model."""
    try:
        model = pipeline.named_steps["model"]
        importances = model.feature_importances_
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n🏆 Top 15 XGBoost Features by Importance:")
        print(feature_importance_df.head(15).to_string(index=False, float_format='%.4f'))
        
        # Plot feature importance
        plt.figure(figsize=(12,8))
        top_features = feature_importance_df.head(15)
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title('🎯 Top 15 Feature Importances (XGBoost)')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df
        
    except Exception as e:
        print(f"❌ Error in feature importance analysis: {str(e)}")
        return None

# ---- 9) Main Analysis Function -----------------------------------------
def main():
    """Main function to run the complete churn prediction analysis."""
    print("🔮 CHURN PREDICTION ANALYSIS")
    print("="*70)
    
    # Configuration
    TARGET_COL = "Churn Label"
    
    # 1) Load dataset
    df = load_dataset()
    
    # 2) Explore dataset
    explore_dataset(df, TARGET_COL)
    
    # 3) Process target
    df = process_target(df, TARGET_COL)
    
    # 4) EDA
    perform_eda(df, TARGET_COL)
    
    # 5) Prepare features
    X, y = prepare_features(df, TARGET_COL)
    
    # 6) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 TRAIN/TEST SPLIT")
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.2%}")
    print(f"Test churn rate: {y_test.mean():.2%}")
    
    # 7) Create preprocessor
    preprocessor = create_preprocessor(X_train)
    
    # 8) Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1,
                                 random_state=42, eval_metric="logloss")
    }
    
    pipelines = {
        name: Pipeline(steps=[("preprocess", preprocessor), ("model", model)]) 
        for name, model in models.items()
    }
    
    # 9) Train and evaluate all models
    print("\n" + "="*70)
    print("🚀 MODEL TRAINING & EVALUATION")
    print("="*70)
    
    trained_pipelines = {}
    for name, pipe in pipelines.items():
        trained_pipeline = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)
        if trained_pipeline is not None:
            trained_pipelines[name] = trained_pipeline
    
    # 10) Feature importance analysis (XGBoost)
    if "XGBoost" in trained_pipelines:
        print("\n" + "="*70)
        print("🔍 FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        
        xgb_pipeline = trained_pipelines["XGBoost"]
        
        # Get feature names after preprocessing
        feature_names = []
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if num_cols:
            feature_names.extend(num_cols)
        
        if cat_cols:
            try:
                ohe = xgb_pipeline.named_steps["preprocess"].named_transformers_['cat'].named_steps['onehot']
                cat_features = ohe.get_feature_names_out(cat_cols)
                feature_names.extend(cat_features)
            except:
                pass
        
        feature_importance_df = analyze_feature_importance(xgb_pipeline, X_train, feature_names)
        
        # 11) Retrain all models after dropping most important feature
        if feature_importance_df is not None and len(feature_importance_df) > 0:
            most_influential = feature_importance_df.iloc[0]['feature']
            
            if most_influential in X.columns:
                print(f"\n" + "="*70)
                print(f"🔄 RETRAINING ALL MODELS (Dropping: '{most_influential}')")
                print("="*70)
                
                X_dropped = X.drop(columns=[most_influential])
                X_train2, X_test2, y_train2, y_test2 = train_test_split(
                    X_dropped, y, test_size=0.2, random_state=42, stratify=y
                )
                
                print(f"📊 New feature set shape: {X_dropped.shape}")
                
                # Create new preprocessor
                preprocessor2 = create_preprocessor(X_train2)
                
                # Create new pipelines
                models2 = {
                    "Logistic Regression (no top feature)": LogisticRegression(max_iter=1000, random_state=42),
                    "Random Forest (no top feature)": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                    "XGBoost (no top feature)": XGBClassifier(n_estimators=100, learning_rate=0.1,
                                                             random_state=42, eval_metric="logloss")
                }
                
                pipelines2 = {
                    name: Pipeline(steps=[("preprocess", preprocessor2), ("model", model)]) 
                    for name, model in models2.items()
                }
                
                # Train and evaluate all models with dropped feature
                trained_pipelines2 = {}
                for name, pipe in pipelines2.items():
                    trained_pipeline = evaluate_model(name, pipe, X_train2, X_test2, y_train2, y_test2)
                    if trained_pipeline is not None:
                        trained_pipelines2[name] = trained_pipeline
                
                # Show updated feature importance for XGBoost
                if "XGBoost (no top feature)" in trained_pipelines2:
                    print(f"\n🔍 Updated Feature Importance (after dropping '{most_influential}')")
                    
                    xgb_pipeline_new = trained_pipelines2["XGBoost (no top feature)"]
                    
                    # Get new feature names
                    feature_names2 = []
                    num_cols2 = X_train2.select_dtypes(include=[np.number]).columns.tolist()
                    cat_cols2 = X_train2.select_dtypes(exclude=[np.number]).columns.tolist()
                    
                    if num_cols2:
                        feature_names2.extend(num_cols2)
                    if cat_cols2:
                        try:
                            ohe2 = xgb_pipeline_new.named_steps["preprocess"].named_transformers_['cat'].named_steps['onehot']
                            cat_features2 = ohe2.get_feature_names_out(cat_cols2)
                            feature_names2.extend(cat_features2)
                        except:
                            pass
                    
                    feature_importance_df2 = analyze_feature_importance(xgb_pipeline_new, X_train2, feature_names2)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print("🎉 Thank you for using the Churn Prediction Analysis tool!")

# ---- 10) Run Analysis --------------------------------------------------
if __name__ == "__main__":
    main()