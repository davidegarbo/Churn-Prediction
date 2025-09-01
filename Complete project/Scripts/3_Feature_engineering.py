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
