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