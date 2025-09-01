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