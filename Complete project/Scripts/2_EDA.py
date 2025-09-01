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