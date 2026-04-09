import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print(" "*20 + "COMPLETE DATA SCIENCE PIPELINE")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n" + "="*80)
print("STEP 1: DATA LOADING")
print("="*80)

df = pd.read_csv('employee_company_clean_data.csv')
print(f"\n✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80)

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n--- Unique Values per Column ---")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# ============================================================================
# 3. DATA VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("STEP 3: DATA VISUALIZATION")
print("="*80)

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Age Distribution
axes[0, 0].hist(df['age'].dropna(), bins=20, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Age Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# 2. Salary Distribution
axes[0, 1].hist(df[df['salary'] >= 0]['salary'], bins=30, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Salary Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Salary')
axes[0, 1].set_ylabel('Frequency')

# 3. Company Distribution
company_counts = df['company_name'].value_counts().head(10)
axes[0, 2].barh(company_counts.index, company_counts.values, color='coral')
axes[0, 2].set_title('Top 10 Companies', fontsize=14, fontweight='bold')
axes[0, 2].set_xlabel('Count')

# 4. Department Distribution
dept_counts = df['department'].value_counts().head(10)
axes[1, 0].bar(dept_counts.index, dept_counts.values, color='plum')
axes[1, 0].set_title('Department Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Department')
axes[1, 0].set_ylabel('Count')
axes[1, 0].tick_params(axis='x', rotation=45)

# 5. City Distribution
city_counts = df['city'].value_counts().head(10)
axes[1, 1].bar(city_counts.index, city_counts.values, color='gold')
axes[1, 1].set_title('City Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('City')
axes[1, 1].set_ylabel('Count')
axes[1, 1].tick_params(axis='x', rotation=45)

# 6. Age vs Salary
df_positive = df[df['salary'] >= 0]
axes[1, 2].scatter(df_positive['age'], df_positive['salary'], alpha=0.5, color='purple')
axes[1, 2].set_title('Age vs Salary', fontsize=14, fontweight='bold')
axes[1, 2].set_xlabel('Age')
axes[1, 2].set_ylabel('Salary')

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved as 'eda_visualizations.png'")

# Additional Analysis
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

# Salary by Department
df_clean = df[df['salary'] >= 0].copy()
dept_salary = df_clean.groupby('department')['salary'].mean().sort_values(ascending=False).head(10)
axes2[0].barh(dept_salary.index, dept_salary.values, color='teal')
axes2[0].set_title('Average Salary by Department', fontsize=14, fontweight='bold')
axes2[0].set_xlabel('Average Salary')

# Salary by City
city_salary = df_clean.groupby('city')['salary'].mean().sort_values(ascending=False).head(10)
axes2[1].barh(city_salary.index, city_salary.values, color='orange')
axes2[1].set_title('Average Salary by City', fontsize=14, fontweight='bold')
axes2[1].set_xlabel('Average Salary')

plt.tight_layout()
plt.savefig('salary_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Salary analysis saved as 'salary_analysis.png'")

# ============================================================================
# 4. DATA PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("STEP 4: DATA PREPROCESSING")
print("="*80)

# Handle missing values
df['employee_id'] = df['employee_id'].fillna(df['employee_id'].median())
df['age'] = df['age'].fillna(df['age'].median())
df['salary'] = df['salary'].fillna(df['salary'].median())
df['company_name'] = df['company_name'].fillna('Unknown')
df['department'] = df['department'].fillna('Unknown')
df['city'] = df['city'].fillna('Unknown')

print("\n✓ Missing values handled")

# Remove negative salaries
df_clean = df[df['salary'] >= 0].copy()
print(f"✓ Removed negative salaries. New shape: {df_clean.shape}")

# Feature Engineering
df_clean['salary_category'] = pd.cut(df_clean['salary'], 
                                      bins=[0, 40000, 60000, 80000, 100000],
                                      labels=['Low', 'Medium', 'High', 'Very High'])

print("✓ Feature engineering completed")

# ============================================================================
# 5. FEATURE SELECTION & ENCODING
# ============================================================================
print("\n" + "="*80)
print("STEP 5: FEATURE SELECTION & ENCODING")
print("="*80)

features = ['age', 'company_name', 'department', 'city']
target = 'salary'

X = df_clean[features].copy()
y = df_clean[target].copy()

# Encode categorical variables
label_encoders = {}
for col in ['company_name', 'department', 'city']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"\n✓ Features encoded")
print(f"  Feature shape: {X.shape}")
print(f"  Target shape: {y.shape}")

# ============================================================================
# 6. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("STEP 6: TRAIN-TEST SPLIT")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n✓ Data split completed")
print(f"  Training set: {X_train.shape}")
print(f"  Test set: {X_test.shape}")

# ============================================================================
# 7. MODEL TRAINING & COMPARISON
# ============================================================================
print("\n" + "="*80)
print("STEP 7: MODEL TRAINING & COMPARISON")
print("="*80)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

results = []

print("\nTraining models...\n")
for name, model in models.items():
    print(f"Training {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'CV R² Mean': cv_scores.mean(),
        'CV R² Std': cv_scores.std()
    })
    
    print(f"  ✓ {name} completed")

# ============================================================================
# 8. MODEL COMPARISON RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 8: MODEL COMPARISON RESULTS")
print("="*80)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Find best model
best_model_idx = results_df['Test R²'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test R² Score: {results_df.loc[best_model_idx, 'Test R²']:.4f}")

# Save results
results_df.to_csv('model_comparison_results.csv', index=False)
print("\n✓ Results saved as 'model_comparison_results.csv'")

# ============================================================================
# 9. MODEL VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("STEP 9: MODEL PERFORMANCE VISUALIZATION")
print("="*80)

fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))

# R² Score Comparison
axes3[0, 0].barh(results_df['Model'], results_df['Test R²'], color='steelblue')
axes3[0, 0].set_title('Model Comparison - R² Score', fontsize=14, fontweight='bold')
axes3[0, 0].set_xlabel('R² Score')
axes3[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=1)

# RMSE Comparison
axes3[0, 1].barh(results_df['Model'], results_df['Test RMSE'], color='coral')
axes3[0, 1].set_title('Model Comparison - RMSE', fontsize=14, fontweight='bold')
axes3[0, 1].set_xlabel('RMSE (Lower is Better)')

# MAE Comparison
axes3[1, 0].barh(results_df['Model'], results_df['Test MAE'], color='lightgreen')
axes3[1, 0].set_title('Model Comparison - MAE', fontsize=14, fontweight='bold')
axes3[1, 0].set_xlabel('MAE (Lower is Better)')

# Train vs Test R²
x_pos = np.arange(len(results_df))
width = 0.35
axes3[1, 1].bar(x_pos - width/2, results_df['Train R²'], width, label='Train R²', color='skyblue')
axes3[1, 1].bar(x_pos + width/2, results_df['Test R²'], width, label='Test R²', color='orange')
axes3[1, 1].set_title('Train vs Test R² Score', fontsize=14, fontweight='bold')
axes3[1, 1].set_xlabel('Model')
axes3[1, 1].set_ylabel('R² Score')
axes3[1, 1].set_xticks(x_pos)
axes3[1, 1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
axes3[1, 1].legend()
axes3[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Model comparison saved as 'model_comparison.png'")

# ============================================================================
# 10. SAVE BEST MODEL
# ============================================================================
print("\n" + "="*80)
print("STEP 10: SAVING BEST MODEL")
print("="*80)

best_model = models[best_model_name]
best_model.fit(X_train, y_train)

joblib.dump(best_model, 'best_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print(f"\n✓ Best model ({best_model_name}) saved as 'best_model.pkl'")
print("✓ Label encoders saved as 'label_encoders.pkl'")

# ============================================================================
# 11. FEATURE IMPORTANCE (if applicable)
# ============================================================================
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "="*80)
    print("STEP 11: FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n" + feature_importance.to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='purple')
    plt.title('Feature Importance', fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature importance saved as 'feature_importance.png'")

# ============================================================================
# 12. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"""
✓ Dataset: {df.shape[0]} rows, {df.shape[1]} columns
✓ Clean data: {df_clean.shape[0]} rows (after removing negative salaries)
✓ Models trained: {len(models)}
✓ Best model: {best_model_name}
✓ Best Test R² Score: {results_df.loc[best_model_idx, 'Test R²']:.4f}
✓ Best Test RMSE: {results_df.loc[best_model_idx, 'Test RMSE']:.2f}
✓ Best Test MAE: {results_df.loc[best_model_idx, 'Test MAE']:.2f}

Files Generated:
  1. eda_visualizations.png - Exploratory data analysis plots
  2. salary_analysis.png - Salary analysis by department and city
  3. model_comparison.png - Model performance comparison
  4. feature_importance.png - Feature importance plot
  5. model_comparison_results.csv - Detailed results table
  6. best_model.pkl - Trained best model
  7. label_encoders.pkl - Label encoders for prediction
""")

print("="*80)
print(" "*25 + "PIPELINE COMPLETED!")
print("="*80)
