import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Load clean dataset
print("Loading clean dataset...")
df = pd.read_csv('employee_company_clean_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Data preprocessing
print("\n" + "="*50)
print("Data Preprocessing...")
print("="*50)

# Handle missing values
df['employee_id'].fillna(df['employee_id'].median(), inplace=True)
df['age'].fillna(df['age'].median(), inplace=True)
df['salary'].fillna(df['salary'].median(), inplace=True)
df['company_name'].fillna('Unknown', inplace=True)
df['department'].fillna('Unknown', inplace=True)
df['city'].fillna('Unknown', inplace=True)

# Remove negative salaries (data quality issue)
df = df[df['salary'] >= 0]

print(f"After cleaning: {df.shape}")

# Select features for model
features = ['age', 'company_name', 'department', 'city']
target = 'salary'

X = df[features].copy()
y = df[target].copy()

# Encode categorical variables
print("\nEncoding categorical variables...")
label_encoders = {}
for col in ['company_name', 'department', 'city']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Split data
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train model
print("\n" + "="*50)
print("Training Random Forest Model...")
print("="*50)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Make predictions
print("\nMaking predictions...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate model
print("\n" + "="*50)
print("Model Performance")
print("="*50)

print("\nTraining Set:")
print(f"R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f}")
print(f"MAE: {mean_absolute_error(y_train, y_pred_train):.2f}")

print("\nTest Set:")
print(f"R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_test):.2f}")

# Feature importance
print("\n" + "="*50)
print("Feature Importance")
print("="*50)
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# Save model and encoders
print("\n" + "="*50)
print("Saving Model...")
print("="*50)

joblib.dump(model, 'salary_prediction_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("✓ Model saved as 'salary_prediction_model.pkl'")
print("✓ Encoders saved as 'label_encoders.pkl'")

# Sample predictions
print("\n" + "="*50)
print("Sample Predictions")
print("="*50)

sample_indices = np.random.choice(X_test.index, 5, replace=False)
for idx in sample_indices:
    actual = y_test.loc[idx]
    predicted = model.predict(X_test.loc[[idx]])[0]
    print(f"\nActual: ₹{actual:,.0f} | Predicted: ₹{predicted:,.0f} | Difference: ₹{abs(actual-predicted):,.0f}")

print("\n" + "="*50)
print("Training Complete!")
print("="*50)
