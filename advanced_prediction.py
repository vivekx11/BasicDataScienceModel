import pandas as pd
import joblib
import numpy as np

print("="*80)
print(" "*20 + "ADVANCED SALARY PREDICTION SYSTEM")
print("="*80)

# Load model and encoders
model = joblib.load('best_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

print("\n✓ Model loaded successfully!")

# Load original data for reference
df = pd.read_csv('employee_company_clean_data.csv')
df_clean = df[df['salary'] >= 0]

# Get unique values for each category
companies = sorted(df['company_name'].unique())
departments = sorted(df['department'].unique())
cities = sorted(df['city'].unique())

print("\n" + "="*80)
print("AVAILABLE OPTIONS")
print("="*80)

print("\nCompanies:", ", ".join(companies))
print("\nDepartments:", ", ".join(departments))
print("\nCities:", ", ".join(cities))
print("\nAge Range: 22-45 years")

def predict_salary_advanced(age, company_name, department, city):
    """Advanced salary prediction with confidence intervals"""
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'company_name': [company_name],
        'department': [department],
        'city': [city]
    })
    
    # Encode categorical variables
    for col in ['company_name', 'department', 'city']:
        try:
            input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
        except ValueError:
            input_data[col] = 0
    
    # Make prediction
    predicted_salary = model.predict(input_data)[0]
    
    # Calculate statistics from similar profiles
    similar = df_clean[
        (df_clean['age'] == age) &
        (df_clean['company_name'] == company_name) &
        (df_clean['department'] == department) &
        (df_clean['city'] == city)
    ]
    
    stats = {
        'predicted': predicted_salary,
        'similar_count': len(similar),
        'similar_mean': similar['salary'].mean() if len(similar) > 0 else None,
        'similar_min': similar['salary'].min() if len(similar) > 0 else None,
        'similar_max': similar['salary'].max() if len(similar) > 0 else None
    }
    
    return stats

def batch_predict(csv_file):
    """Predict salaries for multiple employees from CSV"""
    
    print(f"\n📂 Loading data from {csv_file}...")
    data = pd.read_csv(csv_file)
    
    predictions = []
    for idx, row in data.iterrows():
        stats = predict_salary_advanced(
            row['age'],
            row['company_name'],
            row['department'],
            row['city']
        )
        predictions.append(stats['predicted'])
    
    data['predicted_salary'] = predictions
    output_file = 'predictions_output.csv'
    data.to_csv(output_file, index=False)
    
    print(f"✓ Predictions saved to {output_file}")
    return data

# Example predictions with detailed analysis
print("\n" + "="*80)
print("EXAMPLE PREDICTIONS WITH ANALYSIS")
print("="*80)

test_cases = [
    {"age": 30, "company_name": "TechNova", "department": "IT", "city": "Bangalore"},
    {"age": 25, "company_name": "DataWorks", "department": "HR", "city": "Mumbai"},
    {"age": 40, "company_name": "InnoSoft", "department": "Sales", "city": "Delhi"},
    {"age": 35, "company_name": "CloudNine", "department": "Finance", "city": "Hyderabad"},
    {"age": 45, "company_name": "NextGen", "department": "Marketing", "city": "Pune"},
]

for i, case in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"PREDICTION #{i}")
    print(f"{'='*80}")
    
    stats = predict_salary_advanced(**case)
    
    print(f"\n📋 Employee Profile:")
    print(f"   Age: {case['age']} years")
    print(f"   Company: {case['company_name']}")
    print(f"   Department: {case['department']}")
    print(f"   City: {case['city']}")
    
    print(f"\n💰 Predicted Salary: ₹{stats['predicted']:,.0f}")
    
    if stats['similar_count'] > 0:
        print(f"\n📊 Similar Profiles in Dataset: {stats['similar_count']}")
        print(f"   Average Salary: ₹{stats['similar_mean']:,.0f}")
        print(f"   Salary Range: ₹{stats['similar_min']:,.0f} - ₹{stats['similar_max']:,.0f}")
    else:
        print(f"\n⚠️  No exact matches found in dataset")

# Salary comparison by different factors
print("\n" + "="*80)
print("SALARY COMPARISON ANALYSIS")
print("="*80)

print("\n--- Average Salary by Company ---")
company_avg = df_clean.groupby('company_name')['salary'].mean().sort_values(ascending=False)
for company, salary in company_avg.items():
    print(f"{company:15s}: ₹{salary:,.0f}")

print("\n--- Average Salary by Department ---")
dept_avg = df_clean.groupby('department')['salary'].mean().sort_values(ascending=False)
for dept, salary in dept_avg.items():
    print(f"{dept:15s}: ₹{salary:,.0f}")

print("\n--- Average Salary by City ---")
city_avg = df_clean.groupby('city')['salary'].mean().sort_values(ascending=False)
for city, salary in city_avg.items():
    print(f"{city:15s}: ₹{salary:,.0f}")

print("\n--- Average Salary by Age ---")
age_avg = df_clean.groupby('age')['salary'].mean().sort_values(ascending=False)
for age, salary in age_avg.items():
    print(f"{age:.0f} years: ₹{salary:,.0f}")

# Interactive prediction function
print("\n" + "="*80)
print("INTERACTIVE PREDICTION")
print("="*80)

def interactive_predict():
    """Interactive salary prediction"""
    print("\nEnter employee details for salary prediction:")
    print("(Press Ctrl+C to exit)")
    
    try:
        age = int(input("\nAge (22-45): "))
        if age < 22 or age > 45:
            print("⚠️  Age should be between 22 and 45")
            return
        
        print(f"\nAvailable companies: {', '.join(companies)}")
        company = input("Company name: ")
        
        print(f"\nAvailable departments: {', '.join(departments)}")
        department = input("Department: ")
        
        print(f"\nAvailable cities: {', '.join(cities)}")
        city = input("City: ")
        
        stats = predict_salary_advanced(age, company, department, city)
        
        print("\n" + "="*80)
        print("PREDICTION RESULT")
        print("="*80)
        print(f"\n💰 Predicted Salary: ₹{stats['predicted']:,.0f}")
        
        if stats['similar_count'] > 0:
            print(f"\n📊 Based on {stats['similar_count']} similar profiles")
            print(f"   Average: ₹{stats['similar_mean']:,.0f}")
            print(f"   Range: ₹{stats['similar_min']:,.0f} - ₹{stats['similar_max']:,.0f}")
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")

print("\n📝 To use interactive prediction, call: interactive_predict()")
print("📊 To predict from CSV file, call: batch_predict('your_file.csv')")

print("\n" + "="*80)
print("SYSTEM READY FOR PREDICTIONS!")
print("="*80)
