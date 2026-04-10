import pandas as pd
import joblib

# Load trained model and encoders.....
print("Loading trained model...")
model = joblib.load('salary_prediction_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

print("Model loaded successfully!\n")

# Function to predict salary
def predict_salary(age, company_name, department, city):
    """
    Predict salary based on employee features
    
    Parameters:
    - age: Employee age (22-45)
    - company_name: Company name (TechNova, DataWorks, InnoSoft, NextGen, CloudNine, Unknown)
    - department: Department (IT, HR, Sales, Marketing, Finance, Unknown)
    - city: City (Delhi, Mumbai, Bangalore, Chennai, Hyderabad, Pune, Unknown)
    """
    
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
            # If unknown category, use most frequent
            input_data[col] = 0
    
    # Make prediction
    predicted_salary = model.predict(input_data)[0]
    
    return predicted_salary

# Example predictions
print("="*60)
print("Example Salary Predictions")
print("="*60)

examples = [
    {"age": 30, "company_name": "TechNova", "department": "IT", "city": "Bangalore"},
    {"age": 25, "company_name": "DataWorks", "department": "HR", "city": "Mumbai"},
    {"age": 40, "company_name": "InnoSoft", "department": "Sales", "city": "Delhi"},
    {"age": 35, "company_name": "CloudNine", "department": "Finance", "city": "Hyderabad"},
    {"age": 22, "company_name": "NextGen", "department": "Marketing", "city": "Pune"},
]

for i, example in enumerate(examples, 1):
    salary = predict_salary(**example)
    print(f"\n{i}. Employee Profile:")
    print(f"   Age: {example['age']}")
    print(f"   Company: {example['company_name']}")
    print(f"   Department: {example['department']}")
    print(f"   City: {example['city']}")
    print(f"   → Predicted Salary: ₹{salary:,.0f}")

print("\n" + "="*60)
print("\nYou can use the predict_salary() function to predict")
print("salary for any employee with these features!")
print("="*60)
