import pandas as pd

# Dataset load karo
df = pd.read_csv("employee_company_clean_data.csv")

# 1️⃣ Total Employees
print("Total Employees:", df.shape[0])

# 2️⃣ Average Age
print("Average Age:", df["age"].mean())

# 3️⃣ Average Salary
print("Average Salary:", df["salary"].mean())

# 4️⃣ Minimum aur Maximum Salary
print("Minimum Salary:", df["salary"].min())
print("Maximum Salary:", df["salary"].max())

# 5️⃣ Most Common Department
print("\nEmployees per Department:")
print(df["department"].value_counts())

# 6️⃣ City Distribution
print("\nEmployees per City:")
print(df["city"].value_counts())

# 7️⃣ Company Distribution
print("\nEmployees per Company:")
print(df["company_name"].value_counts())

# 8️⃣ Age Distribution
print("\nAge Distribution:")
print(df["age"].value_counts())