import pandas as pd

# Step 1: Dataset load karo...
df = pd.read_csv("raw.csv")

# Step 2: Raw data preview
print("First 5 rows:")
print(df.head())

# Step 3: Dataset info
print("\nDataset Info:")
print(df.info())

# Step 4: Missing values check
print("\nMissing values:")
print(df.isnull().sum())

# Step 5: Duplicate rows check
print("\nDuplicate rows:", df.duplicated().sum())

print("break")

# Step 6: Duplicate remove karo
df = df.drop_duplicates()

# Step 7: Text columns se extra spaces remove
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Step 8: Numeric columns ki missing values fill karo
#df["age"] = df["age"].fillna(df["age"].mean())
df["age"] = df["age"].round()
df = df[(df["age"] > 18) & (df["age"] < 65)]
df["salary"] = df["salary"].fillna(df["salary"].mean())

# Step 9: Text columns ki missing values fill karo
df["department"] = df["department"].fillna("Unknown")
df["city"] = df["city"].fillna("Unknown")
df["company_name"] = df["company_name"].fillna("Unknown")

# Step 10: Important fields missing ho to row remove
df = df.dropna(subset=["employee_name", "email"])

# Step 11: Date format correct karo
df["joining_date"] = pd.to_datetime(df["joining_date"], errors="coerce")

# Step 12: Final null check
print("\nFinal missing values:")
print(df.isnull().sum())

# Step 13: Clean dataset save karo
df.to_csv("employee_company_clean_data.csv", index=False)

print("\n✅ Data cleaning complete!")
print("Clean file saved: employee_company_clean_data.csv")
