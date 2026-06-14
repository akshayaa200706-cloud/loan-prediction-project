import pandas as pd

# Load dataset
df = pd.read_csv("data/archive/train_u6lujuX_CVtuZ9i.csv")
# Display first 5 rows
print(df.head())
import pandas as pd

# Load dataset
df = pd.read_csv("data/archive/train_u6lujuX_CVtuZ9i.csv")

# First 5 rows
print("First 5 Rows:")
print(df.head())

# Shape
print("\nDataset Shape:")
print(df.shape)

# Column Names
print("\nColumns:")
print(df.columns)

# Information
print("\nDataset Info:")
print(df.info())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())