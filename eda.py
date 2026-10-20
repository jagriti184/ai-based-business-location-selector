
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("data/business_locations.csv")

print("DATASET SHAPE:", df.shape)
print("\nFIRST FIVE RECORDS:")
print(df.head())

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Check duplicate records
print("\nDUPLICATE ROWS:", df.duplicated().sum())

# Remove duplicate rows, if any
df = df.drop_duplicates()

# Basic statistics
print("\nNUMERICAL SUMMARY:")
print(df.describe().round(2))

# Distribution of monthly rent
plt.figure(figsize=(7, 4))
sns.histplot(df["monthly_rent"], bins=20, kde=True)
plt.title("Distribution of Monthly Rent")
plt.xlabel("Monthly Rent")
plt.ylabel("Number of Locations")
plt.tight_layout()
plt.savefig("outputs/rent_distribution.png")
plt.show()

# Relationship between footfall and target demand
plt.figure(figsize=(7, 4))
sns.scatterplot(
    data=df,
    x="daily_footfall",
    y="target_demand"
)
plt.title("Footfall vs Estimated Demand")
plt.tight_layout()
plt.savefig("outputs/footfall_vs_demand.png")
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(9, 6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

# Save cleaned dataset
df.to_csv("data/business_locations.csv", index=False)

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Charts saved in the outputs folder.")
