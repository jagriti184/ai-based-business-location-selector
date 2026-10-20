
import os
import numpy as np
import pandas as pd

# Reproducible sample dataset
np.random.seed(42)

os.makedirs("data", exist_ok=True)

areas = [
    "MP Nagar", "New Market", "Arera Colony",
    "Kolar Road", "Hoshangabad Road", "Bawadiya Kalan",
    "Bairagarh", "Ayodhya Bypass", "Shahpura",
    "Gulmohar Colony"
]

n = 1000

df = pd.DataFrame({
    "location_name": np.random.choice(areas, n),
    "business_type": np.random.choice(
        ["Cafe", "Pharmacy", "Grocery", "Clothing"], n
    ),
    "monthly_rent": np.random.randint(10000, 80001, n),
    "population": np.random.randint(5000, 50001, n),
    "competitor_count": np.random.randint(0, 21, n),
    "daily_footfall": np.random.randint(100, 5001, n),
    "accessibility_score": np.random.randint(1, 11, n),
    "parking_score": np.random.randint(1, 11, n),
    "public_transport_score": np.random.randint(1, 11, n),
    "income_level": np.random.choice(
        ["Low", "Medium", "High"], n
    )
})

# Illustrative synthetic demand target
df["target_demand"] = (
    0.35 * df["daily_footfall"]
    + 0.25 * df["population"] / 10
    + 50 * df["accessibility_score"]
    + 30 * df["public_transport_score"]
    - 10 * df["competitor_count"]
    + np.random.normal(0, 100, n)
).clip(lower=0).round(2)

df.to_csv("data/business_locations.csv", index=False)

print("Dataset created successfully!")
print("Total records:", len(df))
print("\nFirst five records:")
print(df.head())
print("\nDataset saved to data/business_locations.csv")
