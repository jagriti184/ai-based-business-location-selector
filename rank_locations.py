
import os
import pandas as pd

os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("data/business_locations.csv")

# Example monthly rent budget
monthly_budget = 50000

# Normalize a value between 0 and 100
def normalize(series):
    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(50.0, index=series.index)

    return (series - minimum) / (maximum - minimum) * 100


# 1. Customer demand score
df["demand_score"] = normalize(df["target_demand"])

# 2. Accessibility score
df["accessibility_component"] = (
    df["accessibility_score"] * 10
)

# 3. Footfall score
df["footfall_component"] = normalize(df["daily_footfall"])

# 4. Parking and public transport score
df["transport_component"] = (
    df["parking_score"] * 5
    + df["public_transport_score"] * 5
)

# 5. Competition opportunity score
# Fewer competitors receive a higher score in this example
df["competition_component"] = (
    100 - normalize(df["competitor_count"])
)

# 6. Rent affordability score
df["rent_affordability"] = (
    (1 - df["monthly_rent"] / monthly_budget)
    .clip(0, 1) * 100
)

# Weighted suitability score
df["suitability_score"] = (
    0.30 * df["demand_score"]
    + 0.20 * df["accessibility_component"]
    + 0.15 * df["footfall_component"]
    + 0.15 * df["transport_component"]
    + 0.10 * df["competition_component"]
    + 0.10 * df["rent_affordability"]
).round(2)

# Rank all candidate records
df = df.sort_values(
    "suitability_score", ascending=False
).reset_index(drop=True)

df.insert(0, "rank", range(1, len(df) + 1))

# Save complete ranking
df.to_csv("outputs/location_ranking.csv", index=False)

# Save top five candidates
top_five = df.head(5)

top_five.to_csv(
    "outputs/top_5_locations.csv", index=False
)

# Display important information
columns_to_show = [
    "rank",
    "location_name",
    "business_type",
    "monthly_rent",
    "daily_footfall",
    "competitor_count",
    "suitability_score"
]

print("\nTOP 5 BUSINESS LOCATION CANDIDATES")
print(top_five[columns_to_show].to_string(index=False))

print("\nRanking completed successfully!")
print("Full ranking: outputs/location_ranking.csv")
print("Top 5: outputs/top_5_locations.csv")
