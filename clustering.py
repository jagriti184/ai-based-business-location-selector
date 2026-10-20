
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("data/business_locations.csv")

# Select features for clustering
features = [
    "monthly_rent",
    "population",
    "competitor_count",
    "daily_footfall",
    "accessibility_score",
    "parking_score",
    "public_transport_score"
]

X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compare different cluster counts
scores = []

for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append({"Clusters": k, "Silhouette Score": round(score, 4)})

results = pd.DataFrame(scores)
print("CLUSTER EVALUATION")
print(results.to_string(index=False))

# Select 3 clusters for the final demonstration
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Save clustering model and scaler
import joblib

joblib.dump(scaler, "models/clustering_scaler.pkl")
joblib.dump(kmeans, "models/kmeans_model.pkl")

# Save locations with their cluster labels
df.to_csv("outputs/clustered_locations.csv", index=False)
results.to_csv("outputs/silhouette_scores.csv", index=False)

# Plot clusters
plt.figure(figsize=(8, 5))
plt.scatter(
    df["monthly_rent"],
    df["daily_footfall"],
    c=df["cluster"],
    alpha=0.7
)
plt.xlabel("Monthly Rent")
plt.ylabel("Daily Footfall")
plt.title("Business Location Clusters")
plt.tight_layout()
plt.savefig("outputs/location_clusters.png")
plt.show()

print("\nCLUSTER PROFILES")
print(df.groupby("cluster")[features].mean().round(2))

print("\nClustering completed successfully!")
