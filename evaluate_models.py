
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "business_locations.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------
# 1. Check dataset
# --------------------------------

print("\n1. DATASET TEST")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

# --------------------------------
# 2. Prepare test data
# --------------------------------

X = df.drop(columns=["target_demand"])
y = df["target_demand"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# 3. Evaluate saved ML models
# --------------------------------

print("\n2. MODEL EVALUATION")

model_files = {
    "Linear Regression": "linear_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl"
}

results = []

for model_name, filename in model_files.items():
    model_path = MODELS_DIR / filename

    if not model_path.exists():
        print(f"{model_name}: Model file not found.")
        continue

    try:
        model = joblib.load(model_path)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": model_name,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2 Score": round(r2, 4)
        })

        print(f"\n{model_name}")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2 Score: {r2:.4f}")

    except Exception as error:
        print(f"{model_name} evaluation failed: {error}")

if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "evaluation_results.csv", index=False
    )

    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False))
    print("\nSaved: outputs/evaluation_results.csv")
else:
    print("No trained models were available for evaluation.")

# --------------------------------
# 4. Check clustering quality
# --------------------------------

print("\n3. CLUSTERING TEST")

cluster_path = OUTPUT_DIR / "clustered_locations.csv"

cluster_features = [
    "monthly_rent",
    "population",
    "competitor_count",
    "daily_footfall",
    "accessibility_score",
    "parking_score",
    "public_transport_score"
]

if cluster_path.exists():
    clustered = pd.read_csv(cluster_path)

    if "cluster" in clustered.columns:
        scaler = StandardScaler()
        cluster_data = scaler.fit_transform(
            clustered[cluster_features]
        )

        labels = clustered["cluster"]
        number_of_clusters = labels.nunique()

        if 2 <= number_of_clusters < len(clustered):
            score = silhouette_score(cluster_data, labels)
            print("Number of clusters:", number_of_clusters)
            print(f"Silhouette Score: {score:.4f}")
        else:
            print("Cannot calculate silhouette score for these labels.")
    else:
        print("Cluster labels are missing.")
else:
    print("Cluster output not found. Run python clustering.py")

# --------------------------------
# 5. Check location ranking
# --------------------------------

print("\n4. RANKING TEST")

ranking_path = OUTPUT_DIR / "location_ranking.csv"
top5_path = OUTPUT_DIR / "top_5_locations.csv"

if ranking_path.exists():
    ranking = pd.read_csv(ranking_path)

    print("Ranking file exists.")
    print("Ranked records:", len(ranking))

    if "suitability_score" in ranking.columns:
        print(
            "Scores are descending:",
            ranking["suitability_score"].is_monotonic_decreasing
        )
        print("Top 5 preview:")

        if top5_path.exists():
            top5 = pd.read_csv(top5_path)
            print(top5.head(5).to_string(index=False))
    else:
        print("Suitability score column is missing.")
else:
    print("Ranking file not found. Run python rank_locations.py")

print("\nTesting completed.")
