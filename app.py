
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI-Based Business Location Selector",
    page_icon="📍",
    layout="wide"
)

# Use the folder containing this app.py file
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "business_locations.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"

# --------------------------------------------------
# 2. HEADER
# --------------------------------------------------

st.title("📍 AI-Based Business Location Selector")

st.write(
    "Compare business locations using machine learning, "
    "rent affordability, customer footfall and suitability scores."
)

st.info(
    "Educational prototype: the dataset and demand targets are synthetic. "
    "Results do not establish actual business profitability."
)

# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.write("Please check that data/business_locations.csv exists.")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except Exception as error:
    st.error(f"Could not read the dataset: {error}")
    st.stop()

required_columns = [
    "location_name",
    "business_type",
    "monthly_rent",
    "population",
    "competitor_count",
    "daily_footfall",
    "accessibility_score",
    "parking_score",
    "public_transport_score",
    "income_level",
    "target_demand"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()

if df.empty:
    st.error("The dataset contains no records.")
    st.stop()

# --------------------------------------------------
# 4. CALCULATE SUITABILITY COMPONENTS
# --------------------------------------------------

def normalize(series):
    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(50.0, index=series.index)

    if maximum == minimum:
        return pd.Series(50.0, index=series.index)

    return (series - minimum) / (maximum - minimum) * 100


df["demand_score"] = normalize(df["target_demand"])

df["accessibility_component"] = (
    df["accessibility_score"].clip(0, 10) * 10
)

df["footfall_component"] = normalize(df["daily_footfall"])

df["transport_component"] = (
    df["parking_score"].clip(0, 10) * 5
    + df["public_transport_score"].clip(0, 10) * 5
)

df["competition_component"] = (
    100 - normalize(df["competitor_count"])
)

# --------------------------------------------------
# 5. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Business Preferences")

business_types = sorted(
    df["business_type"].dropna().unique().tolist()
)

if not business_types:
    st.error("No business types are available in the dataset.")
    st.stop()

selected_business = st.sidebar.selectbox(
    "Select Business Type",
    business_types
)

maximum_rent = int(df["monthly_rent"].max())
minimum_rent = int(df["monthly_rent"].min())

default_budget = min(50000, maximum_rent)
default_budget = max(default_budget, minimum_rent)

budget = st.sidebar.slider(
    "Maximum Monthly Rent Budget (₹)",
    min_value=minimum_rent,
    max_value=maximum_rent,
    value=default_budget,
    step=max(1, (maximum_rent - minimum_rent) // 100)
)

st.sidebar.caption(
    "Only candidate records within the selected rent budget are displayed."
)

# --------------------------------------------------
# 6. FILTER CANDIDATE LOCATIONS
# --------------------------------------------------

filtered = df[
    (df["business_type"] == selected_business)
    & (df["monthly_rent"] <= budget)
].copy()

if filtered.empty:
    st.warning(
        "No locations match this business type and budget. "
        "Try increasing the rent budget."
    )
    st.stop()

# Calculate affordability using the selected budget
filtered["rent_affordability"] = (
    (1 - filtered["monthly_rent"] / budget)
    .clip(0, 1) * 100
)

# Weighted suitability score
filtered["suitability_score"] = (
    0.30 * filtered["demand_score"]
    + 0.20 * filtered["accessibility_component"]
    + 0.15 * filtered["footfall_component"]
    + 0.15 * filtered["transport_component"]
    + 0.10 * filtered["competition_component"]
    + 0.10 * filtered["rent_affordability"]
).round(2)

filtered = filtered.sort_values(
    "suitability_score",
    ascending=False
).reset_index(drop=True)

filtered.insert(0, "rank", range(1, len(filtered) + 1))

# --------------------------------------------------
# 7. SUMMARY METRICS
# --------------------------------------------------

st.subheader("📊 Business Location Overview")

best = filtered.iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Matching Candidates",
    f"{len(filtered):,}"
)

col2.metric(
    "Highest Suitability Score",
    f"{best['suitability_score']:.2f}/100"
)

col3.metric(
    "Best Candidate Rent",
    f"₹{best['monthly_rent']:,.0f}"
)

st.caption(
    "Suitability scores are calculated from the selected budget "
    "and the illustrative scoring weights."
)

# --------------------------------------------------
# 8. TOP FIVE LOCATIONS
# --------------------------------------------------

st.subheader("🏆 Top 5 Recommended Locations")

display_columns = [
    "rank",
    "location_name",
    "business_type",
    "monthly_rent",
    "population",
    "daily_footfall",
    "competitor_count",
    "suitability_score"
]

top_five = filtered.head(5)

st.dataframe(
    top_five[display_columns].rename(columns={
        "rank": "Rank",
        "location_name": "Location",
        "business_type": "Business Type",
        "monthly_rent": "Monthly Rent (₹)",
        "population": "Population",
        "daily_footfall": "Daily Footfall",
        "competitor_count": "Competitors",
        "suitability_score": "Suitability Score"
    }),
    hide_index=True,
    use_container_width=True
)

# --------------------------------------------------
# 9. SUITABILITY SCORE CHART
# --------------------------------------------------

st.subheader("📈 Suitability Score Comparison")

chart_data = top_five.copy()

fig = px.bar(
    chart_data,
    x="location_name",
    y="suitability_score",
    hover_data=[
        "monthly_rent",
        "daily_footfall",
        "competitor_count"
    ],
    labels={
        "location_name": "Candidate Location",
        "suitability_score": "Suitability Score"
    },
    title="Top 5 Candidate Locations"
)

fig.update_layout(
    yaxis_range=[0, 100],
    xaxis_title="Candidate Location",
    yaxis_title="Score out of 100"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 10. RENT VS FOOTFALL CHART
# --------------------------------------------------

st.subheader("💰 Rent vs Daily Footfall")

scatter_fig = px.scatter(
    filtered,
    x="monthly_rent",
    y="daily_footfall",
    color="suitability_score",
    hover_data=[
        "location_name",
        "business_type",
        "competitor_count"
    ],
    labels={
        "monthly_rent": "Monthly Rent (₹)",
        "daily_footfall": "Daily Footfall",
        "suitability_score": "Suitability Score"
    },
    title="Compare Rent and Footfall"
)

st.plotly_chart(scatter_fig, use_container_width=True)

# --------------------------------------------------
# 11. EXPLORE A LOCATION
# --------------------------------------------------

st.subheader("🔎 Explore a Candidate Location")

selected_index = st.selectbox(
    "Choose a candidate location",
    filtered.index.tolist(),
    format_func=lambda index: (
        f"{filtered.loc[index, 'location_name']} | "
        f"Rent ₹{filtered.loc[index, 'monthly_rent']:,.0f} | "
        f"Score {filtered.loc[index, 'suitability_score']:.1f}"
    )
)

candidate = filtered.loc[selected_index]

left, right = st.columns(2)

with left:
    st.markdown("**Location Details**")
    st.write("Location:", candidate["location_name"])
    st.write("Business Type:", candidate["business_type"])
    st.write("Monthly Rent:", f"₹{candidate['monthly_rent']:,.0f}")
    st.write("Population:", f"{candidate['population']:,.0f}")
    st.write("Daily Footfall:", f"{candidate['daily_footfall']:,.0f}")
    st.write("Competitors:", int(candidate["competitor_count"]))

with right:
    st.markdown("**Accessibility and Suitability**")
    st.write(
        "Accessibility:",
        f"{candidate['accessibility_score']}/10"
    )
    st.write("Parking:", f"{candidate['parking_score']}/10")
    st.write(
        "Public Transport:",
        f"{candidate['public_transport_score']}/10"
    )
    st.write(
        "Suitability Score:",
        f"{candidate['suitability_score']:.2f}/100"
    )

    if candidate["monthly_rent"] > budget * 0.7:
        st.warning("Rent uses more than 70% of the selected budget.")
    else:
        st.success("Rent uses 70% or less of the selected budget.")

# --------------------------------------------------
# 12. RANDOM FOREST DEMAND ESTIMATE
# --------------------------------------------------

st.subheader("🤖 Machine Learning Demand Estimate")

st.write(
    "Estimate the synthetic demand target for the selected candidate "
    "using the trained Random Forest model."
)

if MODEL_PATH.exists():

    if st.button("Predict Estimated Demand"):
        try:
            model = joblib.load(MODEL_PATH)

            model_input = pd.DataFrame([{
                column: candidate[column]
                for column in required_columns
                if column != "target_demand"
            }])

            prediction = model.predict(model_input)[0]

            st.metric(
                "Estimated Demand Score",
                f"{prediction:,.2f}"
            )

            st.caption(
                "This prediction estimates a synthetic target, "
                "not actual customers, revenue or profit."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
else:
    st.warning(
        "Random Forest model not found. Run "
        "'python train_model.py' from the project folder."
    )

# --------------------------------------------------
# 13. DOWNLOAD RESULTS
# --------------------------------------------------

st.subheader("📥 Download Results")

download_data = filtered[display_columns].rename(columns={
    "rank": "Rank",
    "location_name": "Location",
    "business_type": "Business Type",
    "monthly_rent": "Monthly Rent (₹)",
    "population": "Population",
    "daily_footfall": "Daily Footfall",
    "competitor_count": "Competitors",
    "suitability_score": "Suitability Score"
})

csv_data = download_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Ranked Locations (CSV)",
    data=csv_data,
    file_name="ranked_business_locations.csv",
    mime="text/csv"
)

# --------------------------------------------------
# 14. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Based Business Location Selector | "
    "Machine Learning, Clustering and Decision Support"
)

st.caption(
    "Disclaimer: This educational prototype uses synthetic data. "
    "Suitability rankings are illustrative and do not guarantee "
    "business success or profitability."
)
