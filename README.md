
# AI-Based Business Location Selector

## 1. Project Overview

The AI-Based Business Location Selector is a machine learning-based educational project that helps users compare potential business locations based on rent, population, customer footfall, competition, accessibility, and other factors.

The project provides a Streamlit dashboard where users can select a business type, set their monthly rent budget, and view the top five recommended locations.

**Note:** This project uses synthetic/sample data. Its results are for educational purposes and do not guarantee actual business profitability.

## 2. Objectives

- Compare candidate business locations.
- Analyze rent, population, footfall, and competition.
- Train and evaluate machine learning models.
- Group locations using clustering.
- Rank locations using suitability scores.
- Display recommendations through an interactive dashboard.

## 3. Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly / Matplotlib, where used
- Visual Studio Code

## 4. Dataset Description

Dataset file: `data/business_locations.csv`

The dataset contains 1,000 records and 11 columns.

| Column | Description |
|---|---|
| location_name | Name of the candidate location |
| business_type | Type of business |
| monthly_rent | Monthly rental cost |
| population | Population of the area |
| competitor_count | Number of competitors |
| daily_footfall | Estimated daily customer footfall |
| accessibility_score | Accessibility rating |
| parking_score | Parking availability rating |
| public_transport_score | Public transport rating |
| income_level | Income level of the area |
| target_demand | Sample target demand value |

The dataset contains synthetic/sample values rather than verified real-world market data.

## 5. Machine Learning Algorithms

### Linear Regression
Used to predict the target demand value from input features.

### Decision Tree Regressor
Uses feature-based decision rules to estimate the target value.

### Random Forest Regressor
Combines multiple decision trees to produce a prediction.

### K-Means Clustering
Groups locations based on similarities in their features.

### Suitability Ranking
Ranks candidate locations using the scoring criteria implemented in the project.

## 6. Main Features

- Business type selection
- Monthly rent budget selection
- Matching candidate count
- Highest suitability score
- Best candidate rent
- Top five recommended locations
- Data visualizations
- Model evaluation results

## 7. Project Structure

```text
AI-Based Business Location Selector/
|
|-- app.py
|-- train_model.py
|-- clustering.py
|-- rank_locations.py
|-- evaluate_models.py
|
|-- data/
|   |-- business_locations.csv
|
|-- models/
|   |-- linear_regression.pkl
|   |-- decision_tree.pkl
|   |-- random_forest.pkl
|
|-- outputs/
|   |-- model_comparison.csv
|   |-- evaluation_results.csv
|   |-- clustered_locations.csv
|   |-- silhouette_scores.csv
|   |-- location_clusters.png
|   |-- location_ranking.csv
|   |-- top_5_locations.csv
|
|-- README.md
```

The exact generated files may differ depending on which scripts have been executed.

## 8. How to Run the Project

### Step 1: Open the Project

Open the project folder in Visual Studio Code.

### Step 2: Install Required Libraries

If a `requirements.txt` file exists, run:

```powershell
python -m pip install -r requirements.txt
```

Otherwise, install the libraries required by your project's Python imports.

### Step 3: Train the Models

```powershell
python train_model.py
```

### Step 4: Run Clustering

```powershell
python clustering.py
```

### Step 5: Rank Locations

```powershell
python rank_locations.py
```

### Step 6: Evaluate Models

```powershell
python evaluate_models.py
```

### Step 7: Start the Dashboard

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

If the generated models and output files already exist, rerun the preparation scripts only when necessary.

## 9. Model Evaluation

The previously recorded evaluation results were:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 78.768 | 98.637 | 0.9749 |
| Decision Tree | 199.704 | 247.790 | 0.8417 |
| Random Forest | 123.157 | 155.044 | 0.9380 |

The previous K-Means clustering run produced three clusters with a silhouette score of 0.1011.

These results are based on synthetic/sample data. They do not demonstrate equivalent accuracy on real business locations. Verify the metrics against the latest evaluation output before final submission.

## 10. Limitations

- The dataset is synthetic/sample data.
- The suitability score depends on the implemented scoring rules.
- The model results may not generalize to real locations.
- Actual market conditions and business profitability are not verified.

## 11. Future Scope

- Use verified real-world location data.
- Add geographic maps and verified location information.
- Improve and validate suitability scoring.
- Evaluate models on observed business outcomes.
- Add more business categories and regions.

## 12. Conclusion

The AI-Based Business Location Selector demonstrates how machine learning, clustering, suitability scoring, and data visualization can be combined to compare potential business locations. The Streamlit dashboard presents matching candidates and a ranked list of recommended locations. The current version is an educational prototype and requires real-world data and validation before practical commercial use.

