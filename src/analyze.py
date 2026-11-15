# Market Research & Consumer Behaviour Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Market Research Project Started!")

print("All libraries imported successfully.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 50)
print("MARKET RESEARCH & CONSUMER BEHAVIOUR ANALYSIS")
print("=" * 50)

# Load dataset
df = pd.read_csv("data/online_shoppers.csv")

# Basic information
print("\n1. DATASET OVERVIEW")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Data types
print("\n2. DATA TYPES")
print(df.dtypes)

# Missing values
print("\n3. MISSING VALUES")
print(df.isnull().sum())

# Duplicate records
print("\n4. DUPLICATE RECORDS")
print("Duplicates:", df.duplicated().sum())

# Target distribution
print("\n5. PURCHASE DISTRIBUTION")
print(df["Revenue"].value_counts())
print("\nPurchase percentage:")
print(df["Revenue"].value_counts(normalize=True) * 100)

# Descriptive statistics
print("\n6. DESCRIPTIVE STATISTICS")
print(df.describe(include="all").T)

print("\nINITIAL DATA ANALYSIS COMPLETED!")

# ==========================================
# 7. DATA CLEANING
# ==========================================

print("\n7. DATA CLEANING")

df = df.drop_duplicates().copy()

print("Shape after removing duplicates:", df.shape)

df.to_csv("data/cleaned_online_shoppers.csv", index=False)

# Convert target into numerical form for analysis
df["Purchase"] = df["Revenue"].astype(int)

print("Cleaned dataset saved!")

# ==========================================
# 8. EXPLORATORY DATA ANALYSIS
# ==========================================

Path("figures").mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# Graph 1: Purchase distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Revenue")
plt.title("Purchase vs Non-Purchase Sessions")
plt.savefig("figures/purchase_distribution.png",
            bbox_inches="tight")
plt.close()

# Graph 2: Purchase by visitor type
visitor_data = df.groupby("VisitorType")["Purchase"].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(data=visitor_data,
            x="VisitorType", y="Purchase")
plt.title("Purchase Conversion by Visitor Type")
plt.ylabel("Purchase Conversion Rate")
plt.savefig("figures/visitor_conversion.png",
            bbox_inches="tight")
plt.close()

# Graph 3: Purchase conversion by month
month_data = df.groupby("Month")["Purchase"].mean().reset_index()

plt.figure(figsize=(9, 5))
sns.barplot(data=month_data,
            x="Month", y="Purchase")
plt.title("Purchase Conversion by Month")
plt.ylabel("Purchase Conversion Rate")
plt.savefig("figures/monthly_conversion.png",
            bbox_inches="tight")
plt.close()

# Graph 4: Page values and purchase behaviour
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Revenue", y="PageValues")
plt.title("Page Values vs Purchase Behaviour")
plt.savefig("figures/page_values.png",
            bbox_inches="tight")
plt.close()

print("\nEDA COMPLETED!")
print("4 graphs saved in figures folder.")

print("\nPURCHASE RATE BY VISITOR TYPE")
print(visitor_data)

print("\nPURCHASE RATE BY MONTH")
print(month_data)

print("\nFINAL DATASET SHAPE:", df.shape)


# ==========================================
# 9. STATISTICAL HYPOTHESIS TESTING
# ==========================================

from scipy.stats import chi2_contingency, mannwhitneyu

print("\n9. STATISTICAL ANALYSIS")

# Test 1: Visitor Type vs Purchase
visitor_table = pd.crosstab(df["VisitorType"], df["Revenue"])
chi2, p_value, dof, expected = chi2_contingency(visitor_table)

print("\nVisitor Type vs Purchase")
print("Chi-square:", chi2)
print("P-value:", p_value)

if p_value < 0.05:
    print("Result: Significant association found.")
else:
    print("Result: No significant association found.")

# Test 2: Weekend vs Purchase
weekend_table = pd.crosstab(df["Weekend"], df["Revenue"])
chi2_w, p_w, dof_w, expected_w = chi2_contingency(weekend_table)

print("\nWeekend vs Purchase")
print("Chi-square:", chi2_w)
print("P-value:", p_w)

if p_w < 0.05:
    print("Result: Significant association found.")
else:
    print("Result: No significant association found.")

# Test 3: PageValues comparison
buyers = df.loc[df["Revenue"] == True, "PageValues"]
non_buyers = df.loc[df["Revenue"] == False, "PageValues"]

stat, p_page = mannwhitneyu(
    buyers, non_buyers, alternative="two-sided"
)

print("\nPageValues vs Purchase")
print("Mann-Whitney statistic:", stat)
print("P-value:", p_page)

# ==========================================
# 10. MACHINE LEARNING
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay
)
import joblib

print("\n10. MACHINE LEARNING")

# Separate features and target
X = df.drop(columns=["Revenue", "Purchase"])
y = df["Revenue"].astype(int)

# Categorical features
categorical_cols = [
    "Month", "VisitorType", "OperatingSystems",
    "Browser", "Region", "TrafficType", "Weekend"
]

numerical_cols = [
    col for col in X.columns
    if col not in categorical_cols
]

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)
    ap = average_precision_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc,
        "Average Precision": ap
    })

    print(f"\n{name}")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))
    print("Average Precision:", round(ap, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions
    )
    plt.title(f"{name} - Confusion Matrix")
    filename = name.lower().replace(" ", "_")
    plt.savefig(
        f"figures/{filename}_confusion_matrix.png",
        bbox_inches="tight"
    )
    plt.close()

    joblib.dump(
        pipeline,
        f"reports/{filename}_model.joblib"
    )

# Save comparison
results_df = pd.DataFrame(results)
results_df.to_csv(
    "reports/model_comparison.csv", index=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nSTATISTICAL ANALYSIS AND ML COMPLETED!")


# ==========================================
# 11. FEATURE IMPORTANCE
# ==========================================

print("\n11. FEATURE IMPORTANCE")

rf_pipeline = pipeline

feature_names = (
    rf_pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

importance = rf_pipeline.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values("Importance", ascending=False)

print("\nTOP 15 IMPORTANT FEATURES")
print(importance_df.head(15).to_string(index=False))

importance_df.to_csv(
    "reports/feature_importance.csv", index=False
)

plt.figure(figsize=(10, 7))
sns.barplot(
    data=importance_df.head(15),
    x="Importance",
    y="Feature"
)
plt.title("Top 15 Purchase Prediction Features")
plt.savefig(
    "figures/feature_importance.png",
    bbox_inches="tight"
)
plt.close()

# ==========================================
# 12. PAGEVALUES LEAKAGE CHECK
# ==========================================

print("\n12. PAGEVALUES LEAKAGE CHECK")

X_reduced = X.drop(columns=["PageValues"])

reduced_num = [
    col for col in X_reduced.columns
    if col not in categorical_cols
]

reduced_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), reduced_num),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

reduced_model = Pipeline([
    ("preprocessor", reduced_preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

reduced_model.fit(X_train.drop(columns=["PageValues"]), y_train)

reduced_predictions = reduced_model.predict(
    X_test.drop(columns=["PageValues"])
)

reduced_probabilities = reduced_model.predict_proba(
    X_test.drop(columns=["PageValues"])
)[:, 1]

print("ROC AUC without PageValues:",
      roc_auc_score(y_test, reduced_probabilities))

print("F1 Score without PageValues:",
      f1_score(y_test, reduced_predictions))

joblib.dump(
    reduced_model,
    "reports/random_forest_without_pagevalues.joblib"
)

print("\nFEATURE ANALYSIS COMPLETED!")

# Online Shoppers Purchasing Intention Analysis

A data science project focused on understanding online consumer behaviour, analysing purchase patterns, and predicting purchase intention using machine learning.

## Project Workflow
1. Data collection
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Statistical hypothesis testing
5. Machine learning model development
6. Model evaluation
7. Feature importance and sensitivity analysis

## Models
- Logistic Regression
- Random Forest Classifier

## Project Structure
- `data/` — Original and cleaned datasets
- `src/` — Python analysis script
- `figures/` — Visualisations and confusion matrices
- `reports/` — Model results, trained models and final report

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy ucimlrepo joblib
python src/analyze.py
```

## Key Result
Random Forest achieved an ROC-AUC score of 92.88% on the test set. A sensitivity check without PageValues produced an ROC-AUC of 78.23%.

## Technologies
Python | Pandas | NumPy | Matplotlib | Seaborn | SciPy | Scikit-learn


