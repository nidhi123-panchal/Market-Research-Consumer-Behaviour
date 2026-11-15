# Online Shoppers Purchasing Intention: Market Research & Consumer Behaviour Analysis

## 1. Executive Summary

This project analyses online shopping sessions to understand consumer purchasing behaviour and identify factors associated with purchase decisions. It combines exploratory data analysis, statistical hypothesis testing and machine learning.

The analysis uses the UCI Online Shoppers Purchasing Intention dataset containing 12,330 original records. After removing 125 duplicate records, 12,205 records remained.

Two machine learning models were evaluated: Logistic Regression and Random Forest. Random Forest achieved stronger performance across the reported evaluation metrics. However, its performance was substantially lower when PageValues was excluded, highlighting an important limitation.

## 2. Project Objectives

- Analyse online shopping and customer purchasing patterns.
- Identify differences in purchase conversion across visitor types and months.
- Test statistical associations between selected variables and purchases.
- Develop and evaluate machine learning classification models.
- Identify influential features and examine model limitations.
- Translate findings into practical business recommendations.

## 3. Dataset Description

**Source:** UCI Machine Learning Repository  
**Dataset:** Online Shoppers Purchasing Intention

| Dataset detail | Value |
|---|---:|
| Original records | 12,330 |
| Features | 17 |
| Target variable | Revenue |
| Duplicate records removed | 125 |
| Final records | 12,205 |

The dataset contains session-level information, including administrative browsing, informational browsing, product-related activity, bounce rates, exit rates, page values, visitor type, month and weekend activity.

## 4. Data Cleaning and Preparation

The following steps were performed:

- Inspected dataset structure and data types.
- Checked missing values.
- Identified 125 duplicate records.
- Removed duplicate records.
- Created a numerical `Purchase` variable from the `Revenue` target.
- Saved the cleaned dataset for further analysis.

No missing values were identified in the original dataset.

## 5. Exploratory Data Analysis

### 5.1 Purchase Distribution

- Non-purchasing sessions: approximately 84.53%.
- Purchasing sessions: approximately 15.47%.

This shows that purchasing sessions represent a minority of the observed online shopping sessions.

### 5.2 Purchase Conversion by Visitor Type

| Visitor type | Purchase conversion |
|---|---:|
| New Visitor | 24.93% |
| Other | 19.75% |
| Returning Visitor | 14.09% |

New visitors recorded the highest purchase conversion rate among the three visitor categories.

### 5.3 Purchase Conversion by Month

| Month | Purchase conversion |
|---|---:|
| November | 25.49% |
| October | 20.95% |
| September | 19.20% |
| August | 17.55% |
| July | 15.28% |
| December | 12.66% |
| May | 10.96% |
| March | 10.32% |
| June | 10.18% |
| February | 1.66% |

November recorded the highest purchase conversion rate in this dataset.

## 6. Statistical Hypothesis Testing

Three statistical comparisons were performed.

### 6.1 Visitor Type vs Purchase

- Test: Chi-square test of independence
- Chi-square statistic: 130.67
- P-value: 4.23 × 10⁻²⁹

**Result:** A statistically significant association was found between visitor type and purchase outcome.

### 6.2 Weekend vs Purchase

- Test: Chi-square test of independence
- Chi-square statistic: 9.20
- P-value: 0.0024

**Result:** A statistically significant association was found between weekend activity and purchase outcome.

### 6.3 PageValues vs Purchase

- Test: Mann-Whitney U test
- Statistic: 16,951,382
- P-value: Reported as 0.0 by the software due to numerical precision.

**Result:** PageValues differed significantly between purchasing and non-purchasing sessions.

These statistical results indicate associations or distributional differences. They do not establish causation.

## 7. Machine Learning Models

Two classification algorithms were trained and evaluated using an 80:20 stratified train-test split.

### 7.1 Model Performance

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 84.60% | 87.51% |
| Precision | 50.50% | 57.20% |
| Recall | 79.58% | 80.10% |
| F1 Score | 61.79% | 66.74% |
| ROC-AUC | 90.97% | 92.88% |
| Average Precision | 66.27% | 72.25% |

Random Forest achieved higher scores across all reported metrics on this test split.

### 7.2 Model Interpretation

Random Forest achieved an F1 score of 66.74% and recall of 80.10% for the purchase class.

This indicates that the model identified a substantial proportion of purchasing sessions, although its purchase-class precision was 57.20%.

## 8. Feature Importance Analysis

The Random Forest model identified the following influential features:

| Feature | Importance |
|---|---:|
| PageValues | 40.57% |
| ExitRates | 9.88% |
| ProductRelated_Duration | 7.65% |
| ProductRelated | 5.92% |
| BounceRates | 4.75% |
| Administrative_Duration | 4.27% |
| Administrative | 3.71% |

PageValues was the most influential feature in the fitted model.

Feature importance represents the model's use of a feature; it does not establish that the feature causes purchases.

## 9. PageValues Sensitivity Analysis

To examine the model's dependence on PageValues, a second Random Forest model was trained without this feature.

| Metric | With PageValues | Without PageValues |
|---|---:|---:|
| ROC-AUC | 92.88% | 78.23% |
| F1 Score | 66.74% | 42.78% |

The reduction in performance indicates substantial dependence on PageValues.

Its availability and calculation timing should be investigated before using this model for predictions during an ongoing shopping session. This sensitivity check raises a potential prediction-time leakage concern but does not independently prove leakage.

## 10. Business Insights and Recommendations

### Insight 1: Visitor type is associated with purchasing behaviour

New visitors recorded a higher purchase conversion rate than returning visitors.

**Recommendation:** Investigate which acquisition channels and landing-page experiences attract high-converting new visitors. Test onboarding and product discovery improvements for returning visitors.

### Insight 2: Monthly conversion varies

November recorded the highest conversion rate in the dataset.

**Recommendation:** Examine seasonal campaigns, product demand and promotional activity around high-conversion months before planning marketing budgets.

### Insight 3: Product-related browsing behaviour matters

Product-related page activity and duration were among the influential features in the model.

**Recommendation:** Evaluate product-page navigation, product information quality and checkout journey friction to identify opportunities to improve the shopping experience.

### Insight 4: Model performance depends heavily on PageValues

Removing PageValues reduced ROC-AUC to 78.23%.

**Recommendation:** Validate feature availability at the intended prediction time and evaluate a model using only information genuinely available before the purchase decision.

These recommendations are proposed actions for further business investigation, not proven causal interventions.

## 11. Project Limitations

- The analysis uses one historical dataset.
- Results may not generalise to other e-commerce platforms or future customer traffic.
- Statistical associations do not establish causal relationships.
- The dataset contains an imbalanced purchase target.
- PageValues creates a potential prediction-time leakage concern.
- The reported model scores come from one train-test split and require further validation.

## 12. Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Joblib

## 13. Project Deliverables

- Cleaned dataset
- Exploratory data analysis visualisations
- Statistical hypothesis testing results
- Logistic Regression model
- Random Forest model
- Model comparison results
- Feature importance analysis
- PageValues sensitivity analysis

## 14. Conclusion

This project demonstrates an end-to-end analysis of online shopping behaviour using data cleaning, exploratory analysis, statistical testing and machine learning.

The findings highlight differences in purchase conversion across visitor types and months. Random Forest outperformed Logistic Regression on the evaluated test split, while the PageValues sensitivity analysis revealed an important limitation that needs further investigation.

The project provides a foundation for data-driven customer behaviour analysis and e-commerce decision-making.
