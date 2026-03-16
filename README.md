# Chef's Compass — Revenue Prediction Model

A supervised regression analysis predicting individual customer revenue for a meal-prep delivery platform. Built as a final project for Computational Analytics at Hult International Business School.

**Final Model:** Gradient Boosting Machine  
**Test R²:** 0.8008 | **Train R²:** 0.8502 | **Gap:** 0.0494

---

## Project Overview

Chef's Compass is a meal-prep delivery business with a diverse customer base spanning varying levels of digital engagement, subscription behavior, and order history. This project builds a machine learning pipeline to predict customer-level revenue using behavioral and transactional features, enabling data-driven decisions around customer segmentation, retention, and lifetime value estimation.

---

## Repository Structure

```
├── Linear_modeling_case_code.ipynb   # Main analysis notebook
├── README.md
├── Data Dictionary - Chef's Compass.Xlsx
├── Dataset - Chef's Compass.Xlsx
├── Course Case - Chef's Compass.pdf # Business description
```

---

## Methodology

### 1. Exploratory Data Analysis
- Inspected feature distributions, skewness, and missing values
- Identified right-skewed features requiring log transformation
- Analyzed Pearson correlation and Mutual Information (MI) scores against both raw and log-transformed revenue

### 2. Feature Engineering
Eight behavioral features were constructed from raw columns:

| Feature | Description |
|---|---|
| `purchase_per_session` | Orders ÷ total login sessions — conversion efficiency |
| `unique_meal_ratio` | Unique meals ÷ total meals ordered — exploration behavior |
| `customer_loyalty` | Orders ÷ categories viewed — purchase commitment |
| `photo_dependency` | Photos viewed ÷ categories viewed — visual engagement intensity |
| `problem_order_ratio` | (Cancellations + late deliveries) ÷ orders — service quality proxy |
| `mobile_preference` | Mobile logins ÷ total logins — platform preference |
| `service_contact_rate` | Customer service contacts ÷ orders — support dependency |
| `browse_to_purchase` | (Categories + photos viewed) ÷ orders — browsing efficiency |
| `weekly_plan_flag` | Binary: customer has ever used weekly plan (1/0) |
| `is_low_meals` | Binary: log total meals ordered < 3.2 threshold (1/0) |

Features with visual evidence of weak or noisy relationships (`mobile_preference`, `log_problem_order_ratio`, `log_browse_to_purchase`, `log_service_contact_rate`) were excluded from the final model after scatterplot validation.

### 3. Preprocessing
- Log transformation (`np.log1p`) applied to all features with skewness > 0.5
- StandardScaler applied to continuous features for linear model compatibility
- Binary and categorical features left unscaled

### 4. Model Development
Candidate models tested across multiple feature sets (`x_all`, `x_best`, `x_sp`, `df_scaled`) and target configurations (`y_origin`, `y_log`):

| Model | Notes |
|---|---|
| OLS | Baseline linear regression |
| Lasso | L1 regularization, alpha grid search |
| Ridge | L2 regularization, alpha grid search |
| Elastic Net (SGD) | Combined L1/L2, grid searched alpha + l1_ratio |
| KNN | Tested k = 5–50 with uniform weights |
| Random Forest | Tuned via `quick_tree`, max_depth=8 |
| **Gradient Boosting Machine** | **Final model — best generalization** |

GridSearchCV (5-fold CV) used for hyperparameter tuning on all regularized models.

---

## Final Model

```python
GradientBoostingRegressor(
    learning_rate=0.04,
    max_depth=5,
    min_samples_leaf=72,
    subsample=0.8,
    random_state=42
)
```

**Target:** `log_REVENUE` (log-transformed for distribution normalization)  
**Feature set:** All log-transformed features + engineered behavioral features + binary flags

### Performance

| Split | R² |
|---|---|
| Train | 0.8502 |
| Test | 0.8008 |
| Gap | 0.0494 |

### Top Predictors (by feature importance)
1. `log_TOTAL_MEALS_ORDERED` — strongest single predictor (corr = 0.69)
2. `log_purchase_per_session` — conversion efficiency
3. `log_customer_loyalty` — purchase commitment relative to browsing
4. `log_unique_meal_ratio` — menu exploration behavior
5. `log_photo_dependency` — visual engagement intensity

---

## Tech Stack

| Tool | Use |
|---|---|
| Python 3 | Core language |
| pandas / NumPy | Data manipulation |
| scikit-learn | Modeling, preprocessing, grid search |
| seaborn / matplotlib | Visualizations |
| baserush | Rapid baseline modeling (`quick_lm`, `quick_tree`, `lr_summary`) |
| Jupyter Notebook | Development environment |

---

## Key Findings

- Log-transforming the target variable (`log_REVENUE`) was essential — raw revenue's right skew degraded all model performances significantly
- GBM outperformed all linear models by capturing non-linear feature interactions, particularly between order volume and engagement efficiency metrics
- A train-test gap of 0.0494 confirms strong generalization with no meaningful overfitting
- ~20% of revenue variance remains unexplained, likely driven by unobserved factors such as geographic pricing, promotional history, and seasonality

---

## Limitations & Future Work

- **Weekly plan data** — 30% zero values and ambiguous scale (total uses vs. active weeks) limits discount behavior modeling; a binary flag was used as a workaround
- **Email engagement rate** — domain group proxies customer engagement, but actual survey/email response rates would better capture NPS-revenue relationships documented in the company white paper
- **Time dimension** — no date/time columns available; adding weekday, weekend, and seasonal order indicators would likely improve model performance
- **External validation** — model trained and tested on a single static snapshot; performance on future customer cohorts should be validated before production use

---

## Author

**Nils (Chen-Yu Liu)**  
MBA + MSBA Candidate — Hult International Business School, Boston  
Linkedin: https://www.linkedin.com/in/nils-chen-yu-liu/
