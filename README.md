# Customer Churn & Retention Optimization

An end-to-end machine learning solution for predicting customer churn and translating churn propensity into a **budget-constrained retention strategy** optimized around **net retained revenue**.

The project covers customer churn prediction, exploratory data analysis, model comparison, risk-based targeting, what-if simulation, sensitivity analysis, and an experimentation/rollout strategy.

---

## Business Problem

A subscription business such as a telecom, OTT, or SaaS company is experiencing high customer churn risk over the next 30 days.

The business has a constrained retention budget:

* A retention offer can be provided to at most **30% of active customers**
* The offer costs **20% of the customer's monthly fee**
* The primary business objective is **net retained revenue**, rather than simply reducing the churn rate

The solution therefore needs to answer three questions:

1. **Who is likely to churn?**
2. **Who should receive a retention offer given the budget constraint?**
3. **Which targeting strategy is expected to generate the highest net retained revenue?**

---

## Solution Overview

The solution follows an end-to-end decisioning workflow:

```text
Customer Data
     │
     ▼
Data Quality & EDA
     │
     ▼
Churn Propensity Modeling
     │
     ├── Logistic Regression
     ├── Gradient Boosting
     └── XGBoost
             │
             ▼
       Model Selection
       based on AUC
             │
             ▼
      Customer Risk Scores
             │
             ▼
   Risk-based Targeting
       (≤ 30% budget)
             │
             ▼
   Revenue Impact Analysis
             │
             ▼
     What-if Simulation
       ├── Conservative
       └── Aggressive
             │
             ▼
   Sensitivity Analysis
             │
             ▼
    A/B Test & Rollout Plan
```

---

## Dataset

The project uses a **synthetic customer dataset** designed around the business scenario.

The dataset contains customer-level attributes such as:

* `customer_id`
* `tenure_months`
* `monthly_fee`
* `total_usage_last_30d`
* `support_tickets_last_90d`
* `has_outstanding_invoice`
* `discount_received_last_6m`
* `num_competing_offers_in_region`
* `is_premium_plan`
* `engagement_score`
* `region`
* `segment`
* `device_type`
* `churn_next_30d`

The synthetic data is designed to represent realistic relationships between customer behavior and churn risk.

---

## Project Tiers

All three tiers from the assignment have been completed.

### Tier 1 — Foundations

Completed:

* Business problem framing
* Prediction and decision objectives
* Exploratory data analysis
* Missing-value analysis
* Outlier detection
* Potential data leakage analysis
* Churn pattern analysis
* Production data-quality considerations

Key production considerations include non-random missingness, skewed/outlier-prone monthly fees, and strict prevention of target leakage through an as-of-date feature cutoff.

---

### Tier 2 — Modeling & Targeting

#### Churn Propensity Modeling

Three models are evaluated:

1. Logistic Regression
2. Gradient Boosting
3. XGBoost

A stratified **75/25 train-test split** is used for validation.

The boosted model with the highest test-set AUC is automatically selected as the model used for downstream scoring and targeting.

The notebook evaluates:

* ROC-AUC
* Classification performance
* Risk-decile churn rates
* Lift versus the overall churn rate
* Feature importance
* Logistic regression coefficients

#### Budget-Constrained Targeting

Customers are ranked using predicted churn probability.

The baseline targeting strategy selects the **top 30% highest-risk customers**, subject to the business budget constraint.

The analysis estimates:

* Customers targeted
* Expected churners
* Assumed retention uplift
* Customers saved
* Offer cost
* Retained revenue
* Net retained revenue impact

---

### Tier 3 — Simulation & Experimentation

Two retention strategies are simulated.

#### Scenario A — Conservative

Target the:

> **Top 10% highest-risk customers**

This strategy focuses the retention budget on customers with the strongest predicted churn risk.

#### Scenario B — Aggressive

Target the:

> **Top 30% highest-risk customers**

This strategy uses the maximum budget allowed by the business constraint.

Both scenarios estimate:

* Customers targeted
* Expected churners
* Estimated churners saved
* 3-month retained revenue
* Total offer cost
* Net gain

A retention-uplift sensitivity analysis is also included to understand how the preferred strategy changes under different assumptions.

---

## Key Business Assumptions

The main business assumptions are centralized in the notebook's `ASSUMPTIONS` dictionary so that the analysis can easily be rerun.

| Assumption                 |              Value |
| -------------------------- | -----------------: |
| Maximum targeted customers |                30% |
| Offer cost                 | 20% of monthly fee |
| Assumed retention uplift   |                35% |
| Conservative targeting     |            Top 10% |
| Aggressive targeting       |            Top 30% |
| Revenue horizon            |           3 months |

The **35% retention uplift is an assumption**, not a measured causal effect.

This is important because a churn prediction model identifies customers who are likely to churn; it does not by itself prove that offering them a discount will prevent churn.

---

## Experimentation Strategy

Before a full production rollout, the recommended approach is a randomized control/treatment experiment.

### Treatment

Eligible high-risk customers receive the retention offer.

### Control

A randomly selected control group is scored but does not receive the offer.

The experiment should measure the actual incremental impact of the retention offer.

### Primary KPI

**Net retained revenue**

```text
Net Retained Revenue =
Retained Revenue - Offer Cost
```

### Secondary KPIs

* Churn-rate reduction
* Incremental retention
* Offer redemption rate
* ROI
* Customer complaints
* Support-ticket volume

The initial rollout should start with the **Conservative Top-10% strategy** to limit budget exposure while validating the assumed retention uplift.

---

## Why This Approach?

A traditional churn model answers:

> "Who is likely to churn?"

This project goes one step further:

> "Given limited budget, who should we target and what is the expected financial impact?"

This distinction is important because the highest-risk customers are not necessarily the customers that generate the highest business value.

The final decision should therefore consider:

```text
Churn Risk
     +
Customer Value
     +
Offer Cost
     +
Expected Retention Uplift
     =
Business Decision
```

---

## Repository Contents

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── Churn_Retention_Assignment_Notebook.ipynb
├── data/
│   └── README.md
├── reports/
│   └── stakeholder_summary.pdf
└── .gitignore
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/churn-retention-optimization.git
cd churn-retention-optimization
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open:

```text
notebooks/Churn_Retention_Assignment_Notebook.ipynb
```

Run the notebook from top to bottom.

---

## Recommended Python Dependencies

```text
numpy
pandas
matplotlib
scikit-learn
xgboost
jupyter
```

---

## Limitations

This project is designed as a take-home analytical exercise and uses synthetic data.

Important limitations include:

1. The retention uplift is assumed rather than estimated from randomized treatment data.
2. Model performance is evaluated using a single stratified train/test split.
3. Revenue calculations use simplified assumptions about customer retention and monthly revenue.
4. The model predicts churn propensity but does not directly estimate treatment uplift or customer-level treatment effect.
5. Production deployment would require monitoring for data drift, concept drift, model performance degradation, and changes in customer behavior.
6. A production system should enforce strict feature time windows to prevent target leakage.

---

## Future Improvements

A production-ready version could extend the solution with:

* Time-based validation
* Cross-validation and hyperparameter tuning
* Model calibration
* SHAP-based explainability
* Customer Lifetime Value (CLV)
* Uplift modeling / causal ML
* Individual Treatment Effect estimation
* Real-time churn scoring
* Model monitoring and drift detection
* Automated retraining
* Experimentation platform integration
* Cost-sensitive optimization
* Multi-offer optimization

A particularly important next step would be moving from **churn propensity modeling** to **uplift modeling**, where the system predicts which customers are actually more likely to be retained *because of* the intervention.

---

## Conclusion

This project demonstrates an end-to-end approach to converting machine learning predictions into a constrained business decision.

Rather than optimizing solely for model accuracy, the solution connects:

**Data → Prediction → Targeting → Revenue → Simulation → Experimentation**

The ultimate objective is to identify a retention strategy that maximizes **incremental net retained revenue** while respecting the company's retention budget.



---

## Author

**Ankush Shrikhande**

Data Scientist | Machine Learning | GenAI | AI Decisioning
