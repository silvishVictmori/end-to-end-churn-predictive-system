# MSIN0097 Predictive Analytics — Customer Churn Prediction System (End-to-End)

This repository contains an end-to-end predictive analytics pipeline for customer churn prediction. It follows the coursework workflow structure (Steps 1–6) and is designed to be reproducible, auditable, and business-interpretable: raw data is preserved, preprocessing is fitted only on training data, model selection is evidence-based, and final outputs include both performance results and actionable churn drivers.

---

## 1) Business Context and Objective

Customer churn is costly because it directly reduces recurring revenue and increases acquisition pressure. The objective of this system is to **predict churn risk** so the business can:

- identify customers at elevated churn risk early,
- prioritise retention actions (email/call offers, service recovery, billing support),
- measure trade-offs between catching churners (recall) and operational cost (false positives).

This is a **risk scoring** system first (probability ranking), with a configurable decision threshold for operational use.

---

## 2) Predictive Problem Definition

**Task type:** Binary classification  
**Target variable:** `churn` (1 = churned, 0 = retained)  
**Prediction output:** churn probability `p(churn=1)` and a binary flag based on an operating threshold.

### Success Criteria (Metrics)

Because churn is relatively rare (~10%), accuracy is not a good primary metric. This project prioritises:

- **PR-AUC** (positive-class ranking under imbalance)
- **Recall and F1** (ability to detect churners at a chosen operating point)
- **ROC-AUC** (overall ranking quality)
- **Brier score** (probability calibration quality)

### Operational Constraint / Cost Assumption

False negatives (missed churners) are assumed more costly than false positives, because missed churners mean lost revenue and no intervention opportunity. Therefore, the operating threshold is selected to support **high recall** while keeping outreach cost manageable.

---

## 3) Dataset and Provenance

**Dataset:** `customer_churn_business_dataset.csv`  
**Location:** `data/raw/`  
Raw data is never modified. All transformations and splits are stored in `data/processed/`.

**Dataset Summary**
The model is trained on a dataset of 10,000 customer records with 32 features.

Target Variable: churn (Binary classification).

Churn Rate: 10.21%, indicating a significant class imbalance.

Feature Categories: Engagement behavior (logins, session time), financial activity (revenue, payment failures), service interactions (support tickets), and satisfaction metrics (CSAT, NPS).

**Key Pipeline Features**
Statistically Valid Preprocessing: Implements a reproducible pipeline with strict train-validation-test split discipline to avoid procedural leakage.

Handling Missing Data: Missing values in complaint_type (20.45%) are treated as informative (indicating no complaint) rather than random noise.

Business-Centric Evaluation: Given the class imbalance, the model prioritizes PR-AUC, Precision, Recall, and F1-score over simple accuracy.

Agent Collaboration: Developed using a structured "plan → delegate → verify → revise" framework with AI agents, where all contributions were experimentally validated.

### Data Constraints and Notes

- The dataset is a single-table customer-level snapshot (no event timestamps), so the pipeline cannot simulate a true time-based deployment split.
- Some variables (e.g., survey and support outcomes) can be **close to the churn event** and may create leakage risk depending on how the business defines “prediction time”.

---

## 4) Workflow Overview (Coursework Steps 1–6)

### Step 1 — Obtain dataset and frame the predictive problem
- Define target and task type
- Define evaluation metrics and operational constraints
- State assumptions and limitations
- Document what was supported by an agent and what was manually verified

### Step 2 — Explore the data to gain insights
- Visual EDA: distributions, missingness, class imbalance, potential leakage risk, outliers
- Identify data quality issues and modelling pitfalls
- Summarise implications for modelling and evaluation

### Step 3 — Prepare the data (reproducible pipeline)
- Train/validation/test splits (stratified)
- Reproducible preprocessing pipeline (numeric + categorical)
- Validation checks (missingness, bounds)
- Save processed splits and split metadata

### Step 4 — Explore different models and shortlist
- Train multiple baseline models with consistent preprocessing
- Compare models on validation using the same metrics
- Produce standard plots (ROC/PR/calibration)
- Shortlist models based on evidence and stability (not “vibes”)

### Step 5 — Fine-tune and evaluate shortlisted models
- Tune shortlisted models (validation strategy, early stopping if applicable)
- Optimise decision threshold based on validation metrics aligned with business goal
- Robust evaluation and error analysis:
  confusion matrix, calibration, failure modes by segment / feature ranges
- Document at least one agent error that was caught and corrected

### Step 6 — Present the final solution
- Retrain the selected model on train+validation
- Evaluate once on test (unseen during selection)
- Save final model artifact, metrics, predictions, and driver analysis
- Produce business-oriented summary and “model card” (intended use, caveats)

---

## 5) Repository Structure


```
msin0097-e2e-predictive-system/
│
├── data/
│   ├── raw/                # raw dataset (unchanged)
│   └── processed/          # saved splits and metadata
│
├── notebooks/
│   └── e2e_workflow.ipynb  # narrative analysis (Steps 1–6)
│
├── scripts/                # runnable workflow scripts
│   ├── prepare_data.py     # Step 3: splits + preprocess metadata
│   ├── train.py            # Step 4: baseline model comparison
│   ├── tune.py             # Step 5: tuning + threshold + error analysis
│   └── finalize.py         # Step 6: final retrain + test + artifact export
│
├── src/                    # reusable pipeline utilities
│   ├── config.py           # constants and paths
│   ├── data.py             # load/save helpers
│   ├── interpret.py
│   ├── preprocess.py
│   ├── utils.py
│   └── evaluate.py         # metrics + plotting helpers
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── runs/               # timestamped runs
│
├── requirements.txt
├── .gitignore
└── README.md
```


**Note:** `outputs/` and virtual environments are typically excluded from Git to keep the repo lightweight.

---

## 6) Quickstart (How to Run)

### 6.1 Environment setup (Python 3.11 recommended)

Create/activate a Python 3.11 environment, then:

pip install -r requirements.txt

#### Limitations & Risks
Temporal Leakage: The lack of timestamps in the current dataset prevents full verification of whether satisfaction metrics were recorded before or after the churn event.

Synthetic Data: The dataset may not capture the full complexity or operational noise of a live production environment.

Non-Causal: Identified feature importance reflects associations with churn rather than direct causal influence.

#### Future Improvements
Incorporate timestamps for time-based splits to eliminate temporal leakage.

Add fairness checks across demographic groups (e.g., gender or country) prior to operational deployment.

Implement ongoing monitoring for model drift as customer behavior evolves.