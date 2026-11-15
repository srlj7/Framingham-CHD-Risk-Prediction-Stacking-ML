#  An End-to-End Machine Learning Framework for 10-Year Coronary Heart Disease (CHD) Risk Prediction


> **Full Research Paper & Experimental Codebase**  
> **Authors:** Saif Asaad Naeem, Mustafa Saleh Jodah, Hussein Ali Muhsen  
> **Supervisor:** Prof. Ali Gamal  
> **Institution:** Department of Artificial Intelligence Engineering, College of Engineering, Al-Maaqal University, Basra, Iraq

---

##  Executive Summary

Coronary Heart Disease (CHD) is the leading cause of mortality worldwide. Predicting 10-year CHD risk from routine clinical and laboratory records is severely challenged by **conditional missing values (13.73%)**, **extreme right-tailed pathological skewness (glucose skew = +6.40)**, **severe multicollinearity ($r = 0.784$)**, and an acute **5.58 : 1 target class imbalance (84.80% Healthy vs. 15.20% CHD)** that causes standard baseline classifiers to suffer a catastrophic collapse in sensitivity (**Recall = 6.21%**).

Rather than relying on arbitrary, single-tool choices, this repository implements a rigorous **Multi-Algorithm Comparative Benchmarking Framework** on the **Framingham Heart Study** cohort ($N = 4,238$ patients, $16$ attributes), systematically evaluating **5 competing algorithms at every pipeline stage** (31 distinct operators in total) to select the mathematically and clinically optimal pipeline.

###  Key Achievements at a Glance
* **Test Recall (Sensitivity):** **98.87%** (*703 out of 711 true CHD patients identified; only 8 missed*)
* **Test Precision (PPV):** **98.18%** (*only 13 false alarms out of 519 healthy controls*)
* **Test F1-Score:** **98.53%** (*5-Fold Stratified CV F1 = 98.22%*)
* **Test ROC-AUC:** **99.88%** (*Average Precision AP = 1.00*)
* **Test Accuracy:** **98.29%**

---

##  System Architecture & Six-Stage Pipeline

![System Architecture](figures/fig1_system_architecture.png)

### Summary of Multi-Algorithm Benchmarks Across All 6 Stages

| Pipeline Stage | Tested Algorithms (5 per Stage) | Winning Algorithm | Key Quantitative & Clinical Outcome |
| :--- | :--- | :--- | :--- |
| **Stage 1: Zero Audit & Imputation** | Mean, Median, KNN ($k=5$), MICE (`BayesianRidge`), **Hybrid MissForest** | **Hybrid MissForest** (`RF + Domain Rules`) | Lowest reconstruction error (**Avg NRMSE = 0.8848**); resolved the 29 active-smoker zero-median paradox (**Cigs RMSE = 9.62** vs. 18.85). |
| **Stage 2: Distribution Normalization** | Sqrt, Log1p, Box-Cox, **Yeo-Johnson**, Quantile Normal | **Yeo-Johnson Power Transform** | Reduced mean absolute skewness by **96%** (`1.7121` $\rightarrow$ `0.0689`) without deleting critical diabetic/hypertensive patients. |
| **Stage 3: Clinical FE & Selection** | Poly Degree-2, Mutual Info, ANOVA F-Test, L1 Lasso, **Clinical FE + RFE** | **10 Clinical Markers + RFE ($k=20$)** | Engineered `age_sysBP_index` achieved **$r = +0.2662$** (surpassing all raw features); RFE boosted Recall to **68.47%** and pruned 5 redundant cols. |
| **Stage 4: Scaling & Dim. Reduction** | MinMax, MaxAbs, YJ+Std, Robust, **StandardScaler** \| FA, Agglom, PCA, FastICA, **LDA+PCA** | **StandardScaler** & **Parallel LDA + PCA** | Preserved Z-score distance of high-risk outliers (**F1 = 39.57%**); Parallel `FeatureUnion(1 LDA || 11 PCA)` captured **99.53%** variance. |
| **Stage 5: Class Imbalance (5.58:1)** | Standard SMOTE, ADASYN, SMOTE+Tomek, Borderline-SMOTE, **Hybrid SMOTE+ENN** | **Hybrid SMOTE + ENN** | Purged ambiguous boundary instances; baseline sensitivity skyrocketed from **6.21%** to **93.95%** (**F1 = 89.84%**, **AUC = 95.12%**). |
| **Stage 6: Two-Stage Tuning & Stacking** | `ExtraTrees`, `RandomForest`, `HistGB`, `KNN`, `SVC` + **Two-Level Stacking** | **Stacking (Top 4 Base + Tuned SVC Meta)** | Continuous `RandomizedSearchCV ()` $\rightarrow$ Exhaustive `GridSearchCV []` propelled SVC by **+17.19% F1**; Stacking reached **98.87% Recall**. |

---

##  Engineered Cardiovascular & Hemodynamic Features

Guided by cardiovascular pathophysiology, 10 clinical interaction markers were synthesized from the 15 raw attributes:

| # | Engineered Feature | Formula | Target Correlation ($r$) | RFE Retained? | Clinical & Physiological Meaning |
| :-: | :--- | :--- | :---: | :---: | :--- |
| 1 | **`age_sysBP_index`** | $(\text{age} \times \text{sysBP}) / 100$ | **+0.2662 (Rank #1)** |  Yes | Cumulative vascular aging & arterial wall stress index |
| 2 | **`age_glucose_index`** | $(\text{age} \times \text{glucose}) / 100$ | **+0.2101** |  Yes | Age-compounded glycemic endothelial damage marker |
| 3 | **`framingham_risk_count`** | $\sum \text{Clinical Risk Flags}$ | **+0.2098** |  Pruned | Syndromic cardiometabolic risk factor burden |
| 4 | **`pulse_pressure`** | $\text{sysBP} - \text{diaBP}$ | **+0.2068** |  Yes | Direct clinical proxy for large-artery & aortic stiffness |
| 5 | **`log_gluc_sysBP`** | $\ln(1+\text{gluc}) \cdot \ln(1+\text{sysBP})$ | **+0.1985** |  Yes | Non-linear log interaction between hyperglycemia & hypertension |
| 6 | **`MAP`** | $\text{diaBP} + (\text{sysBP} - \text{diaBP})/3$ | **+0.1900** |  Yes | Mean arterial pressure governing coronary organ perfusion |
| 7 | **`rate_pressure_product`** | $(\text{sysBP} \times \text{heartRate})/1000$ | **+0.1552** |  Yes | Myocardial workload & oxygen demand ($MVO_2$) |
| 8 | **`sys_dia_ratio`** | $\text{sysBP} / (\text{diaBP} + \epsilon)$ | **+0.1445** |  Yes | Systolic-to-diastolic hemodynamic balance ratio |
| 9 | **`bmi_chol_index`** | $(\text{BMI} \times \text{totChol}) / 1000$ | **+0.1056** |  Yes | Synergistic dyslipidemia-obesity atherogenic index |
| 10 | **`pack_years_proxy`** | $\text{cigsPerDay} \times (\text{age}-18)/20$ | **+0.1048** |  Pruned | Estimated lifetime cumulative smoking pack-years exposure |

---

##  Final Master Leaderboard (5-Fold Stratified CV & Unseen Test Set)

Each classifier underwent Stage-1 continuous exploration via **`RandomizedSearchCV ()`** (`scipy.stats.randint`, `uniform`, `loguniform`) followed by Stage-2 exhaustive exploitation via **`GridSearchCV []`** optimizing F1-score under 5-Fold Stratified Cross-Validation:

| Rank | Model / Configuration | Optimal Hyperparameters (`GridSearchCV []`) | CV Recall | CV Prec | CV F1 | Test Recall | Test Prec | Test F1 | Test ROC-AUC | Test Acc |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| — | **Imbalanced Baseline** | Raw Unprocessed Data | 6.21% | 54.72% | 11.11% | — | — | — | 70.98% | 84.95% |
| 6 | **RandomForest** *(Excluded)* | `n_est=168, depth=28, feat=0.304` | 94.30% | 88.35% | 91.22% | 94.37% | 90.07% | 92.17% | 96.83% | 90.73% |
| 5 | **HistGradientBoosting** *(Base 4)* | `iter=453, lr=0.043, depth=22, leaves=77` | 95.36% | 91.38% | 93.32% | 95.36% | 94.17% | 94.76% | 98.37% | 93.90% |
| 4 | **ExtraTrees** *(Base 3)* | `n_est=168, depth=33, feat=0.324` | 95.92% | 91.68% | 93.74% | 96.62% | 93.47% | 95.02% | 98.58% | 94.15% |
| 3 | **KNN Classifier** *(Base 2)* | `n_neighbors=2, p=1.81, leaf=30, uniform` | 97.40% | 96.08% | 96.73% | 98.73% | 96.16% | 97.43% | 97.09% | 96.99% |
| 2 | **SVC (RBF)** *(Base 1 & Meta)* | `C=3.905, gamma=0.6376, kernel='rbf'` | 96.45% | 97.76% | 97.09% | 98.45% | 98.04% | 98.25% | 99.78% | 97.97% |
| **1** | **Stacking Ensemble (Winner)** | **Meta-SVC (`C=1.0, γ=0.0682, passthrough=True`)** | **98.66%** | **97.78%** | **98.22%** | **98.87%** | **98.18%** | **98.53%** | **99.88%** | **98.29%** |

---

##  Getting Started & Reproducibility

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/Framingham-CHD-Risk-Prediction-Stacking-ML.git
cd Framingham-CHD-Risk-Prediction-Stacking-ML
