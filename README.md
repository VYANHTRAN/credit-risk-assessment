# Credit Risk Assessment using Taiwanese Credit Card Data

This project focuses on building and evaluating machine learning models to predict credit card default using a publicly available dataset of Taiwanese credit card holders from 2005.

## Overview

The primary goal is to assess the likelihood of a credit card customer defaulting on their payment in the subsequent month. The project involves:

1.  **Exploratory Data Analysis (EDA):** Understanding the data distribution, patterns, and correlations.
2.  **Feature Engineering:** Creating new, informative features from the original dataset based on demographic information, credit utilization, payment behavior, delinquency history, and temporal trends.
3.  **Data Preprocessing:** Handling categorical features (One-Hot Encoding), scaling numerical features (StandardScaler), and implementing imputation strategies (though no missing values were found in the original dataset).
4.  **Handling Class Imbalance:** Using the Synthetic Minority Over-sampling Technique (SMOTE) within the training pipeline to address the imbalance between defaulters and non-defaulters.
5.  **Model Training and Tuning:** Implementing Logistic Regression and Random Forest classifiers. Using GridSearchCV with 5-fold cross-validation to optimize hyperparameters, focusing on the ROC AUC metric.
6.  **Model Evaluation:** Assessing the performance of the tuned models on an unseen test set using metrics like Accuracy, Precision, Recall, F1-Score, ROC AUC, and Precision-Recall (PR) AUC.

## Dataset

The project utilizes the "Default of Credit Card Clients Dataset" which contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

*   **Source:** UCI Machine Learning Repository 
*   **File:** `taiwan_default_credit_dataset.csv` located in the `/data` directory.

## Methodology

The analysis follows these steps:

1.  Load and perform initial cleaning on the dataset.
2.  Conduct EDA to visualize distributions and relationships (Age, Default Rates by Age, Target Class Distribution).
3.  Engineer a comprehensive set of features capturing various aspects of customer behavior (Demographics, Utilization, Payment Behavior, Delinquency, Trends, Interactions).
4.  Split the data into stratified training (70%) and testing (30%) sets.
5.  Define preprocessing pipelines using `scikit-learn`'s `Pipeline` and `ColumnTransformer` for imputation, scaling, and encoding.
6.  Integrate SMOTE into the pipelines using `imblearn`'s `Pipeline` to handle class imbalance correctly during cross-validation and training.
7.  Train Logistic Regression and Random Forest models.
8.  Perform hyperparameter tuning using `GridSearchCV` optimizing for `roc_auc`.
9.  Evaluate the best-tuned models on the held-out test set using various classification metrics, paying particular attention to ROC AUC and PR AUC due to class imbalance.

## Results

Both the tuned Logistic Regression and Random Forest models demonstrated a reasonable ability to predict credit card default, performing significantly better than random guessing.

*   The Random Forest model showed a slight performance advantage over Logistic Regression on the test set, particularly in terms of ROC AUC (0.768 vs 0.760) and PR AUC (0.511 vs 0.491).
*   The use of SMOTE helped achieve decent Recall scores (around 0.6), indicating the models learned patterns from the minority (default) class.
*   Moderate Precision scores highlight the trade-off between identifying defaulters and minimizing false positives (incorrectly flagging non-defaulters).
*   The results underscore the importance of thoughtful feature engineering and appropriately addressing class imbalance in credit risk modeling.

For detailed results and discussion, please refer to the Jupyter notebook (`notebooks/credit_risk_assessment.ipynb`) and the report (`report/credit_risk_assessment.pdf`).
