# Financial Transaction Fraud Detection Using Machine Learning
# Project Overview

With the rapid growth of digital payments, financial transaction fraud has become a serious concern for banks and financial institutions. Although fraudulent transactions occur rarely, the financial losses can be significant if they are not detected in time.

This project focuses on building a machine learning–based fraud detection system that can distinguish between legitimate and fraudulent transactions. The main challenge addressed is class imbalance, where fraudulent transactions form only a very small fraction of the total data.

# Problem Statement

Fraudulent transactions are extremely rare compared to normal transactions.
Traditional machine learning models perform poorly when evaluated only using accuracy.
The goal is to maximize fraud detection performance by focusing on Precision, Recall, and F1-score rather than accuracy alone.

# Dataset Description

Source: Kaggle – Credit Card Fraud Detection Dataset
The dataset contains anonymized transaction features.
Highly imbalanced:
Majority: Legitimate transactions
Minority: Fraudulent transactions
A smaller, processed version of the dataset was used to reduce computational cost.

# Tools & Technologies Used

Programming Language: Python
Libraries:
NumPy – Numerical computations
Pandas – Data manipulation
Matplotlib & Seaborn – Data visualization
Scikit-learn – Machine learning models
Imbalanced-learn – Handling class imbalance
Environment: Jupyter Notebook

# Data Preprocessing

Data preprocessing was a crucial step due to imbalance and outliers.
# Steps performed:

Shuffled the dataset to remove ordering bias
Handled class imbalance using:
Undersampling
SMOTE (Synthetic Minority Oversampling Technique)
Applied Robust Scaling to handle outliers effectively
Split the data into:
80% Training set
20% Testing set

# Machine Learning Models Implemented

The following models were trained and compared:

# Logistic Regression

Used as a baseline model
Simple and interpretable
Helps understand basic linear separation

# Random Forest Classifier

Ensemble-based model
Handles non-linear patterns well
Reduces overfitting compared to single decision trees

# Neural Network (MLP Classifier)

Multi-layer perceptron
Capable of capturing complex transaction patterns
Computationally more expensive

# Model Evaluation Metrics

Due to severe class imbalance, accuracy alone was misleading.
The following metrics were used:
Precision: How many detected frauds were actually fraud
Recall: How many real frauds were correctly detected
F1-Score: Balance between precision and recall
Confusion Matrix: Detailed error analysis

# Results & Findings

Random Forest delivered the best balance between precision and recall.
Robust scaling improved model stability.
Ensemble methods proved effective for structured, imbalanced datasets.
Accuracy alone failed to reflect true model performance.

# Key Learnings

Accuracy can be misleading in fraud detection problems.
Handling class imbalance is essential for meaningful results.
Precision and recall directly impact business outcomes:
High false positives annoy customers
Missed fraud leads to financial loss
Random Forest is a strong baseline for fraud detection tasks.

# Future Enhancements

Implement real-time fraud detection
Apply cost-sensitive learning where fraud has higher penalty
Experiment with advanced models like:
XGBoost
LightGBM
Deploy the model as a web API using Flask


# References

Kaggle Credit Card Fraud Detection Dataset
Open-source machine learning documentation
Scikit-learn & Imbalanced-learn resources
