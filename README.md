# 🛳️ Titanic Classification Project

This project is part of the **Kaggle Titanic - Machine Learning from Disaster** competition.  
The goal is to build a **classification model** that predicts whether a passenger survived the Titanic disaster based on features such as age, gender, ticket class, and fare.

---

## Project Structure
Titanic Survival Prediction – Ensemble ML Approach 🚢

This repository contains a machine learning pipeline for predicting passenger survival on the Titanic. The project demonstrates data preprocessing, feature engineering, and ensemble modeling using multiple classifiers, including logistic regression, gradient boosting, XGBoost, CatBoost, and SVM.

## Features

# Data Loading & Exploration

Load training and test datasets using pandas.

Inspect data for missing values and statistical summaries.

# Data Preprocessing

Handle missing values intelligently (median, mode, grouping by categories).

Drop irrelevant columns (Cabin, Ticket, etc.) to reduce noise.

Encode categorical features (Sex, Embarked, Title) into numeric values.

Feature engineering:

FamilySize

IsAlone

AgeBin

# Modeling & Ensemble

Models used:

Logistic Regression

LightGBM

XGBoost

CatBoost

Support Vector Machine (SVM)

10-fold cross-validation for robust performance evaluation.

Predictions blended via RidgeClassifier for final ensemble output.

# Scaling & Evaluation

StandardScaler applied for models sensitive to feature scale (LOG, SVM).

Out-of-fold predictions stored for meta-model stacking.

# Submission

Generates a submission_ensemble.csv ready for Kaggle or other evaluation.

## Usage

Place train.csv and test.csv in the project directory.

Run the notebook or Python script:

python titanic_classification.py

Check the generated submission_ensemble.csv.

## Results

Combines predictions from multiple models to improve accuracy.

Meta-model (RidgeClassifier) used to blend predictions.

Flexible pipeline to add more features or models.

## Improvements Are Appreciated 

Feature engineering (e.g., adding new interaction terms).

Hyperparameter tuning of individual models.

Experiment with other ensemble techniques or stacking strategies.

Handling rare titles in Title feature more intelligently.

## Acknowledgements

Inspired by Kaggle’s Titanic dataset competition.

Built using Python, pandas, scikit-learn, LightGBM, XGBoost, and CatBoost.