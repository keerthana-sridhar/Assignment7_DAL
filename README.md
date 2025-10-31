# 🧠 DA5401 A7: Multi-Class Model Selection using ROC and Precision–Recall Curves

## 🎯 Objective

This assignment explores **Receiver Operating Characteristic (ROC)** and **Precision–Recall Curves (PRC)** for model selection in a **multi-class classification** setting using the **UCI Landsat Satellite dataset**.

The goal is to compare the performance of diverse classifiers — ranging from simple baselines to more complex models — and interpret their behavior beyond simple accuracy metrics.

### Key Tasks:
- Train multiple models of varying complexity  
- Evaluate each using **Accuracy**, **Weighted F1**, **ROC-AUC**, and **Average Precision (PRC-AP)**  
- Interpret results to identify the **best and worst** models  
- Provide **visual ROC and PRC comparisons** across all models  
- Experiment with additional “brownie point” models for deeper insights  

---

## 📦 Dataset Description

| Feature | Description |
|----------|--------------|
| **Dataset** | UCI Landsat Satellite |
| **Input features** | 36 numerical attributes per pixel |
| **Output classes** | 6 distinct land cover types (e.g., soil, vegetation, water) |
| **Task type** | Multi-class classification |
| **Challenge** | High-dimensional data and significant class overlap |

---

## 🧪 Models Compared

| Model | Library Reference | Expected Performance |
|:------|:------------------|:---------------------|
| **K-Nearest Neighbors (KNN)** | `sklearn.neighbors.KNeighborsClassifier` | Moderate to Good |
| **Decision Tree Classifier** | `sklearn.tree.DecisionTreeClassifier` | Moderate |
| **Dummy Classifier (Prior)** | `sklearn.dummy.DummyClassifier` | Baseline (Random) |
| **Logistic Regression** | `sklearn.linear_model.LogisticRegression` | Good Linear Baseline |
| **Gaussian Naive Bayes** | `sklearn.naive_bayes.GaussianNB` | Poor to Moderate |
| **Support Vector Machine (SVC)** | `sklearn.svm.SVC` | Good (with probability=True) |
| *(Brownie Point)* **Random Forest** | `sklearn.ensemble.RandomForestClassifier` | Strong Nonlinear Model |
| *(Brownie Point)* **XGBoost** | `xgboost.XGBClassifier` | Excellent Performance |
| *(Brownie Point)* **Decision Stump** | `sklearn.tree.DecisionTreeClassifier(max_depth=1)` | Poor / Underfit |

---
