# Masonry Buildings ML

Machine learning and symbolic regression models for predicting the natural frequencies of masonry building archetypes.

---

## Overview

This repository contains the Python codes and numerical dataset developed for the study:

**Machine Learning and Symbolic Regression Models for Predicting the Natural Frequencies of Masonry Buildings**

The repository includes the complete data-driven workflow used for:

- Group-based nested cross-validation
- Hyperparameter optimization
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Multi-Layer Perceptron (MLP)
- Linear Regression
- Symbolic Regression using PySR
- Permutation feature importance
- SHAP-based model interpretation
- Performance evaluation and graphical comparison

The numerical dataset was generated from finite element modal analyses performed in Abaqus and was subsequently used for machine-learning and regression-based prediction of the natural frequencies.

---

## Repository Structure

```text
Masonry-Buildings-ML
│
├── data/
│   └── Kombinations.xlsx
│
├── src/
│   ├── f1_f2_group_nested_final_test_only_code.py
│   └── group_based_outer_cv_linear_symbolic_regression_final_fixed.py
│
├── results/
│
├── figures/
│
├── requirements.txt
├── LICENSE
└── README.md
