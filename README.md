Environment Setup

Python 3.10 or a compatible recent Python version is recommended.

Install the required packages using:

pip install -r requirements.txt

The main Python dependencies include:

NumPy
pandas
scikit-learn
XGBoost
LightGBM
CatBoost
matplotlib
SHAP
PySR
openpyxl
python-docx

The complete dependency list is provided in:

requirements.txt
Dataset

The dataset used in the analyses is located at:

data/Kombinations.xlsx

Each numerical case contains geometric and material parameters together with the corresponding natural-frequency outputs obtained from finite element modal analyses.

The principal input variables include:

B: plan width
L: plan length
H: structural height
E: modulus of elasticity
rho: masonry wall density

The corresponding natural-frequency outputs are provided in the dataset.

Dataset Generation

The numerical database was generated from finite element modal analyses performed in Abaqus.

The numerical models represent masonry building archetypes with variations in geometric and material parameters. For each case, modal analysis was carried out and the resulting natural frequencies were stored together with the corresponding input parameters.

The present repository contains the finalized numerical dataset used for the machine-learning, interpretability, and regression analyses.

The Abaqus finite element model-generation files are not included in this repository. Therefore, the repository reproduces the complete data-driven workflow starting from the finalized finite-element-generated dataset.

Machine-Learning Workflow

The main machine-learning analysis is implemented in:

src/f1_f2_group_nested_final_test_only_code.py

The script performs:

Data loading and preprocessing
Geometry-group definition
Group-based nested cross-validation
Hyperparameter optimization
Model training
Independent outer-fold performance evaluation
Prediction-table generation
Parity-plot generation
Permutation feature importance
SHAP-based model interpretation
Export of numerical results

The evaluated machine-learning models include:

Random Forest
XGBoost
LightGBM
CatBoost
Multi-Layer Perceptron
Group-Based Nested Cross-Validation

To avoid information leakage between numerical realizations sharing the same structural geometry, geometry groups are defined using the combined values of:

B, L, and H

The evaluation procedure uses:

5-fold outer GroupKFold for independent performance assessment
5-fold inner GroupKFold for hyperparameter optimization

All numerical realizations corresponding to the same geometry group are retained within the same fold.

Accordingly, the reported outer-test results represent prediction performance for previously unseen geometry groups rather than random sample-wise interpolation.

Hyperparameter Optimization

Hyperparameter optimization is performed independently within each outer-training fold using randomized search and group-based inner cross-validation.

The machine-learning hyperparameters are selected using the training portion of each outer fold only.

After the nested cross-validation assessment, the models may also be retuned using the complete dataset with group-based cross-validation to obtain final configurations for reproducibility and interpretation.

Evaluation Metrics

Model performance is evaluated using:

Coefficient of determination, R²
Mean Absolute Error, MAE
Root Mean Square Error, RMSE
Relative-error measures where applicable

The main reported performance values are obtained from the held-out outer test folds.

Parity plots are also generated to provide a graphical comparison between the finite-element reference frequencies and the model predictions.

Interpretability Analysis

Two complementary model-interpretation approaches are used.

Permutation Feature Importance

Permutation importance is calculated on the held-out outer test folds.

For each feature, its values are randomly permuted while the remaining variables are retained unchanged. The resulting decrease in predictive performance is used to quantify feature importance.

This procedure is repeated within each outer fold and subsequently summarized across the outer folds.

SHAP Analysis

SHAP analysis is performed for the final tree-based models:

Random Forest
XGBoost
LightGBM
CatBoost

The SHAP results describe how the fitted models use the different geometric and material inputs when producing predictions.

Permutation importance and SHAP values should be interpreted as model-based predictive associations within the investigated numerical parameter domain rather than as causal physical effects.

Linear and Symbolic Regression

The regression-based analysis is implemented in:

src/group_based_outer_cv_linear_symbolic_regression_final_fixed.py

This script performs:

Linear regression
Symbolic regression using PySR
Geometry-grouped outer cross-validation
Regression performance evaluation
Final equation generation
Prediction-table generation
Parity-plot generation
Export of numerical results

Symbolic regression is used to derive an explicit nonlinear mathematical expression while balancing predictive accuracy and algebraic simplicity.

The regression equations are empirical, unit-specific formulations calibrated using the numerical database and should be used only within the stated input ranges and unit conventions.

Running the Codes
Machine-Learning Models

From the repository directory:

cd src
python f1_f2_group_nested_final_test_only_code.py
Linear and Symbolic Regression

From the src directory:

python group_based_outer_cv_linear_symbolic_regression_final_fixed.py

Before running the scripts, verify that the dataset path used in the Python files points to:

../data/Kombinations.xlsx

or update the input path according to your local directory structure.

Reproducing the Analyses

The principal analyses reported in the manuscript can be reproduced using the following scripts.

f1_f2_group_nested_final_test_only_code.py

Reproduces:

Machine-learning training
Group-based nested cross-validation
Hyperparameter optimization
Outer-test predictions
Performance metrics
Parity plots
Permutation feature importance
SHAP analysis
group_based_outer_cv_linear_symbolic_regression_final_fixed.py

Reproduces:

Linear regression
Symbolic regression
Group-based outer cross-validation
Regression-performance metrics
Final explicit equations
Prediction tables
Regression parity plots

The repository is intended to reproduce the computational analyses starting from the finalized numerical dataset.

Outputs

Depending on the analysis script, the generated outputs include:

Excel result files
Prediction tables
Performance summaries
Hyperparameter information
Parity plots
Feature-importance plots
SHAP plots
Regression equations
Additional diagnostic figures

The generated outputs can be stored in:

results/

and graphical outputs in:

figures/

If different output paths are defined in the scripts, they can be modified according to the user's local directory structure.

Reproducibility Notes

The following points should be considered when reproducing the results:

The same dataset should be used without changing the geometry-group identifiers.
Geometry groups are defined using the combined values of B, L, and H.
Group-based splitting must be retained to avoid leakage between realizations sharing the same geometry.
The same random-state settings should be retained when exact computational reproducibility is required.
PySR results may depend on the installed Julia/PySR environment and software version.
The finite element model-generation stage is not reproduced by this repository; the provided dataset represents the starting point of the data-driven analysis.
Data Availability

The numerical dataset and Python source codes required to reproduce the machine-learning and regression-based analyses are provided in this repository.

Repository:

https://github.com/aliekbersever1/Masonry-Buildings-ML
Citation

If you use this repository, dataset, or source code, please cite the associated publication:

Machine Learning and Symbolic Regression Models for Predicting the Natural Frequencies of Masonry Buildings

Full bibliographic information will be added after publication.

License

This project is distributed under the MIT License.

See:

LICENSE
