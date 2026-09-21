"""
CS 171 - Assignment 2: Linear Regression, Regularization, and Logistic Regression
Starter Code

The data loading, train/test splitting, and standardization are done for you.
Fill in the sections marked TODO. See the assignment PDF for the details of
each step.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes, load_breast_cancer

np.set_printoptions(precision=2, suppress=True)


# =============================================================
#  PART A: LINEAR REGRESSION
#  Dataset: Diabetes Progression
#  Goal: Predict a patient's disease progression (a number)
#        from 2 medical measurements.
# =============================================================

# -------------------------------------------------------------
#  DATA LOADING (do not modify)
# -------------------------------------------------------------
diabetes = load_diabetes()
X_reg = diabetes.data[:, [2, 8]]   # Use only BMI and Serum_s5
y_reg = diabetes.target            # Disease progression (continuous)
feature_names = ["BMI", "Serum_s5"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.25, random_state=7
)

print("=" * 55)
print("PART A: LINEAR REGRESSION -- Diabetes Dataset")
print("=" * 55)
print(f"Training samples: {len(X_train_r)}")
print(f"Test samples:     {len(X_test_r)}")
print(f"Features:         {feature_names}")
print()


# =============================================================
#  STEP 1: Fit a Linear Regression model (8 points)
#
#  TODO: Create a LinearRegression model, fit it on the
#        training data, and print the intercept, the
#        coefficients, and the full equation.
# =============================================================
print("-" * 55)
print("STEP 1: Fit the model")
print("-" * 55)

# YOUR CODE HERE
linear_model_fit = LinearRegression().fit(X_train_r, y_train_r)
# intercept
print(f"Intercept: {linear_model_fit.intercept_}")
# coefficients, this is the slope of the equation
print(f"Coefficients: {linear_model_fit.coef_}")
# full equation
print(f"Full Equation: Y = {linear_model_fit.intercept_} + {linear_model_fit.coef_[0]} * BMI + {linear_model_fit.coef_[1]} * Serum_s5")

print()


# =============================================================
#  STEP 2: Predict on the test set (6 points)
#
#  TODO: Use your model to predict on X_test_r. Print the
#        first 10 predicted values and the first 10 true
#        values side by side.
# =============================================================
print("-" * 55)
print("STEP 2: Predict and compare")
print("-" * 55)

# YOUR CODE HERE
true_values = y_test_r
predicted_values = linear_model_fit.predict(X_test_r)
print(f"True Values: {true_values[:10]}")
print(f"Predicted Values: {predicted_values[:10]}")
print()

# =============================================================
#  STEP 3: Compute MSE and MAE (10 points)
#
#  TODO: Using the predictions and true values, compute MSE
#        and MAE BY HAND using NumPy. Do NOT use sklearn.metrics.
#        Also print the largest absolute error in the test set.
# =============================================================
print("-" * 55)
print("STEP 3: Compute MSE and MAE")
print("-" * 55)

# YOUR CODE HERE
# Computing MSE
MSE = np.sum((true_values - predicted_values) ** 2) / len(true_values)
print(f"MSE: {MSE}")

# Computing MAE
MAE = np.sum(np.abs(true_values - predicted_values)) / len(true_values)
print(f"MAE: {MAE}")

max_abs_error = np.max(np.abs(true_values - predicted_values))
print(f"Maximum Absolute Error: {max_abs_error}")

print()


# =============================================================
#  STEP 4: Interpret the results (6 points)
# =============================================================
print("-" * 55)
print("STEP 4: Interpretation")
print("-" * 55)

# (No code needed -- answer in your PDF report)

print()


# =============================================================
#  PART B: REGULARIZATION
#  Dataset: Diabetes Progression, all 10 features
#  Goal: See what the L2 (Ridge) and L1 (Lasso) penalties do
#        to the coefficients and to the training and test error.
# =============================================================

# -------------------------------------------------------------
#  DATA LOADING (do not modify)
#  Same split as Part A. The features are standardized using
#  the training set. Use Z_train and Z_test in this part.
# -------------------------------------------------------------
X_all = diabetes.data
all_names = list(diabetes.feature_names)   # age sex bmi bp s1 s2 s3 s4 s5 s6

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_all, y_reg, test_size=0.25, random_state=7
)
scaler = StandardScaler().fit(X_train_a)
Z_train = scaler.transform(X_train_a)
Z_test = scaler.transform(X_test_a)

print("=" * 55)
print("PART B: REGULARIZATION -- Diabetes Dataset, 10 features")
print("=" * 55)
print(f"Training samples: {len(Z_train)}")
print(f"Test samples:     {len(Z_test)}")
print(f"Features:         {all_names}")
print()


# =============================================================
#  STEP 5: Baseline with all 10 features, no penalty (8 points)
#
#  TODO: Fit a LinearRegression model on (Z_train, y_train_a).
#        Print the coefficients with their names, the sum of
#        their absolute values, and the training and test MSE.
# =============================================================
print("-" * 55)
print("STEP 5: Baseline (no penalty)")
print("-" * 55)

# YOUR CODE HERE
standard_linear_model = LinearRegression().fit(Z_train, y_train_a)

# coefficients, this is the slope of the equation
for i in range(len(all_names)):
    print(f"Feature: {all_names[i]} and coefficient is {standard_linear_model.coef_[i]}")

print(f"Sum of the absolute values of the coefficients: {np.sum(np.abs(standard_linear_model.coef_))}")

# MSE's
print(f"Training MSE: {np.sum((y_train_a - standard_linear_model.predict(Z_train)) ** 2) / len(y_train_a)}")
print(f"Test MSE: {np.sum((y_test_a - standard_linear_model.predict(Z_test)) ** 2) / len(y_test_a)}")

print()


# =============================================================
#  STEP 6: Ridge (L2) and Lasso (L1) (18 points)
#
#  TODO: Fit Ridge for each alpha in ridge_alphas and Lasso
#        (with max_iter=100000) for each alpha in lasso_alphas.
#        Print what the assignment PDF asks for each alpha, then
#        the coefficients of Ridge at 1000 and Lasso at 20.
#        Note: alpha is scikit-learn's name for lambda.
# =============================================================
print("-" * 55)
print("STEP 6: Ridge and Lasso")
print("-" * 55)

ridge_alphas = [1, 10, 100, 1000]
lasso_alphas = [0.1, 1, 5, 20]

# YOUR CODE HERE
print("Ridge: ")
for alpha in ridge_alphas:
    model = Ridge(alpha=alpha).fit(Z_train, y_train_a)
    print(f"Sum of the absolute values of the coefficients: {np.sum(np.abs(model.coef_))}")
    print(f"Number of coefficients that are exactly 0: {np.sum(model.coef_ == 0)}")
    features_are_0 = [all_names[i] for i in range(len(all_names)) if model.coef_[i] == 0]
    print(f"Names of features that are exactly 0: {features_are_0}")
    print(f"Training MSE: {np.sum((y_train_a - model.predict(Z_train)) ** 2) / len(y_train_a)}")
    print(f"Test MSE: {np.sum((y_test_a - model.predict(Z_test)) ** 2) / len(y_test_a)}")
    print()

print("Lasso: ")
for alpha in lasso_alphas:
    model = Lasso(alpha=alpha, max_iter=100000).fit(Z_train, y_train_a)
    print(f"Sum of the absolute values of the coefficients: {np.sum(np.abs(model.coef_))}")
    print(f"Number of coefficients that are exactly 0: {np.sum(model.coef_ == 0)}")
    features_are_0 = [all_names[i] for i in range(len(all_names)) if model.coef_[i] == 0]
    print(f"Names of features that are exactly 0: {features_are_0}")
    print(f"Training MSE: {np.sum((y_train_a - model.predict(Z_train)) ** 2) / len(y_train_a)}")
    print(f"Test MSE: {np.sum((y_test_a - model.predict(Z_test)) ** 2) / len(y_test_a)}")
    print()

print("Ridge Coefficients: ")
ridge_model = Ridge(alpha=1000).fit(Z_train, y_train_a)
for i in range(len(all_names)):
    print(f"Feature: {all_names[i]} and coefficient is {ridge_model.coef_[i]}")
print()
print("Lasso Coefficients: ")
lasso_model = Lasso(alpha=20).fit(Z_train, y_train_a)
for i in range(len(all_names)):
    print(f"Feature: {all_names[i]} and coefficient is {lasso_model.coef_[i]}")
print()


# =============================================================
#  STEP 7: Regularization questions (14 points)
# =============================================================
print("-" * 55)
print("STEP 7: Regularization questions")
print("-" * 55)

# (No code needed -- answer in your PDF report)

print()


# =============================================================
#  PART C: LOGISTIC REGRESSION
#  Dataset: Breast Cancer Wisconsin
#  Goal: Predict whether a tumor is malignant (0) or
#        benign (1) from 3 cell measurements.
# =============================================================

# -------------------------------------------------------------
#  DATA LOADING (do not modify)
# -------------------------------------------------------------
cancer = load_breast_cancer()
X_cls = cancer.data[:, [0, 1, 4]]  # mean_radius, mean_texture, mean_smoothness
y_cls = cancer.target               # 0 = malignant, 1 = benign
feature_names_cls = ["mean_radius", "mean_texture", "mean_smoothness"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.25, random_state=7
)

print("=" * 55)
print("PART C: LOGISTIC REGRESSION -- Breast Cancer Dataset")
print("=" * 55)
print(f"Training samples: {len(X_train_c)}")
print(f"Test samples:     {len(X_test_c)}")
print(f"Features:         {feature_names_cls}")
print(f"Classes:          0 (malignant) and 1 (benign)")
print()


# =============================================================
#  STEP 8: Fit two Logistic Regression models (8 points)
#
#  TODO: Create and fit model_default (max_iter=10000) and
#        model_strong (max_iter=10000, C=0.01). For each, print
#        the intercept, the coefficients, and the sum of their
#        absolute values.
# =============================================================
print("-" * 55)
print("STEP 8: Fit the models")
print("-" * 55)

# YOUR CODE HERE

print()


# =============================================================
#  STEP 9: Predict, count mistakes, look at probabilities (12 points)
#
#  TODO: Predict with model_default and print the first 15
#        predicted and true labels. Count the test mistakes for
#        both models, and the training mistakes for model_default.
#        Use predict_proba to print the probability table and
#        P(benign) for each misclassified test example.
# =============================================================
print("-" * 55)
print("STEP 9: Predict and evaluate")
print("-" * 55)

# YOUR CODE HERE

print()


# =============================================================
#  STEP 10: Observations (10 points)
# =============================================================
print("-" * 55)
print("STEP 10: Observations")
print("-" * 55)

