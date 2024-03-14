import pandas as pd

errors = pd.DataFrame({
    'Models': ['Random Forest', 'Linear Regression', 'SVM', 'XGBoost', 'Polynomial Regression'],
    'Errors': [random_error, regression_error, svm_error, xgb_error, poly_error]
})

errors2 = pd.DataFrame({
    'Models': ['Random Forest', 'Linear Regression', 'SVM', 'XGBoost'],
    'Errors': [random_error, regression_error, svm_error, xgb_error]
})


print(f'The Random Forest Error is: {random_error}')
print(f'The Support Vector Machine Error is: {svm_error}')
print(f'The Linear Regression Error is: {regression_error}')
print(f'The XGBoost Error is: {xgb_error}')
print(f'The Polynomial Regression Error is: {poly_error}')