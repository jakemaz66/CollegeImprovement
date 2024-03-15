import pandas as pd
import numpy as np
from COLLEGEIMPROVEMENT.models import random_forest
from COLLEGEIMPROVEMENT.models import XGBoost
from COLLEGEIMPROVEMENT.models import linear_regression
from COLLEGEIMPROVEMENT.models import support_vector_machine
import matplotlib.pyplot as plt
import seaborn as sns

#Importing all the errors from the models
error_debt_random = random_forest.error_debt
error_salary_random = random_forest.error_salary
error_job_random = random_forest.error_job
avg_error_random = np.average([error_debt_random, error_salary_random, error_job_random])

error_debt_xgb = XGBoost.error_debt
error_salary_xgb = XGBoost.error_salary
error_job_xgb = XGBoost.error_job
avg_error_xgb = np.average([error_debt_xgb, error_salary_xgb, error_job_xgb])

error_debt_reg = linear_regression.error_debt
error_salary_reg = linear_regression.error_salary
error_job_reg = linear_regression.error_job
avg_error_reg= np.average([error_debt_reg, error_salary_reg, error_job_reg])

error_debt_svm = support_vector_machine.error_debt
error_salary_svm = support_vector_machine.error_salary
error_job_svm = support_vector_machine.error_job
avg_error_svm= np.average([error_debt_svm, error_salary_svm, error_job_svm])

#Creating a dataframe for the average errors
errors = pd.DataFrame({
    'Models': ['Random Forest', 'Linear Regression', 'SVM', 'XGBoost'],
    'Errors': [avg_error_random, avg_error_reg, avg_error_svm, avg_error_xgb]
})

#Printing the average errors
print(f'The Random Forest Error is: {avg_error_random}')
print(f'The Support Vector Machine Error is: {avg_error_svm}')
print(f'The Linear Regression Error is: {avg_error_reg}')
print(f'The XGBoost Error is: {avg_error_xgb}')

#Creating plot of errors
plt.figure(figsize=(14, 9))

sns.barplot(x='Models', y='Errors', data=errors, palette='viridis')

plt.title('Regression Models and Average Error on Full Dataset')
plt.xlabel('Model Type')
plt.ylabel('Error')
plt.show()
