from COLLEGEIMPROVEMENT.models import random_forest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Importing Models and Errors
reg = random_forest.reg
errors = random_forest.errors
errors2 = random_forest.errors2
df1 = random_forest.df1
xgb_error = random_forest.xgb_error
xgb_default_error = random_forest.xgb_default_error

#Obtaining coefficient values form regression model
coefficients = pd.DataFrame({'Feature': random_forest.X.columns, 'Coefficient': reg.coef_})

#Obtaining predcitions from models
recent_pred = random_forest.xgb.predict(random_forest.X_train2)
recent_pred = np.median(recent_pred)
actual_pred = df1[(df1['INSTNM'] == 'University of Pittsburgh-Pittsburgh Campus') & (df1['Year'] == '2019_20')]
actual_pred = actual_pred['GRAD_DEBT_MDN_SUPP'].max()


#FIGURE1
#plt.figure(figsize=(14, 9))

#sns.barplot(x='Feature', y='Coefficient', data=coefficients, palette='viridis')

#plt.title('Feature Coefficients in the Linear Regression Model')
#plt.xlabel('Coefficient Value', fontsize=16)
#plt.ylabel('Feature', fontsize=16)

#plt.axhline(y=0, color='black', linestyle='--', linewidth=2)

#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#plt.show()


#FIGURE2
plt.figure(figsize=(14, 9))

sns.barplot(x='Models', y='Errors', data=errors2, palette='viridis')

plt.title('Regression Models and Total Error on Full Dataset')
plt.xlabel('Model Type')
plt.ylabel('Error')
plt.show()


#FIGURE3
#plt.figure(figsize=(14, 9))

#labels = ['Predicted Debt', 'Actual Debt']
#values = [recent_pred, actual_pred]

#graph = plt.bar(labels, values, color=['blue', 'green'])
#plt.title('Comparison of Predicted and Actual Debt for University of Pittsburgh')
#plt.ylabel('Debt')

#for bar, value in zip(graph, values):
#    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.5, str(round(value, 2)),
#             ha='center', va='bottom', color='black', fontsize=12)
#plt.show()


#FIGURE4
#grid_results = pd.DataFrame({'Models': ['XGBoost Best Params', 'Default XGBoost'],
#                            'Errors': [xgb_error, xgb_default_error]})

#plt.figure(figsize=(14, 9))
#plt.bar(grid_results['Models'], grid_results['Errors'], color=['blue', 'green'])
#plt.title('Comparison of XGBoost with and without Grid Search')
#plt.ylabel('Error')
#plt.show()
