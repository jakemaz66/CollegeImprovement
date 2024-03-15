from COLLEGEIMPROVEMENT.models import random_forest
from COLLEGEIMPROVEMENT.models import XGBoost
from COLLEGEIMPROVEMENT.models import linear_regression
from COLLEGEIMPROVEMENT.models import support_vector_machine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Importing Models and Errors


#Obtaining coefficient values form regression model
coefficients = pd.DataFrame({'Feature': random_forest.X.columns, 'Coefficient of Importance': random_forest.rfr.feature_importances_})


#FIGURE1
plt.figure(figsize=(14, 9))

sns.barplot(x='Feature', y='Coefficient', data=coefficients, palette='viridis')

plt.title('Feature Importances in the Randome Forest Model')
plt.xlabel('Coefficient Value', fontsize=16)
plt.ylabel('Feature', fontsize=16)

plt.axhline(y=0, color='black', linestyle='--', linewidth=2)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


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
