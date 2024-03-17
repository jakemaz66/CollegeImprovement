from COLLEGEIMPROVEMENT.models import random_forest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Obtaining coefficient values form regression model
coefficients = pd.DataFrame({'Feature': random_forest.X.columns, 'Coefficient of Importance': random_forest.rfr.feature_importances_})

#FIGURE1
coefficients_filtered = coefficients[coefficients['Feature'] != 'UGDS']

plt.figure(figsize=(14, 9))
sns.barplot(x='Coefficient of Importance', y='Feature', data=coefficients_filtered, palette='viridis')
plt.title('Feature Importances in the Random Forest Model')
plt.xlabel('Coefficient Value', fontsize=16)
plt.ylabel('Feature', fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


