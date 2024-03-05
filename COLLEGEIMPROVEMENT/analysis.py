from COLLEGEIMPROVEMENT.models import regression4
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

reg = regression4.reg
errors = regression4.errors


coefficients = pd.DataFrame({'Feature': regression4.X.columns, 'Coefficient': reg.coef_})

#plt.figure(figsize=(14, 9))

#sns.barplot(x='Feature', y='Coefficient', data=coefficients, palette='viridis')

#plt.title('Feature Coefficients in the Linear Regression Model')
#plt.xlabel('Coefficient Value')
#plt.ylabel('Feature')

#plt.axhline(y=0, color='black', linestyle='--', linewidth=2)

#plt.show()

plt.figure(figsize=(14, 9))

sns.barplot(x='Models', y='Errors', data=errors, palette='viridis')

plt.title('Regression Models and Total Error on Full Dataset')
plt.xlabel('Model Type')
plt.ylabel('Error')

plt.show()
