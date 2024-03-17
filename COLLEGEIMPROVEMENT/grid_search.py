from sklearn.model_selection import GridSearchCV
from COLLEGEIMPROVEMENT.models import random_forest
from COLLEGEIMPROVEMENT.models import XGBoost
from COLLEGEIMPROVEMENT.models import linear_regression
import pandas as pd

#Setting option to see all results
pd.set_option('display.max_colwidth', None)

#Importing models
rfr = random_forest.rfr
xgb = XGBoost.xgb
reg = linear_regression.reg

X_train = random_forest.scaled_x
y_train = random_forest.y_train

#Param grid for random forest
param_grid_1 = {
    'n_estimators': [50, 100, 150],            
    'max_depth': [None, 10, 20],                
    'min_samples_split': [2, 5, 10]
}
#{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 150}

#Param grid for XGBoost
param_grid_2 = {
    'n_estimators': [50, 100, 200],  
    'learning_rate': [0.01, 0.1, 0.2],  
    'max_depth': [3, 5, 7]  
}
#{'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200}

#Param grid for linear regression
param_grid_3 = {
    'fit_intercept': [True, False],    
    'positive': [False, True],       
}

#Building Grid Search objects with repspective param_grids
clf = GridSearchCV(rfr, param_grid_1)
clf.fit(X_train, y_train)

clf2 = GridSearchCV(xgb, param_grid_2)
clf2.fit(X_train, y_train)

clf3 = GridSearchCV(reg, param_grid_3)
clf3.fit(X_train, y_train)

best_params_rfr = clf.best_params_
best_params_xgb = clf2.best_params_
best_params_reg = clf3.best_params_

#Printing the best hyperparameters in a table
params_table = pd.DataFrame({'Model Type': ['Random Forest', 'XGBoost', 'Regression'],
        'Best Parameters for Model': [best_params_rfr, best_params_xgb, best_params_reg]})
print(params_table)