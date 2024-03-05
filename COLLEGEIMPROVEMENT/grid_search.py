from sklearn.model_selection import GridSearchCV
from COLLEGEIMPROVEMENT.models import regression4

rfr = regression4.rfr
svm = regression4.svm1
reg = regression4.reg

X_train = regression4.scaled_x
y_train = regression4.y_train

#Param grid for random forest
param_grid_1 = {
    'n_estimators': [50, 100, 150],            
    'max_depth': [None, 10, 20],                
    'min_samples_split': [2, 5, 10]
}

param_grid_3 = {
    'fit_intercept': [True, False],    
    'positive': [False, True],       
}

#Building Grid Search objects with repspective param_grids
clf = GridSearchCV(rfr, param_grid_1)
clf.fit(X_train, y_train)

clf3 = GridSearchCV(reg, param_grid_3)
clf3.fit(X_train, y_train)

#Getting the Parameters
print(f'The Best Params for Random Forest: {clf.best_params_}')
print(f'The Best Params for Regression: {clf3.best_params_}')