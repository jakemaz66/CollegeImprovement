from sklearn.model_selection import GridSearchCV
from COLLEGEIMPROVEMENT.models import regression

rfr = regression.rfr
svm = regression.svm
reg = regression.reg

X_train = regression.X_train
y_train = regression.y_train

#Param grid for random forest
param_grid_1 = {
    'n_estimators': [50, 100, 150],            
    'max_depth': [None, 10, 20],                
    'min_samples_split': [2, 5, 10]
}

param_grid_2 = {
    'C': [1, 10, 100],                  
    'kernel': ['linear', 'rbf', 'poly'], 
    'gamma': ['scale', 'auto', 0.1] 
}

param_grid_3 = {
    'fit_intercept': [True, False],  
    'normalize': [True, False],      
    'positive': [False, True],       
}

#Building Grid Search objects with repspective param_grids
clf = GridSearchCV(rfr, param_grid_1)
clf.fit(X_train, y_train)

clf2 = GridSearchCV(svm, param_grid_2)
clf2.fit(X_train, y_train)

clf3 = GridSearchCV(reg, param_grid_3)
clf3.fit(X_train, y_train)

#Getting the Parameters
print(f'The Best Params for Random Forest: {clf.best_params_}')
print(f'The Best Params for SVM: {clf2.best_params_}')
print(f'The Best Params for Regression: {clf3.best_params_}')