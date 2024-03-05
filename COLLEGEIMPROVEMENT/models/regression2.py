import pandas as pd
import numpy as np
from COLLEGEIMPROVEMENT import data_reader
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer

#Reading in KNN Imputed Data, but testing Debt as target variable
df1 = pd.read_csv(r'C:\Users\jakem\CollegeImprovement-1\COLLEGEIMPROVEMENT\data\CollegeImprovementFinalFileImputed.csv')

#Splitting the data
X = df1[['ADM_RATE', 'TUITIONFEE_IN', 'IRPS_NRA', 'ADMCON7',
         'AVGFACSAL', 'PFTFAC', 'UGDS', 'TRANS_4', 'INEXPFTE',
         'OPENADMP', 'STUFACR', 'PRGMOFR']]
y = df1['GRAD_DEBT_MDN_SUPP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scaling the data
scaler = StandardScaler()
scaled_x = scaler.fit_transform(X_train)
scaled_x_test = scaler.fit_transform(X_test)

#Building Regression models
rfr = RandomForestRegressor()
svm1 = svm.SVR()
reg = LinearRegression()

rfr.fit(scaled_x, y_train)
svm1.fit(scaled_x, y_train)
reg.fit(scaled_x, y_train)

#Calculating Errors
random_error = mean_squared_error(y_test, rfr.predict(scaled_x_test))
svm_error = mean_squared_error(y_test, svm1.predict(scaled_x_test))
regression_error = mean_squared_error(y_test, reg.predict(scaled_x_test))

print(f'The Random Forest Error is: {random_error}')
print(f'The Support Vector Machine Error is: {svm_error}')
print(f'The Linear Regression Error is: {regression_error}')

#Predicting Duquesne
Duquesne = df1[df1['INSTNM'] == 'Harvard University']
X = Duquesne[['ADM_RATE', 'TUITIONFEE_IN', 'IRPS_NRA', 'ADMCON7',
         'AVGFACSAL', 'PFTFAC', 'UGDS', 'TRANS_4', 'INEXPFTE',
         'OPENADMP', 'STUFACR', 'PRGMOFR']]
y = Duquesne['GRAD_DEBT_MDN_SUPP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler.transform(X_train)

print(f'Duquesne Estimated Median: {rfr.predict(X_train)}')