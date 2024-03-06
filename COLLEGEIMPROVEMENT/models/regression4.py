from sklearn.experimental import enable_iterative_imputer 
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
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

from sklearn.impute import IterativeImputer

#Reading in my dataframes
df1 = pd.read_csv(r'C:\Users\jakem\CollegeImprovement-1\COLLEGEIMPROVEMENT\data\CollegeImprovementFinalFile.csv')
df2 = data_reader.data_reader_study()

#Replacing Privacy Supressed Values with NaNs
df1.replace('PrivacySuppressed', np.nan, inplace=True)

#Filling in missing values
columns = ['TUITIONFEE_IN', 'ADM_RATE', 'ADMCON7', 'AVGFACSAL', 'PFTFAC', 'INEXPFTE', 'STUFACR', 'PRGMOFR']

for col in columns:
    df1[col] = df1[col].fillna(df1.groupby('INSTNM')[col].transform('median'))

df1['MD_EARN_WNE_1YR'] = df1['MD_EARN_WNE_1YR'].replace('nan', np.nan)
df1['MD_EARN_WNE_1YR'] = df1['MD_EARN_WNE_1YR'].astype(float)
df1['MD_EARN_WNE_1YR'] = df1['MD_EARN_WNE_1YR'].fillna(df1.groupby('INSTNM')['MD_EARN_WNE_1YR'].transform('median'))

df_model = df1[['TUITIONFEE_IN', 'ADM_RATE', 'ADMCON7','AVGFACSAL', 'PFTFAC', 'INEXPFTE', 'STUFACR',
           'PRGMOFR', 'MD_EARN_WNE_1YR']]

#Filling in rest of NaNs with the iterative imputer
columns_to_impute = ['ADM_RATE', 'TUITIONFEE_IN', 'ADMCON7',
                      'AVGFACSAL', 'INEXPFTE', 'STUFACR', 'PRGMOFR']
numeric_columns = df_model[columns_to_impute]
iterative_imputer = IterativeImputer(max_iter=25, random_state=0)  
numeric_columns_imputed = pd.DataFrame(iterative_imputer.fit_transform(numeric_columns), columns=columns_to_impute)

df_imputed = pd.concat([numeric_columns_imputed, df1.drop(columns=columns_to_impute)], axis=1)

#Dropping rest
df_imputed.dropna(inplace=True)

#Splitting the data
X = df_imputed[['ADM_RATE', 'TUITIONFEE_IN', 'ADMCON7',
         'AVGFACSAL', 'INEXPFTE',
        'STUFACR', 'PRGMOFR']]

y = df_imputed['MD_EARN_WNE_1YR']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scaling the data
scaler = StandardScaler()
scaled_x = scaler.fit_transform(X_train)
scaled_x_test = scaler.fit_transform(X_test)

#Creating Polynomial Regression
degree = 6  
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train) 
X_test_poly = poly.transform(X_test) 

#Building Regression models
rfr = RandomForestRegressor(max_depth= 20, min_samples_split= 5, n_estimators=50)
svm1 = svm.SVR()
reg = LinearRegression(fit_intercept=True, positive=False)
xgb = XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=100)
xgb_default = XGBRegressor()
poly_reg = LinearRegression()

rfr.fit(scaled_x, y_train)
svm1.fit(scaled_x, y_train)
reg.fit(scaled_x, y_train)
xgb.fit(scaled_x, y_train)
xgb_default.fit(scaled_x, y_train)
poly_reg.fit(X_train_poly, y_train)

#Calculating Errors
random_error = mean_squared_error(y_test, rfr.predict(scaled_x_test))
svm_error = mean_squared_error(y_test, svm1.predict(scaled_x_test))
regression_error = mean_squared_error(y_test, reg.predict(scaled_x_test))
xgb_error = mean_squared_error(y_test, xgb.predict(scaled_x_test))
xgb_default_error = mean_squared_error(y_test, xgb_default.predict(scaled_x_test))
poly_error = mean_squared_error(y_test, poly_reg.predict(X_test_poly))

errors = pd.DataFrame({
    'Models': ['Random Forest', 'Linear Regression', 'SVM', 'XGBoost', 'Polynomial Regression'],
    'Errors': [random_error, regression_error, svm_error, xgb_error, poly_error]
})

print(f'The Random Forest Error is: {random_error}')
print(f'The Support Vector Machine Error is: {svm_error}')
print(f'The Linear Regression Error is: {regression_error}')
print(f'The XGBoost Error is: {xgb_error}')
print(f'The Polynomial Regression Error is: {poly_error}')

#Predicting Duquesne
Duquesne = df1[df1['INSTNM'] == 'University of Pittsburgh-Pittsburgh Campus']
X = Duquesne[['ADM_RATE', 'TUITIONFEE_IN', 'ADMCON7',
         'AVGFACSAL', 'INEXPFTE',
         'STUFACR', 'PRGMOFR']]

y = Duquesne['MD_EARN_WNE_1YR']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)
scaler.transform(X_train2)

print(f'Duquesne Estimated Median: {xgb.predict(X_train2)}')