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
from xgboost import XGBRegressor

from sklearn.impute import IterativeImputer

#Reading in my dataframes
df1 = pd.read_csv(r'C:\Users\jakem\CollegeImprovement-1\COLLEGEIMPROVEMENT\data\CollegeImprovementFinalFile.csv')
df2 = data_reader.data_reader_study()

#Replacing Privacy Supressed Values with NaNs
df1.replace('PrivacySuppressed', np.nan, inplace=True)

#Filling in missing values
df1['TUITIONFEE_IN'] = df1['TUITIONFEE_IN'].fillna(df1.groupby('INSTNM')['TUITIONFEE_IN'].transform('median'))
df1['ADM_RATE'] = df1['ADM_RATE'].fillna(df1.groupby('INSTNM')['ADM_RATE'].transform('median'))
df1['ADMCON7'] = df1['ADMCON7'].fillna(df1.groupby('INSTNM')['ADMCON7'].transform('median'))
df1['AVGFACSAL'] = df1['AVGFACSAL'].fillna(df1.groupby('INSTNM')['AVGFACSAL'].transform('median'))
df1['PFTFAC'] = df1['PFTFAC'].fillna(df1.groupby('INSTNM')['PFTFAC'].transform('median'))
df1['INEXPFTE'] = df1['INEXPFTE'].fillna(df1.groupby('INSTNM')['INEXPFTE'].transform('median'))
df1['STUFACR'] = df1['STUFACR'].fillna(df1.groupby('INSTNM')['STUFACR'].transform('median'))

df1['PRGMOFR'] = df1['PRGMOFR'].fillna(df1.groupby('INSTNM')['PRGMOFR'].transform('median'))

df1['GRAD_DEBT_MDN_SUPP'] = df1['GRAD_DEBT_MDN_SUPP'].replace('nan', np.nan)
df1['GRAD_DEBT_MDN_SUPP'] = df1['GRAD_DEBT_MDN_SUPP'].astype(float)
df1['GRAD_DEBT_MDN_SUPP'] = df1['GRAD_DEBT_MDN_SUPP'].fillna(df1.groupby('INSTNM')['GRAD_DEBT_MDN_SUPP'].transform('median'))

df_model = df1[['TUITIONFEE_IN', 'ADM_RATE', 'ADMCON7','AVGFACSAL', 'PFTFAC', 'INEXPFTE', 'STUFACR',
           'PRGMOFR', 'GRAD_DEBT_MDN_SUPP']]

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

y = df_imputed['GRAD_DEBT_MDN_SUPP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scaling the data
scaler = StandardScaler()
scaled_x = scaler.fit_transform(X_train)
scaled_x_test = scaler.fit_transform(X_test)

#Building Regression models
rfr = RandomForestRegressor()
svm1 = svm.SVR()
reg = LinearRegression()
xgb = XGBRegressor()

rfr.fit(scaled_x, y_train)
svm1.fit(scaled_x, y_train)
reg.fit(scaled_x, y_train)
xgb.fit(scaled_x, y_train)

#Calculating Errors
random_error = mean_squared_error(y_test, rfr.predict(scaled_x_test))
svm_error = mean_squared_error(y_test, svm1.predict(scaled_x_test))
regression_error = mean_squared_error(y_test, reg.predict(scaled_x_test))
xgb_error = mean_squared_error(y_test, xgb.predict(scaled_x_test))

print(f'The Random Forest Error is: {random_error}')
print(f'The Support Vector Machine Error is: {svm_error}')
print(f'The Linear Regression Error is: {regression_error}')
print(f'TheXGBoost Error is: {xgb_error}')

#Predicting Duquesne
Duquesne = df1[df1['INSTNM'] == 'Duquesne University']
X = Duquesne[['ADM_RATE', 'TUITIONFEE_IN', 'ADMCON7',
         'AVGFACSAL', 'INEXPFTE',
         'STUFACR', 'PRGMOFR']]

y = Duquesne['GRAD_DEBT_MDN_SUPP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler.transform(X_train)

print(f'Duquesne Estimated Median: {rfr.predict(X_train)}')