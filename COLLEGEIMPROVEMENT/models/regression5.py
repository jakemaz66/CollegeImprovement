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

df1['GRAD_DEBT_MDN_SUPP'] = df1['GRAD_DEBT_MDN_SUPP'].replace('nan', np.nan)
df1['GRAD_DEBT_MDN_SUPP'] = df1['GRAD_DEBT_MDN_SUPP'].astype(float)
df1['GRAD_DEBT_MDN_SUPP'] = df1['GRAD_DEBT_MDN_SUPP'].fillna(df1.groupby('INSTNM')['GRAD_DEBT_MDN_SUPP'].transform('median'))
df1.dropna(subset=['GRAD_DEBT_MDN_SUPP'], inplace=True)

#Splitting the data
X = df1[['ADM_RATE', 'TUITIONFEE_IN', 'ADMCON7',
         'AVGFACSAL', 'INEXPFTE',
        'STUFACR', 'PRGMOFR']]

y = df1['GRAD_DEBT_MDN_SUPP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scaling the data
scaler = StandardScaler()
scaled_x = scaler.fit_transform(X_train)
scaled_x_test = scaler.fit_transform(X_test)

#Building Regression models
xgb = XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=100)

xgb.fit(scaled_x, y_train)

xgb_error = mean_squared_error(y_test, xgb.predict(scaled_x_test))


errors = pd.DataFrame({
    'Models': [ 'XGBoost'],
    'Errors': [xgb_error]
})


print(f'The XGBoost Error is: {xgb_error}')


#Predicting Duquesne
Duquesne = df1[df1['INSTNM'] == 'Duquesne University']
X = Duquesne[['ADM_RATE', 'TUITIONFEE_IN', 'ADMCON7',
         'AVGFACSAL', 'INEXPFTE',
         'STUFACR', 'PRGMOFR']]

y = Duquesne['GRAD_DEBT_MDN_SUPP']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)
scaler.transform(X_train2)

print(f'Duquesne Estimated Median: {xgb.predict(X_train2)}')