#Importing libraries
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

# Reading in my dataframes
df1 = pd.read_csv(r'C:\Users\jakem\CollegeImprovement-1\COLLEGEIMPROVEMENT\data\CollegeImprovementFinalFile.csv')

# Replacing Privacy Suppressed Values with NaNs
df1.replace('PrivacySuppressed', np.nan, inplace=True)

# Dropping columns with missing target variables
df1.dropna(subset=['PCT75_EARN_WNE_P10', 'COUNT_WNE_P10', 'MD_EARN_WNE_P10', 'GRAD_DEBT_MDN_SUPP'], inplace=True)

# Filling in missing values
#columns_to_impute = ['TUITIONFEE_IN', 'ADM_RATE', 'ADMCON7', 'AVGFACSAL', 'PFTFAC', 'INEXPFTE', 'STUFACR', 'PRGMOFR', 'UGDS']
#for col in columns_to_impute:
#    df1[col] = df1[col].fillna(df1.groupby('INSTNM')[col].transform('mean'))

df_model = df1[['TUITIONFEE_IN', 'ADM_RATE', 'ADMCON7', 'AVGFACSAL', 'PFTFAC', 'INEXPFTE', 'STUFACR', 'UGDS',
                'PRGMOFR', 'PCT75_EARN_WNE_P10', 'COUNT_WNE_P10', 'MD_EARN_WNE_P10', 'GRAD_DEBT_MDN_SUPP']]

df_final = pd.DataFrame(columns=['University', 'Predicted Salary', 'Predicted Debt', 'Predicted Job', 'Enrollment'])
df_final['University'] = df1['INSTNM']
df_final['Enrollment'] = df1['UGDS']

# Filling in the rest of NaNs with the iterative imputer
columns_to_impute = ['ADM_RATE', 'TUITIONFEE_IN', 'ADMCON7', 'AVGFACSAL', 'INEXPFTE', 'STUFACR', 'PRGMOFR', 'UGDS']
iterative_imputer = IterativeImputer(max_iter=25, random_state=0)  
df_model_imputed = pd.DataFrame(iterative_imputer.fit_transform(df_model), columns=df_model.columns)

#Splitting the data
X = df_model_imputed[['ADM_RATE', 'TUITIONFEE_IN', 'ADMCON7',
         'AVGFACSAL', 'INEXPFTE',
        'STUFACR', 'PRGMOFR', 'UGDS']]

y = df_model_imputed['COUNT_WNE_P10']
y2 = df_model_imputed['MD_EARN_WNE_P10']
y3 = df_model_imputed['GRAD_DEBT_MDN_SUPP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size=0.2, random_state=42)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y3, test_size=0.2, random_state=42)


#Scaling the data
scaler = StandardScaler()
scaled_x = scaler.fit_transform(X_train)
scaled_x2 = scaler.transform(X_train2)
scaled_x3 = scaler.transform(X_train3)
scaled_x_test = scaler.transform(X_test)
scaled_x_test2 = scaler.transform(X_test2)
scaled_x_test3 = scaler.transform(X_test3)

xgb = XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=100)

xgb.fit(scaled_x, y_train)

predictions = xgb.predict(scaler.transform(df_model_imputed[['ADM_RATE', 'TUITIONFEE_IN', 'ADMCON7', 'AVGFACSAL', 'INEXPFTE', 'STUFACR', 'PRGMOFR', 'UGDS']]))
predictions_series = pd.Series(predictions, index=df_final.index)
df_final['Predicted Job'] = predictions_series

error_job = mean_squared_error(y_test, xgb.predict(scaled_x_test))
print(f'Error for Job: {error_job}')

xgb.fit(scaled_x2, y_train2)

predictions = xgb.predict(scaler.transform(df_model_imputed[['ADM_RATE', 'TUITIONFEE_IN', 'ADMCON7', 'AVGFACSAL', 'INEXPFTE', 'STUFACR', 'PRGMOFR', 'UGDS']]))
predictions_series = pd.Series(predictions, index=df_final.index)
df_final['Predicted Salary'] = predictions_series

error_salary = mean_squared_error(y_test2, xgb.predict(scaled_x_test2))
print(f'Error for Salary: {error_salary}')

xgb.fit(scaled_x3, y_train3)

predictions = xgb.predict(scaler.transform(df_model_imputed[['ADM_RATE', 'TUITIONFEE_IN', 'ADMCON7', 'AVGFACSAL', 'INEXPFTE', 'STUFACR', 'PRGMOFR', 'UGDS']]))
predictions_series = pd.Series(predictions, index=df_final.index)
df_final['Predicted Debt'] = predictions_series

error_debt = mean_squared_error(y_test3, xgb.predict(scaled_x_test3))
print(f'Error for Debt: {error_debt}')

df_final.to_csv(r'C:\Users\jakem\CollegeImprovement-1\COLLEGEIMPROVEMENT\data\RegressionOutput.csv')

