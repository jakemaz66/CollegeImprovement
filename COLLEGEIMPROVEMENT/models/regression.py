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
df1['UGDS'] = df1['UGDS'].fillna(df1.groupby('INSTNM')['UGDS'].transform('median'))
df1['BOOKSUPPLY'] = df1['BOOKSUPPLY'].fillna(df1.groupby('INSTNM')['BOOKSUPPLY'].transform('median'))
df1['ROOMBOARD_OFF'] = df1['ROOMBOARD_OFF'].fillna(df1.groupby('INSTNM')['ROOMBOARD_OFF'].transform('median'))
df1['OTHEREXPENSE_OFF'] = df1['OTHEREXPENSE_OFF'].fillna(df1.groupby('INSTNM')['OTHEREXPENSE_OFF'].transform('median'))
df1['OTHEREXPENSE_FAM'] = df1['OTHEREXPENSE_FAM'].fillna(df1.groupby('INSTNM')['OTHEREXPENSE_FAM'].transform('median'))
df1['STUFACR'] = df1['STUFACR'].fillna(df1.groupby('INSTNM')['STUFACR'].transform('median'))
df1['TRANS_4'] = df1['TRANS_4'].fillna(df1.groupby('INSTNM')['TRANS_4'].transform('median'))

df1['IRPS_NRA'].fillna(df1['IRPS_NRA'].mode(), inplace=True)
df1['OPENADMP'].fillna(df1['OPENADMP'].mode(), inplace=True)

df1['PRGMOFR'] = df1['PRGMOFR'].fillna(df1.groupby('INSTNM')['PRGMOFR'].transform('median'))
df1['GRAD_DEBT_MDN_SUPP'] = df1['GRAD_DEBT_MDN_SUPP'].replace('nan', np.nan)
df1['GRAD_DEBT_MDN_SUPP'] = df1['GRAD_DEBT_MDN_SUPP'].astype(float)
df1['GRAD_DEBT_MDN_SUPP'] = df1['GRAD_DEBT_MDN_SUPP'].fillna(df1.groupby('INSTNM')['GRAD_DEBT_MDN_SUPP'].transform('median'))
df1['MD_EARN_WNE_P10'] = df1['MD_EARN_WNE_P10'].replace('nan', np.nan)
df1['MD_EARN_WNE_P10'] = df1['MD_EARN_WNE_P10'].astype(float)
df1['MD_EARN_WNE_P10'] = df1['MD_EARN_WNE_P10'].fillna(df1.groupby('INSTNM')['MD_EARN_WNE_P10'].transform('median'))
df1['COUNT_NWNE_4YR'] = df1['COUNT_NWNE_4YR'].replace('nan', np.nan)
df1['COUNT_NWNE_4YR'] = df1['COUNT_NWNE_4YR'].astype(float)
df1['COUNT_NWNE_4YR'] = df1['COUNT_NWNE_4YR'].fillna(df1.groupby('INSTNM')['COUNT_NWNE_4YR'].transform('median'))

df1.dropna(subset=['COUNT_NWNE_4YR'], inplace=True)


#Filling in all other NaNs
knnimpute = KNNImputer(n_neighbors=2)
numeric_cols = df1.select_dtypes(include='number').columns

df_numeric = df1[numeric_cols]

df_numeric_imputed = pd.DataFrame(knnimpute.fit_transform(df_numeric), columns=numeric_cols)

df1 = pd.concat([df_numeric_imputed, df1.drop(columns=numeric_cols)], axis=1)

#Feature Engineering
df1['EXPENSES'] = (df1['BOOKSUPPLY'] + df1['ROOMBOARD_OFF'] + df1['OTHEREXPENSE_OFF'] +   
                  df1['OTHEREXPENSE_FAM'])

df1.to_csv(r'C:\Users\jakem\CollegeImprovement-1\COLLEGEIMPROVEMENT\data\CollegeImprovementFinalFileImputed2.csv')

#Splitting the data
X = df1[['ADM_RATE', 'TUITIONFEE_IN', 'IRPS_NRA', 'ADMCON7',
         'AVGFACSAL', 'PFTFAC', 'UGDS', 'TRANS_4', 'INEXPFTE',
         'OPENADMP', 'STUFACR', 'PRGMOFR']]
y = df1['MD_EARN_WNE_P10']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scaling the data
scaler = StandardScaler()
scaled_x = scaler.fit_transform(X_train)

print('HEY')

#Building Regression models
rfr = RandomForestRegressor()
#svm1 = svm.SVR()
reg = LinearRegression()

rfr.fit(scaled_x, y_train)
#svm1.fit(scaled_x, y_train)
reg.fit(scaled_x, y_train)

#Calculating Errors
random_error = mean_squared_error(y_test, rfr.predict(X_test))
#svm_error = mean_squared_error(y_test, svm1.predict(X_test))
regression_error = mean_squared_error(y_test, reg.predict(X_test))

print(f'The Random Forest Error is: {random_error}')
#print(f'The Support Vector Machine Error is: {svm_error}')
print(f'The Linear Regression Error is: {regression_error}')

#Predicting Duquesne
Duquesne = df1[df1['INSTNM'] == 'Duquesne University']
X = Duquesne[['ADM_RATE', 'TUITIONFEE_IN', 'IRPS_NRA', 'ADMCON7',
         'AVGFACSAL', 'PFTFAC', 'UGDS', 'TRANS_4', 'INEXPFTE',
         'OPENADMP', 'STUFACR', 'PRGMOFR']]
y = Duquesne['MD_EARN_WNE_P10']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler.fit_transform(X_train)

print(f'Duquesne Estimated Median: {rfr.predict(X_train)}')










