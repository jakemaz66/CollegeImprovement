import pandas as pd
import numpy as np
from COLLEGEIMPROVEMENT import data_reader
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Reading in my dataframes
df1 = data_reader.data_reader()
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
df1['ROOMBOARD_ON'] = df1['ROOMBOARD_ON'].fillna(df1.groupby('INSTNM')['ROOMBOARD_ON'].transform('median'))
df1['OTHEREXPENSE_ON'] = df1['OTHEREXPENSE_ON'].fillna(df1.groupby('INSTNM')['OTHEREXPENSE_ON'].transform('median'))
df1['ROOMBOARD_OFF'] = df1['ROOMBOARD_OFF'].fillna(df1.groupby('INSTNM')['ROOMBOARD_OFF'].transform('median'))
df1['OTHEREXPENSE_OFF'] = df1['OTHEREXPENSE_OFF'].fillna(df1.groupby('INSTNM')['OTHEREXPENSE_OFF'].transform('median'))
df1['OTHEREXPENSE_FAM'] = df1['OTHEREXPENSE_FAM'].fillna(df1.groupby('INSTNM')['OTHEREXPENSE_FAM'].transform('median'))
df1['STUFACR'] = df1['STUFACR'].fillna(df1.groupby('INSTNM')['STUFACR'].transform('median'))

df1['IRPS_NRA'].fillna(df1['IRPS_NRA'].mode(), inplace=True)
df1['OPENADMP'].fillna(df1['OPENADMP'].mode(), inplace=True)

df1['PRGMOFR'] = df1['PRGMOFR'].fillna(df1.groupby('INSTNM')['PRGMOFR'].transform('median'))
df1['GRAD_DEBT_MDN_SUPP'] = df1['GRAD_DEBT_MDN_SUPP'].replace('nan', np.nan)
df1['GRAD_DEBT_MDN_SUPP'] = df1['GRAD_DEBT_MDN_SUPP'].astype(float)
df1['GRAD_DEBT_MDN_SUPP'] = df1['GRAD_DEBT_MDN_SUPP'].fillna(df1['GRAD_DEBT_MDN_SUPP'].median())
df1['MD_EARN_WNE_P10'] = df1['MD_EARN_WNE_P10'].replace('nan', np.nan)
df1['MD_EARN_WNE_P10'] = df1['MD_EARN_WNE_P10'].astype(float)
df1['MD_EARN_WNE_P10'] = df1['MD_EARN_WNE_P10'].fillna(df1['MD_EARN_WNE_P10'].median())
df1['COUNT_NWNE_P10'] = df1['COUNT_NWNE_P10'].replace('nan', np.nan)
df1['COUNT_NWNE_P10'] = df1['COUNT_NWNE_P10'].astype(float)
df1['COUNT_NWNE_P10'] = df1['COUNT_NWNE_P10'].fillna(df1['COUNT_NWNE_P10'].median())


#Dropping all other rows that have NaNs
df1.dropna(inplace=True)

#Feature Engineering
df1['EXPENSES'] = (df1['BOOKSUPPLY'] + df1['ROOMBOARD_ON'] + df1['OTHEREXPENSE_ON'] + df1['ROOMBOARD_OFF'] + df1['OTHEREXPENSE_OFF'] + 
                   
                  df1['OTHEREXPENSE_FAM'])

#Splitting the data
X = df1.iloc[:, :-4]
y = df1.iloc[:, -3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scaling the data
scaler = StandardScaler()

scaled_x = scaler.fit_transform(X_train)






