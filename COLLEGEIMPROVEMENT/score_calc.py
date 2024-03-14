import pandas as pd

df = pd.read_csv(r'C:\Users\jakem\CollegeImprovement-1\COLLEGEIMPROVEMENT\data\RegressionOutput.csv')

#Calculating Score for each university 
df['Score'] =  df['Pred_MD_EARN_WNE_P10'] / df['Pred_GRAD_DEBT_MDN_SUPP'] * (df['Pred_COUNT_WNE_P10'] / (df['UGDS'] * 2.5))

df