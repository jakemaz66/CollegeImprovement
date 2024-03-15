import pandas as pd

df = pd.read_csv(r'C:\Users\jakem\CollegeImprovement-1\COLLEGEIMPROVEMENT\data\RegressionOutput.csv')

#Imputing Enrollment
df['Enrollment'] = df['Enrollment'].fillna(df.groupby('University')['Enrollment'].transform('median'))
df['Enrollment'].fillna(df['Enrollment'].mean(), inplace=True)

#Calculating Score for each university 
df['Score'] =  (df['Predicted Salary'] / df['Predicted Debt']) + ((df['Enrollment'] )/df['Predicted Job'] )

df

