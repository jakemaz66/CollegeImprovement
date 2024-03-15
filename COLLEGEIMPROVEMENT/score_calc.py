import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\jakem\CollegeImprovement-1\COLLEGEIMPROVEMENT\data\RegressionOutputHigherDegree.csv')

#Imputing Enrollment
df['Undergraduate Enrollment'] = df['Undergraduate Enrollment'].fillna(df.groupby('University')['Undergraduate Enrollment'].transform('median'))
df['Undergraduate Enrollment'].fillna(df['Undergraduate Enrollment'].mean(), inplace=True)

#Calculating Score for each university 
df['Score'] =  (df['Predicted Salary'] / df['Predicted Debt']) + ((df['Undergraduate Enrollment'] )/df['Predicted Job'] )
#df_Fres = df[df['University'] == 'Philadelphia College of Osteopathic Medicine']
#df_Fres[['University', 'Score', 'Predicted Debt', 'Predicted Salary', 'Predicted Job', 'Undergraduate Enrollment']]

#CALCULATE CORRELATIONS SCORE AND FEATURES
corr = df[['Score', 'Admission Rate', 'Tuition', 'Admission Test Score', 'Faculty Salary',
                                 'Expenditures per Student', 'Student Faculty Ratio', 'Programs Offered',
                                 'Undergraduate Enrollment']].corr()


#Making Plot of Correlations
plt.figure(figsize=(14, 8))
sns.barplot(data=corr.iloc[0, :], palette='viridis')

plt.title('Correlations Between Features and Score')
plt.xlabel('Features', fontsize=16)
plt.ylabel('Score', fontsize=16)
plt.xticks(rotation=45, ha='right')

plt.axhline(y=0, color='black', linestyle='--', linewidth=2)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


#Seeing top 10
df_grouped = df.groupby('University')['Score'].mean().reset_index()
df_top_10 = df_grouped.sort_values(by='Score', ascending=False).head(30)
print(df_top_10)

df


