import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\jakem\CollegeImprovement-1\COLLEGEIMPROVEMENT\data\RegressionOutputHigherDegree.csv')

#Imputing Enrollment
df['Undergraduate Enrollment'] = df['Undergraduate Enrollment'].fillna(df.groupby('University')['Undergraduate Enrollment'].transform('median'))
df['Undergraduate Enrollment'].fillna(df['Undergraduate Enrollment'].mean(), inplace=True)

#Calculating Score for each university 
mean1 = (df['Predicted Salary'] / df['Predicted Debt']).mean()
std1 = (df['Predicted Salary'] / df['Predicted Debt']).std()

mean2 = (df['Undergraduate Enrollment'] /df['Predicted Job']).mean()
std2 = (df['Undergraduate Enrollment'] /df['Predicted Job']).std() * 2

df['Score'] =  ((((df['Predicted Salary'] / df['Predicted Debt'])-mean1)/std1) + 
(((df['Undergraduate Enrollment'] /df['Predicted Job']) - mean2)/std2))

df.to_csv(r'C:\Users\jakem\CollegeImprovement-1\COLLEGEIMPROVEMENT\data\data_score.csv')



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

#Making Score Graphs for specific college
plt.figure(figsize=(14, 9))
df_grouped = df.groupby('University')['Score'].mean().reset_index()
df_top_10 = df_grouped.sort_values(by='Score', ascending=False).tail(10)
df_college = df[df['University'] == 'Duquesne University']
df_top_10 = pd.concat([df_top_10, df_college])

sns.barplot(x='University', y='Score', data=df_top_10, palette='viridis')

plt.title('Duquesne Score Compared to Bottom 10 Universities')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.xlabel('University')
plt.ylabel('Score')
plt.show()

#Getting Duquesne Rank out of total
df_grouped = df.groupby('University')['Score'].mean().reset_index()
duquesne_mean_score = df_grouped.loc[df_grouped['University'] == 'University of Pittsburgh-Pittsburgh Campus', 'Score'].values[0]
rank = (df_grouped['Score'] > duquesne_mean_score).sum() + 1  
print(f"University of Pittsburgh-Pittsburgh Campus has a mean 'Score' rank of: {rank} out of {len(df_grouped)} universities.")

df_grouped = df.groupby('University')['Score'].mean().reset_index()
duquesne_mean_score = df_grouped.loc[df_grouped['University'] == 'Carnegie Mellon University', 'Score'].values[0]
rank = (df_grouped['Score'] > duquesne_mean_score).sum() + 1  
print(f"Carnegie Mellon University has a mean 'Score' rank of: {rank} out of {len(df_grouped)} universities.")

df_grouped = df.groupby('University')['Score'].mean().reset_index()
duquesne_mean_score = df_grouped.loc[df_grouped['University'] == 'Duquesne University', 'Score'].values[0]
rank = (df_grouped['Score'] > duquesne_mean_score).sum() + 1  
print(f"Duquesne University has a mean 'Score' rank of: {rank} out of {len(df_grouped)} universities.")

#Seeing top 10
df_grouped = df.groupby('University')['Score'].mean().reset_index()
df_top_10 = df_grouped.sort_values(by='Score', ascending=False).tail(30)
print(df_top_10)

df


