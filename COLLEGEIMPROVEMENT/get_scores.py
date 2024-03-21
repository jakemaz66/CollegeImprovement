#Predicting Duquesne
from COLLEGEIMPROVEMENT import score_calc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def predict_uni_score(uni_name):

    df1 = pd.read_csv(r'COLLEGEIMPROVEMENT/data/data_score.csv')

    uni = df1[df1['University'] == uni_name]
    uni_score = uni['Score']
    uni_score = np.average(uni_score)

    print(f'The Score of {uni_name} is {uni_score}')

def top_10_unis(data):
 
 df1 = pd.read_csv(r'COLLEGEIMPROVEMENT/data/data_score.csv')
 
 df1 = data.copy()  
    
 df_mean = df1.groupby('University', as_index=False)['Score'].mean()

 df_mean_sorted = df_mean.sort_values(by='Score', ascending=False)

 df_top_10 = df_mean_sorted.head(10)
 

 sns.barplot(data=df_top_10, x='University', y='Score')
 plt.title("The Top 10 Universities by Mean Score")
 plt.xlabel("University")
 plt.ylabel("Mean Score")
 plt.xticks(rotation=45, ha='right')  
 plt.show()


if __name__ == '__main__':
    #predict_uni_score('Duquesne University')
    top_10_unis(score_calc.df)

    