import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def seaborn_scatter(data, xcol, ycol, title, xtitle, ytitle):
    """
    This function shows a seaborn scatterplot of two variables
    """
    #Setting Style
    sns.set(style="whitegrid")  

    sns.scatterplot(data=data, x=xcol, y=ycol, palette="Set2")

    plt.xlim(0, 40000)  
 

    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\jakem\CollegeImprovement-1\COLLEGEIMPROVEMENT\data\data_score.csv')
    
    #Testing Function
    seaborn_scatter(df,df['Faculty Salary'], df['Score'], 'Average Faculty Salary vs University Score',
                    'Average Faculty Salary', 'University Score')
    
    correlation_matrix = np.corrcoef(df['Faculty Salary'], df['Score'])
    correlation_coefficient = correlation_matrix[0, 1]
    print(f'Correlation: {correlation_coefficient}')




