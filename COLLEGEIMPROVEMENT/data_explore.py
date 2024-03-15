import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def seaborn_scatter(data, xcol, ycol, title, xtitle, ytitle):
    """
    This function shows a seaborn scatterplot of two variables
    """
    #Setting Style
    sns.set(style="whitegrid")  

    sns.scatterplot(data=data, x=xcol, y=ycol, palette="Set2")

    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\jakem\CollegeImprovement-1\COLLEGEIMPROVEMENT\data\ImputedData.csv')
    
    #Testing Function
    seaborn_scatter(df,df['AVGFACSAL'], df['GRAD_DEBT_MDN_SUPP'], 'Average Faculty vs. Debt of Students',
                    'Average Faculty Salary', 'Average Debt of Students')




