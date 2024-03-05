import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def seaborn_scatter(data, xcol, ycol, title, xtitle, ytitle):
    """
    This function shows a seaborn scatterplot of two variables
    """
    sns.set(style="whitegrid")  

    sns.scatterplot(data=data, x=xcol, y=ycol, palette="Set2")

    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    from COLLEGEIMPROVEMENT import data_reader
    df = pd.read_csv(r'C:\Users\jakem\CollegeImprovement-1\COLLEGEIMPROVEMENT\data\ImputedData.csv')
    
    seaborn_scatter(df,df['AVGFACSAL'], df['MD_EARN_WNE_1YR'], 'Average Faculty vs. Salary of Students Post Graduation',
                    'Average Faculty Salary', 'Median Earnings of Students')




