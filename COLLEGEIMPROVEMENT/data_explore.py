import seaborn as sns
import matplotlib.pyplot as plt

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
    df = data_reader.data_reader()




