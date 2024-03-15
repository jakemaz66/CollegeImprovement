#Predicting Duquesne
from COLLEGEIMPROVEMENT.models import random_forest
from COLLEGEIMPROVEMENT import score_calc
import numpy as np
import pandas as pd

rfr = random_forest.rfr
df1 = score_calc.df

def predict_uni_score(uni_name):

    uni = df1[df1['University'] == uni_name]
    uni_score = uni['Score']
    uni_score = np.average(uni_score)

    print(f'The Score of {uni_name} is {uni_score}')


if __name__ == '__main__':
    predict_uni_score('Duquesne University')

    