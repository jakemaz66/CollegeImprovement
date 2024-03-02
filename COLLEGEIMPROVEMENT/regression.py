import pandas as pd
import numpy as np
from COLLEGEIMPROVEMENT import data_reader

#Reading in my dataframes
df1 = data_reader.data_reader()
df2 = data_reader.data_reader_study()

#Feature Engineering
df1['EXPENSES'] = (df1['BOOKSUPPLY'] + df1['ROOMBOARD_ON'] + df1['OTHEREXPENSE_ON'] + df1['ROOMBOARD_OFF'] + df1['OTHEREXPENSE_OFF'] + 
                  df1['OTHEREXPENSE_FAM'])

#





